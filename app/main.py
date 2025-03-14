from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import segmentation_models as sm
import numpy as np
import shutil
import os
import uuid
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import load_model
from utils_and_constants import labels, TARGET_SIZE
from data import ImageSegmentationDataset, DiceFocalLoss


app = FastAPI()

# Dossier temporaire pour stocker les images
cwd = Path(__file__).parent
TEMP_DIR = cwd / "temp_images"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# constants.py
model_path = cwd / "model" / "best_model.keras"

model = load_model(
    model_path,
        custom_objects={
            "DiceFocalLoss": DiceFocalLoss,
            "MeanIoU": sm.metrics.IOUScore,
            "Dice": sm.metrics.FScore,
            **{
                f"IoU_class_{i}": sm.metrics.IOUScore(class_indexes=i) for i in range(8)
            },
        },
    )

BACKBONE = "resnet50"

class ImagePayload(BaseModel):
    image_base64: str
    mask_base64: str

def load_data(paths, labels):
    data_gen = ImageSegmentationDataset(
        paths=paths,
        labels=labels,
        batch_size=1,
        target_size=TARGET_SIZE,
        augmentations=False,
        normalize=BACKBONE,
        shuffle=False,
        label_onehot=True,
    )
    return data_gen

# Fonction pour sauvegarder un fichier temporairement et retourner son path
def save_temp_file(upload_file: UploadFile) -> Path:
    file_extension = Path(upload_file.filename).suffix  # Récupérer l'extension (.png, .jpg, ...)
    temp_filename = f"{uuid.uuid4()}{file_extension}"  # Générer un nom unique
    temp_path = TEMP_DIR / temp_filename

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return temp_path

def decode_base64_to_image(base64_str: str) -> Image.Image:
    image_bytes = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_bytes))

# Convertir en Base64
def img_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")  # PNG garde les valeurs intactes
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def serialize_predictions(predictions, mask, class_mapping, iou_perclass_score, target_size=(1024, 2048)):
    """
    Convertit les prédictions et le mask en dictionnaires avec images en Base64.
    
    :param predictions: np.array de shape (256, 512, 8)
    :param mask: np.array de shape (256, 512, 8)
    :return: Liste de dictionnaires avec encodage Base64.
    """
    results = []
    
    for class_idx in range(predictions.shape[-1]):  # 8 classes
        gt = Image.fromarray(mask[..., class_idx].astype(np.uint8) * 255)  # Convertir en image
        gt_resized = gt.resize(target_size[::-1], Image.NEAREST)  # Resize avec NEAREST
        pred = Image.fromarray(predictions[..., class_idx].astype(np.uint8) * 255)
        pred_resized = pred.resize(target_size[::-1], Image.NEAREST)

        results.append({
            "class": class_mapping[class_idx],
            "iou": iou_perclass_score[class_idx],
            "prediction": img_to_base64(pred_resized),
            "mask": img_to_base64(gt_resized),
        })
    
    return results

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict/")
async def predict(payload: ImagePayload):
    try:
        # Décoder et sauvegarder temporairement les images
        image = decode_base64_to_image(payload.image_base64)
        mask = decode_base64_to_image(payload.mask_base64)
        
        image_path = TEMP_DIR / f"image_{uuid.uuid4()}.png"
        mask_path = TEMP_DIR / f"mask_{uuid.uuid4()}.png"
        
        image.save(image_path)
        mask.save(mask_path)

        # Charger les données
        data_generator = load_data([(image_path, mask_path)], labels)
        pred, mask = data_generator.get_prediction(model=model, onehot=True, ground_truth=True)

        # Convertir les prédictions en one-hot encoding
        pred_classes = np.argmax(pred, axis=-1)
        num_classes = pred.shape[-1]
        pred_one_hot = np.eye(num_classes)[pred_classes]

        id2category = {label.categoryId: label.category for label in labels}
        iou_mean_score = {"Mean_IoU": round(float(sm.metrics.IOUScore(class_indexes=[*range(8)])([mask], [pred]).numpy()), 4)}
        iou_perclass_score = {i: round(float(sm.metrics.IOUScore(class_indexes=[i])([mask], [pred]).numpy()), 4) for i in range(8)}

        # Sérialiser les résultats
        results = serialize_predictions(pred_one_hot, mask, id2category, iou_perclass_score, target_size=(1024, 2048))
        results.insert(0, iou_mean_score)

        # Supprimer les fichiers temporaires
        os.remove(image_path)
        os.remove(mask_path)

        return results
    except Exception as e:
        return {"error": str(e)}
