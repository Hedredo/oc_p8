from fastapi import FastAPI, UploadFile, File
import segmentation_models as sm
import shutil
import os
import uuid
from pathlib import Path
from tensorflow.keras.models import load_model
from utils_and_constants import labels, TARGET_SIZE
from data import ImageSegmentationDataset, DiceFocalLoss

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Désactive CUDA pour éviter les erreurs avec AMD


app = FastAPI()

# Dossier temporaire pour stocker les images
TEMP_DIR = Path("./temp_images")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# constants.py
model_path = "./artifacts/best_model.keras"

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

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict/")
async def predict(image_file: UploadFile = File(...), mask_file: UploadFile = File(...)):
    # Sauvegarder les fichiers reçus temporairement
    image_path = save_temp_file(image_file)
    mask_path = save_temp_file(mask_file)

    # 🔥 Passer les paths au DataGenerator ou pipeline de prédiction
    data_generator = load_data([(image_path, mask_path)], labels)
    pred, mask = data_generator.get_prediction(model=model, onehot=True, ground_truth=True)

    # Optionnel : supprimer les fichiers après utilisation
    os.remove(image_path)
    os.remove(mask_path)

    return {"message": ["Prediction successful"]}
