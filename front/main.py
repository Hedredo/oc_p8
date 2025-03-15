import httpx
import pandas as pd
import gradio as gr
import numpy as np
from dotenv import load_dotenv
import json
import base64
from io import BytesIO
import os
from PIL import Image
from pathlib import Path
from constants import IMAGE_FOLDER
from utils_and_constants import labels

# Récupérer le chemin du dossier contenant les images
cwd = Path(__file__).parent
folder_path = cwd / IMAGE_FOLDER
filenames = os.listdir(folder_path)
ids = sorted(set(filename.split("_")[2] for filename in filenames))

# Récupére le dictionnaire de mapping id: category
id2category = {label.categoryId: label.category for label in labels}

# Récupére la variable d'environnement API_HOST
load_dotenv()
API_HOST = os.getenv("API_HOST")

# Fonction pour charger le masque et l'image associés à un id
def get_mask_and_image(id):
    mask, image = sorted(folder_path / filename for filename in filenames if id in filename and "color" not in filename)
    return image, mask

# Liste des fichiers d'exemple disponibles et renvoie un dictionnaire id: path
example_files = { 
    id: get_mask_and_image(id) for id in ids # TODO: remplacer la clé par l'id de l'image
}

# Fonction pour charger l'image sélectionnée ou le GT en exemple
def load_example(id):
    return example_files[id][0]

def load_gt(id):
    return example_files[id][1]

# Fonction pour encoder une image en base64
def encode_image(image_path):
    img = Image.open(image_path).convert("RGB")  # Convertir en RGB si nécessaire
    buffer = BytesIO()
    img.save(buffer, format="PNG", optimize=True)  # Compression sans perte
    encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")  # Convertir en string base64
    return encoded_string

# Fonction pour construire le JSON à envoyer à l'API
def get_payload(image_path, mask_path):
    # Construire le JSON
    payload = {
        "image_base64": encode_image(image_path),
        "mask_base64": encode_image(mask_path)
    }
    return payload

# Fonction pour segmenter une image utilisée au déclenchement du bouton
async def segment_image(id):
    # Charger l'image et le masque
    image_path, mask_path = get_mask_and_image(id)

    # Construire le JSON
    payload = get_payload(image_path, mask_path)

    # Envoyer la requête POST en mode asynchrone
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(f"{API_HOST}:8000/predict/", json=payload)
        response.raise_for_status()  # Lève une exception si le code HTTP n'est pas 200
    
    try:
        results = response.json()
    except json.JSONDecodeError:
        return {"error": "La réponse de l'API n'est pas un JSON valide"}
        # Vérifier que la réponse contient bien des données attendues
    if not results:
        return {"error": "Réponse vide de l'API"}
    # Crée les listes pour stocker les annotations, les prédictions, les classes et les scores
    mean_score = results[0]
    classes = []
    perclass_score = []
    gt_annotations = []
    pred_annotations = []
    for i, result in enumerate(results[1:]):
        # Décoder la première image reçue
        mask_base64 = result["mask"]  # Récupère l'image encodée en base641
        mask_bytes = base64.b64decode(mask_base64)  # Décodage base64 → bytes
        mask = Image.open(BytesIO(mask_bytes))  # Ouvrir l'image avec PIL
        gt_annotations.append((np.array(mask).astype(bool).astype(np.uint8), result["class"]))
        # Décoder la première image reçue
        pred_base64 = result["prediction"]  # Récupère l'image encodée en base641
        pred_bytes = base64.b64decode(pred_base64)  # Décodage base64 → bytes
        pred = Image.open(BytesIO(pred_bytes))  # Ouvrir l'image avec PIL
        pred_annotations.append((np.array(pred).astype(bool).astype(np.uint8), result["class"]))
        perclass_score.append(result["iou"])
        classes.append(result["class"])
    pil_image = Image.open(image_path)
    df = pd.DataFrame({"class": classes, "iou": perclass_score})
    return (pil_image, gt_annotations), (pil_image, pred_annotations), (mean_score), (df)

# Gradio code pour l'interface utilisateur
with gr.Blocks() as demo:
    gr.Markdown("# Image Segmentation avec Liste Déroulante")
    with gr.Row():
        dropdown = gr.Dropdown(
            label="Choisissez un exemple",
            choices=list(example_files.keys()),
            value=list(example_files.keys())[0],  # Valeur par défaut
        )
    
    with gr.Row():
        image_input = gr.Image(value=list(example_files.values())[0][0], type="filepath", label="Image à segmenter")
        gt_output = gr.AnnotatedImage(label="gt blending")
        pred_output = gr.AnnotatedImage(label="pred blending")

        # Charger l'exemple sélectionné
        dropdown.change(fn=load_example, inputs=dropdown, outputs=image_input)
    with gr.Row():
        with gr.Column():
            score_output = gr.Textbox(label="Score moyen", type="text")
        with gr.Column():
            df_output = gr.BarPlot(
                label="Scores par classe",
                y="class",
                x="iou",
            )

    with gr.Row():
            # Bouton pour lancer la segmentation
            submit_btn = gr.Button("Segmenter")
            submit_btn.click(segment_image, inputs=dropdown, outputs=[gt_output, pred_output, score_output, df_output])

if __name__ == "__main__":
    demo.launch(share=False, allowed_paths=[IMAGE_FOLDER], server_name="0.0.0.0", server_port=7860)