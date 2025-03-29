import os
import json
import base64
from io import BytesIO
from pathlib import Path

import httpx
import pandas as pd
import gradio as gr
import numpy as np
from PIL import Image
from dotenv import load_dotenv

from utils_and_constants import labels, IMAGE_FOLDER
from typing import Tuple, Dict, Any, List

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
def get_mask_and_image(id: str) -> Tuple[Path, Path]:
    """
    Charge le chemin de l'image et du masque correspondant à l'identifiant donné.

    Args:
        id (str): L'identifiant de l'exemple.

    Returns:
        Tuple[Path, Path]: Le chemin de l'image et le chemin du masque.
    """
    # On trie les fichiers correspondant à l'id et on exclut ceux contenant le mot "color"
    mask, image = sorted(folder_path / filename for filename in filenames if id in filename and "color" not in filename)
    return image, mask

# Dictionnaire des fichiers d'exemple disponibles : id -> (chemin de l'image, chemin du masque)
example_files: Dict[str, Tuple[Path, Path]] = { 
    id: get_mask_and_image(id) for id in ids
}

# Fonction pour charger l'image d'exemple correspondante à un id
def load_example(id: str) -> Path:
    """
    Charge l'image d'exemple pour l'identifiant donné.

    Args:
        id (str): L'identifiant de l'exemple.

    Returns:
        Path: Le chemin de l'image d'exemple.
    """
    return example_files[id][0]

# Fonction pour charger le masque ground-truth (GT) correspondant à un id
def load_gt(id: str) -> Path:
    """
    Charge le masque GT pour l'identifiant donné.

    Args:
        id (str): L'identifiant de l'exemple.

    Returns:
        Path: Le chemin du masque GT.
    """
    return example_files[id][1]

# Fonction pour encoder une image en base64
def encode_image(image_path: Path) -> str:
    """
    Encode une image en base64.

    Args:
        image_path (Path): Le chemin de l'image.

    Returns:
        str: L'image encodée en base64.
    """
    img: Image.Image = Image.open(image_path).convert("RGB")  # Convertir en RGB si nécessaire
    buffer = BytesIO()
    img.save(buffer, format="PNG", optimize=True)  # Compression sans perte
    encoded_string: str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded_string

# Fonction pour construire le payload JSON à envoyer à l'API
def get_payload(image_path: Path, mask_path: Path) -> Dict[str, str]:
    """
    Construit le JSON avec les images encodées en base64.

    Args:
        image_path (Path): Le chemin de l'image.
        mask_path (Path): Le chemin du masque.

    Returns:
        Dict[str, str]: Dictionnaire contenant l'image et le masque encodés.
    """
    payload: Dict[str, str] = {
        "image_base64": encode_image(image_path),
        "mask_base64": encode_image(mask_path)
    }
    return payload

# Fonction asynchrone pour lancer la segmentation d'image via l'API
async def segment_image(id: str) -> Tuple[
    Tuple[Image.Image, List[Tuple[np.ndarray, Any]]],
    Tuple[Image.Image, List[Tuple[np.ndarray, Any]]],
    float,
    pd.DataFrame
]:
    """
    Segmente l'image et le masque associés à l'identifiant donné en appelant l'API.
    
    Args:
        id (str): L'identifiant de l'exemple.
    
    Returns:
        Tuple:
            - Tuple contenant l'image PIL et la liste des annotations GT (masque numpy et classe).
            - Tuple contenant l'image PIL et la liste des annotations de prédiction (masque numpy et classe).
            - Score moyen (float).
            - DataFrame des scores par classe.
    """
    # Charger l'image et le masque d'exemple
    image_path, mask_path = get_mask_and_image(id)

    # Construire le payload JSON
    payload: Dict[str, str] = get_payload(image_path, mask_path)

    # Envoyer la requête POST de façon asynchrone
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(f"{API_HOST}/predict/", json=payload)
        response.raise_for_status()  # Lève une exception en cas d'erreur HTTP
    
    try:
        results = response.json()
    except json.JSONDecodeError:
        return {"error": "La réponse de l'API n'est pas un JSON valide"}
    
    if not results:
        return {"error": "Réponse vide de l'API"}

    # Préparation des variables pour stocker les résultats
    mean_score: float = results[0]
    classes: List[Any] = []
    perclass_score: List[Any] = []
    gt_annotations: List[Tuple[np.ndarray, Any]] = []
    pred_annotations: List[Tuple[np.ndarray, Any]] = []

    # Parcourir les résultats pour décoder les images et récupérer les données
    for result in results[1:]:
        # Décoder et traiter le masque GT
        mask_base64: str = result["mask"]
        mask_bytes: bytes = base64.b64decode(mask_base64)
        mask_img: Image.Image = Image.open(BytesIO(mask_bytes))
        gt_annotations.append((np.array(mask_img).astype(bool).astype(np.uint8), result["class"]))
        
        # Décoder et traiter la prédiction
        pred_base64: str = result["prediction"]
        pred_bytes: bytes = base64.b64decode(pred_base64)
        pred_img: Image.Image = Image.open(BytesIO(pred_bytes))
        pred_annotations.append((np.array(pred_img).astype(bool).astype(np.uint8), result["class"]))
        
        perclass_score.append(result["iou"])
        classes.append(result["class"])

    pil_image: Image.Image = Image.open(image_path)
    df: pd.DataFrame = pd.DataFrame({"class": classes, "iou": perclass_score})
    
    return (pil_image, gt_annotations), (pil_image, pred_annotations), mean_score, df

# Gradio code pour l'interface utilisateur
with gr.Blocks() as demo:
    gr.Markdown("# Segmentation d'image pour véhicule autonome")
    with gr.Row():
        dropdown = gr.Dropdown(
            label="Choisissez un label dans la liste déroulante",
            choices=list(example_files.keys()),
            value=list(example_files.keys())[0],  # Valeur par défaut
        )
    
    with gr.Row():
        image_input = gr.Image(value=list(example_files.values())[0][0], type="filepath", label="Image à segmenter")
        gt_output = gr.AnnotatedImage(label="Ground Truth")
        pred_output = gr.AnnotatedImage(label="Prediction")

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