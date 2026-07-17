"""
Auxiliary Image Classifier (DenseNet121 - torchxrayvision)

Uses torchxrayvision DenseNet
Produces pathology probabilities
Batch + single inference
Robust error handling
No fragile view logic
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    PATHOLOGIES,
    DETECTION_THRESHOLD,
    IMAGE_SIZE,
    VISION_MODEL_NAME,
)


# ─────────────────────────────────────────────
# DATA STRUCTURE
# ─────────────────────────────────────────────

@dataclass
class ImagePrediction:
    image_path: str
    pathology_probabilities: Dict[str, float]
    detected_pathologies: List[str]

    top_prediction: Optional[str] = None
    model_name: str = VISION_MODEL_NAME

    def to_dict(self):
        return {
            "image_path": self.image_path,
            "pathology_probabilities": {
                k: round(v, 4) for k, v in self.pathology_probabilities.items()
            },
            "detected_pathologies": self.detected_pathologies,
            "top_prediction": self.top_prediction,
            "model_name": self.model_name,
        }


# ─────────────────────────────────────────────
# MODEL LOADING (CACHED)
# ─────────────────────────────────────────────

_model = None
_transforms = None
_device = None


def load_model():
    global _model, _transforms, _device

    if _model is not None:
        return _model, _transforms, _device

    import torch
    import torchxrayvision as xrv
    import torchvision.transforms as T

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Vision] Loading model: {VISION_MODEL_NAME}")

    model = xrv.models.DenseNet(weights=VISION_MODEL_NAME)
    model = model.to(_device)
    model.eval()

    transforms = T.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(IMAGE_SIZE),
    ])

    _model = model
    _transforms = transforms

    return _model, _transforms, _device


def unload_model():
    """Release the cached DenseNet model so the VLM can use GPU memory."""
    global _model, _transforms, _device

    _model = None
    _transforms = None
    _device = None

    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ─────────────────────────────────────────────
# PATHOLOGY MAPPING
# ─────────────────────────────────────────────

_TXV_TO_CHEXPERT = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum",
    "Fracture": "Fracture",
    "Lung Lesion": "Lung Lesion",
    "Lung Opacity": "Lung Opacity",
    "No Finding": "No Finding",
    "Pleural Effusion": "Pleural Effusion",
    "Pleural Other": "Pleural Other",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
    "Support Devices": "Support Devices",
}


# ─────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────

def preprocess_image(image_path: str, transforms):
    import torch
    import torchxrayvision as xrv
    from PIL import Image

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(image_path)

    img = Image.open(path).convert("L")
    img = np.array(img, dtype=np.float32)

    if img.max() > 0:
        img = img / img.max() * 255.0

    img = xrv.datasets.normalize(img, maxval=255, reshape=True)
    img = transforms(img)

    img = torch.from_numpy(img)

    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:
        img = img.unsqueeze(0)

    return img


# ─────────────────────────────────────────────
# CORE INFERENCE
# ─────────────────────────────────────────────

def _predict_batch(tensor_batch, model, device):
    import torch

    tensor_batch = tensor_batch.to(device)

    with torch.no_grad():
        logits = model(tensor_batch)
        probs = torch.sigmoid(logits).cpu().numpy()

    return probs


def _map_pathologies(probs, model):
    pathology_probs = {}

    for i, name in enumerate(model.pathologies):
        mapped = _TXV_TO_CHEXPERT.get(name)
        if mapped in PATHOLOGIES:
            # clamp for safety
            pathology_probs[mapped] = float(np.clip(probs[i], 0.0, 1.0))

    # ensure full coverage
    for p in PATHOLOGIES:
        pathology_probs.setdefault(p, 0.0)

    return pathology_probs


# ─────────────────────────────────────────────
# SINGLE IMAGE
# ─────────────────────────────────────────────

def classify_image(
    image_path: str,
    threshold: float = DETECTION_THRESHOLD,
):
    import torch

    model, transforms, device = load_model()

    tensor = preprocess_image(image_path, transforms).to(device)

    probs = _predict_batch(tensor, model, device)[0]

    pathology_probs = _map_pathologies(probs, model)

    detected = [
        p for p, v in pathology_probs.items()
        if v >= threshold and p != "No Finding"
    ]

    top_idx = int(np.argmax(probs))
    top_label = model.pathologies[top_idx]

    return ImagePrediction(
        image_path=image_path,
        pathology_probabilities=pathology_probs,
        detected_pathologies=detected,
        top_prediction=top_label,
    )


# ─────────────────────────────────────────────
# BATCH INFERENCE
# ─────────────────────────────────────────────

def classify_images_batch(
    image_paths: List[str],
    threshold: float = DETECTION_THRESHOLD,
    batch_size: int = 64,
):
    import torch

    model, transforms, device = load_model()

    results = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]

        tensors = []
        valid_paths = []

        for p in batch_paths:
            try:
                tensors.append(preprocess_image(p, transforms))
                valid_paths.append(p)
            except Exception as e:
                print(f"[WARN] Skipping image {p}: {e}")

        if not tensors:
            continue

        batch = torch.cat(tensors, dim=0)

        probs_batch = _predict_batch(batch, model, device)

        for j, path in enumerate(valid_paths):
            probs = probs_batch[j]

            pathology_probs = _map_pathologies(probs, model)

            detected = [
                p for p, v in pathology_probs.items()
                if v >= threshold and p != "No Finding"
            ]

            top_idx = int(np.argmax(probs))
            top_label = model.pathologies[top_idx]

            results.append(ImagePrediction(
                image_path=path,
                pathology_probabilities=pathology_probs,
                detected_pathologies=detected,
                top_prediction=top_label,
            ))

    return results


# ─────────────────────────────────────────────
# NOTE
# ─────────────────────────────────────────────

"""
NOTE:
- This module intentionally does NOT handle view (AP/Lateral)
- View handling is done using metadata in the main pipeline
- This module only extracts pathology signals
"""
