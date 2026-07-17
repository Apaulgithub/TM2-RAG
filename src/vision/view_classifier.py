"""
View Classifier (FINAL VERSION)

✔ Image-based (no embedding dependency)
✔ Uses SAME DenseNet backbone as image_classifier
✔ Lightweight trained head for AP vs LV
✔ Works for single + batch inference
✔ Ready for fusion + chatbot pipeline
"""

import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import CHECKPOINT_DIR
from src.vision.image_classifier import load_model, preprocess_image


# ─────────────────────────────────────────────
# MODEL HEAD (TRAINED SEPARATELY)
# ─────────────────────────────────────────────

class ViewClassifierHead(nn.Module):
    """
    Small MLP head for AP vs LV classification
    Input: DenseNet feature vector (1024-d)
    Output: 2 classes (AP=0, LV=1)
    """

    def __init__(self, in_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# MAIN CLASSIFIER
# ─────────────────────────────────────────────

class ViewClassifier:

    def __init__(self, checkpoint: Optional[Path] = None):
        """
        Load backbone + trained head
        """

        # Load shared DenseNet backbone
        self.model, self.transforms, self.device = load_model()

        # Load head
        self.head = ViewClassifierHead().to(self.device)

        ckpt_path = checkpoint or (CHECKPOINT_DIR / "view_classifier.pt")

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"View classifier checkpoint not found: {ckpt_path}"
            )

        state = torch.load(ckpt_path, map_location=self.device)
        self.head.load_state_dict(state)
        self.head.eval()

        print("[ViewClassifier] Loaded successfully")

    # ─────────────────────────────────────────
    # FEATURE EXTRACTION
    # ─────────────────────────────────────────

    def _extract_features(self, img_tensor: torch.Tensor):
        """
        Extract DenseNet features (1024-d)
        """
        with torch.no_grad():
            feats = self.model.features(img_tensor)
            feats = torch.relu(feats)
            feats = feats.mean(dim=[2, 3])  # global average pooling

        return feats

    # ─────────────────────────────────────────
    # SINGLE IMAGE
    # ─────────────────────────────────────────

    def predict(self, image_path: str) -> str:
        """
        Predict view for a single image
        Returns: "ap" or "lv"
        """

        img = preprocess_image(image_path, self.transforms).to(self.device)

        feats = self._extract_features(img)

        with torch.no_grad():
            logits = self.head(feats)
            pred = logits.argmax(dim=1).item()

        return "ap" if pred == 0 else "lv"

    # ─────────────────────────────────────────
    # BATCH
    # ─────────────────────────────────────────

    def predict_batch(self, image_paths: List[str]) -> List[str]:
        """
        Batch prediction for efficiency
        """

        tensors = []
        valid_paths = []

        for p in image_paths:
            try:
                tensors.append(preprocess_image(p, self.transforms))
                valid_paths.append(p)
            except Exception as e:
                print(f"[WARN] Skipping {p}: {e}")

        if not tensors:
            return []

        batch = torch.cat(tensors, dim=0).to(self.device)

        feats = self._extract_features(batch)

        with torch.no_grad():
            logits = self.head(feats)
            preds = logits.argmax(dim=1).cpu().numpy()

        return ["ap" if p == 0 else "lv" for p in preds]


# ─────────────────────────────────────────────
# UTILITY (OPTIONAL)
# ─────────────────────────────────────────────

def classify_view(image_path: str) -> str:
    """
    Quick helper function
    """
    clf = ViewClassifier()
    return clf.predict(image_path)