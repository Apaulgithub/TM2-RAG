"""
CLAF — BiomedCLIP-Aligned Fusion

Both text and images are encoded into a shared 512-dim space where
cosine similarity is semantically meaningful across modalities.

Three query modes:
    text_only  → BiomedCLIP text encoder  → 512-dim
    image_only → BiomedCLIP image encoder → 512-dim
    both       → weighted average         → 512-dim
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    BIOMEDCLIP_MODEL_NAME, CLAF_DIM,
    CLAF_IMAGE_WEIGHT, CLAF_TEXT_WEIGHT
)


# ─────────────────────────────────────────────
# GLOBAL CACHE
# ─────────────────────────────────────────────

_model = None
_preprocess = None
_tokenizer = None
_device = None


# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────

def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────

def load_biomedclip(device: Optional[str] = None):
    global _model, _preprocess, _tokenizer, _device

    device = device or get_device()

    if _model is not None and _device == device:
        return _model, _preprocess, _tokenizer

    import open_clip
    import torch

    print(f"Loading BiomedCLIP on {device}...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        BIOMEDCLIP_MODEL_NAME
    )
    tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_MODEL_NAME)

    model = model.to(device)
    model.eval()

    _model = model
    _preprocess = preprocess
    _tokenizer = tokenizer
    _device = device

    print(f"Loaded (dim={CLAF_DIM})")

    return model, preprocess, tokenizer


def unload_biomedclip():
    """Release cached BiomedCLIP objects so Qwen can use GPU memory."""
    global _model, _preprocess, _tokenizer, _device

    _model = None
    _preprocess = None
    _tokenizer = None
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
# TEXT ENCODING
# ─────────────────────────────────────────────

def encode_text_batch(
    texts,
    batch_size: int = 256,
    device: Optional[str] = None,
    gpu_ids: Optional[list[int]] = None,
):
    import torch

    device = device or get_device()
    model, _, tokenizer = load_biomedclip(device)

    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        tokens = tokenizer(batch).to(device)

        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        embeddings.append(feats.cpu().numpy())

    return np.vstack(embeddings).astype(np.float32)


# ─────────────────────────────────────────────
# IMAGE ENCODING
# ─────────────────────────────────────────────

def encode_image_batch(
    image_paths,
    batch_size: int = 128,
    device: Optional[str] = None,
    gpu_ids: Optional[list[int]] = None,
):
    import torch
    from PIL import Image

    device = device or get_device()
    model, preprocess, _ = load_biomedclip(device)

    embeddings = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]

        tensors = []
        valid_idx = []

        for j, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(preprocess(img))
                valid_idx.append(j)
            except Exception:
                pass

        batch_emb = np.zeros((len(batch_paths), CLAF_DIM), dtype=np.float32)

        if tensors:
            batch_tensor = torch.stack(tensors).to(device)

            with torch.no_grad():
                feats = model.encode_image(batch_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)

            feats = feats.cpu().numpy()

            for k, idx in enumerate(valid_idx):
                batch_emb[idx] = feats[k]

        embeddings.append(batch_emb)

    return np.vstack(embeddings).astype(np.float32)


# ─────────────────────────────────────────────
# FUSION
# ─────────────────────────────────────────────

def fuse_embeddings(text_emb, image_emb):
    """
    Weighted fusion (image-dominant)
    """

    has_t = text_emb is not None and np.linalg.norm(text_emb) > 0
    has_i = image_emb is not None and np.linalg.norm(image_emb) > 0

    if has_t and has_i:
        vec = (
            CLAF_IMAGE_WEIGHT * image_emb +
            CLAF_TEXT_WEIGHT * text_emb
        )
    elif has_t:
        vec = text_emb
    elif has_i:
        vec = image_emb
    else:
        return np.zeros(CLAF_DIM, dtype=np.float32)

    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec
