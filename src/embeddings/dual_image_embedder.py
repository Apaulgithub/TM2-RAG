"""
Phase 5: Dual Image Embedder — Offline feature extraction for a dual-view
architecture.

Extracts two separate embeddings per study:
    - AP embedding: torchxrayvision DenseNet121 (1024-dim, CXR-specialized)
    - LV embedding: ImageNet ResNet50            (2048-dim)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchxrayvision as xrv
from PIL import Image
from torchvision.models import (
    ResNet50_Weights,
    resnet50,
)
from tqdm import tqdm

# ── project root on path ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    DATA_DIR,
    IMAGES_DIR,
    AP_EMBEDDING_DIM,
    LV_EMBEDDING_DIM,
    MIN_STUDIES_PER_PATIENT,
    IMAGE_SIZE,
    VISION_MODEL_NAME,
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

CHECKPOINT_PATH = DATA_DIR / "dual_image_embeddings_checkpoint.npz"
FINAL_PATH = DATA_DIR / "dual_image_embeddings.npz"
CHECKPOINT_EVERY = 1000
BATCH_SIZE = 64


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────

def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading models on {device}...")

    model_ap = xrv.models.DenseNet(weights=VISION_MODEL_NAME)
    model_ap.eval().to(device)

    model_lv = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model_lv.fc = torch.nn.Identity()
    model_lv.eval().to(device)

    return model_ap, model_lv, device


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

lv_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def resolve_path(p):
    if not p:
        return ""
    p = Path(p)

    if p.exists():
        return str(p)

    alt = IMAGES_DIR / p.name
    if alt.exists():
        return str(alt)

    return ""


def preprocess_ap(path):
    path = resolve_path(path)
    if not path:
        return None

    try:
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32)

        if arr.max() > 0:
            arr = (arr / arr.max()) * 255

        arr = xrv.datasets.normalize(arr, maxval=255, reshape=True)
        arr = xrv.datasets.XRayResizer(IMAGE_SIZE)(arr)

        return torch.from_numpy(arr).unsqueeze(0)

    except Exception:
        return None


def preprocess_lv(path):
    path = resolve_path(path)
    if not path:
        return None

    try:
        img = Image.open(path).convert("RGB")
        return lv_transform(img).unsqueeze(0)
    except Exception:
        return None


# ─────────────────────────────────────────────
# BATCH EXTRACTION
# ─────────────────────────────────────────────

def extract_batch(model, device, paths, preprocess_fn, dim):
    results = np.zeros((len(paths), dim), dtype=np.float32)

    tensors, indices = [], []

    for i, p in enumerate(paths):
        t = preprocess_fn(p)
        if t is not None:
            tensors.append(t)
            indices.append(i)

    if not tensors:
        return results

    batch = torch.cat(tensors).to(device)

    with torch.no_grad():
        feats = model.features(batch) if hasattr(model, "features") else model(batch)

        if feats.ndim > 2:
            feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)

        vecs = feats.cpu().numpy()

    # normalize
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)

    for i, idx in enumerate(indices):
        results[idx] = vecs[i]

    return results


# ─────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────

def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return [], np.empty((0, AP_EMBEDDING_DIM)), np.empty((0, LV_EMBEDDING_DIM))

    data = np.load(CHECKPOINT_PATH, allow_pickle=True)

    print(f"Resuming from checkpoint ({len(data['study_keys'])} done)")

    return (
        list(data["study_keys"]),
        data["ap_features"],
        data["lv_features"],
    )


def save_checkpoint(keys, ap_feats, lv_feats):
    np.savez_compressed(
        CHECKPOINT_PATH,
        study_keys=np.array(keys, dtype=object),
        ap_features=ap_feats,
        lv_features=lv_feats,
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def embed_dual_images():
    print("\n=== Phase 5: Dual Image Embeddings ===")

    with open(DATA_DIR / "temporal_patients.json") as f:
        patients = json.load(f)

    with open(DATA_DIR / "view_index.json") as f:
        view_index = json.load(f)

    patients = {
        pid: p for pid, p in patients.items()
        if len(p["studies"]) >= MIN_STUDIES_PER_PATIENT
    }

    study_keys, ap_paths, lv_paths = [], [], []

    for pid, pdata in patients.items():
        for s in pdata["studies"]:
            sid = str(s["study_id"])
            key = f"{pid}_{sid}"

            info = view_index.get(sid, {})
            ap = (info.get("ap_paths") or [""])[0]
            lv = (info.get("lv_paths") or [""])[0]

            study_keys.append(key)
            ap_paths.append(ap)
            lv_paths.append(lv)

    print(f"Total studies: {len(study_keys):,}")

    # ── checkpoint load
    done_keys, done_ap, done_lv = load_checkpoint()
    done_set = set(done_keys)

    # filter remaining
    todo = [
        (k, ap, lv)
        for k, ap, lv in zip(study_keys, ap_paths, lv_paths)
        if k not in done_set
    ]

    print(f"Remaining: {len(todo):,}")

    if not todo:
        print("All done already.")
        return

    model_ap, model_lv, device = load_models()

    new_keys, new_ap, new_lv = [], [], []

    for i in tqdm(range(0, len(todo), BATCH_SIZE)):
        batch = todo[i:i + BATCH_SIZE]

        keys = [b[0] for b in batch]
        ap_batch = [b[1] for b in batch]
        lv_batch = [b[2] for b in batch]

        ap_vec = extract_batch(model_ap, device, ap_batch, preprocess_ap, AP_EMBEDDING_DIM)
        lv_vec = extract_batch(model_lv, device, lv_batch, preprocess_lv, LV_EMBEDDING_DIM)

        new_keys.extend(keys)
        new_ap.append(ap_vec)
        new_lv.append(lv_vec)

        # checkpoint
        if len(new_keys) % CHECKPOINT_EVERY < BATCH_SIZE:
            combined_keys = done_keys + new_keys
            combined_ap = np.vstack([done_ap] + new_ap)
            combined_lv = np.vstack([done_lv] + new_lv)

            save_checkpoint(combined_keys, combined_ap, combined_lv)
            print(f"Checkpoint saved ({len(combined_keys)})")

    # final save
    final_keys = done_keys + new_keys
    final_ap = np.vstack([done_ap] + new_ap)
    final_lv = np.vstack([done_lv] + new_lv)

    np.savez_compressed(
        FINAL_PATH,
        study_keys=np.array(final_keys, dtype=object),
        ap_features=final_ap,
        lv_features=final_lv,
    )

    print(f"\nSaved → {FINAL_PATH}")
    print(f"AP: {final_ap.shape} | LV: {final_lv.shape}")

    # remove checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()


if __name__ == "__main__":
    embed_dual_images()