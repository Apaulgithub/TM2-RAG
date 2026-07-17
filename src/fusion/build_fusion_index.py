"""
Build BiomedCLIP Fused Vector Store (FINAL VERSION)

Uses ViewClassifier (no metadata dependency)
Supports multi-view (AP + LV) fusion
Uses weighted fusion (AP > LV)
Safe checkpointing for large datasets
Clean and modular
"""

import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    DATA_DIR,
    IMAGES_DIR,
    CLAF_DIM,
    CLAF_IMAGE_WEIGHT,
    CLAF_TEXT_WEIGHT,
    MIN_STUDIES_PER_PATIENT,
    AP_VIEW_WEIGHT,
    LV_VIEW_WEIGHT,
)

from src.embeddings.vector_store import VectorStore
from src.fusion.claf_fusion import (
    encode_text_batch,
    encode_image_batch,
)

from src.vision.view_classifier import ViewClassifier


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CHECKPOINT_PATH = DATA_DIR / "fusion_checkpoint.npz"
IMAGE_BATCH_SIZE = 128
CHECKPOINT_EVERY = 1000


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def resolve_image_path(raw_path):
    if not raw_path:
        return ""

    p = Path(raw_path)

    if p.exists():
        return str(p)

    alt = IMAGES_DIR / p.name
    if alt.exists():
        return str(alt)

    return ""


def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return None

    data = np.load(CHECKPOINT_PATH, allow_pickle=True)

    study_keys = data["study_keys"].tolist()
    text_emb = data["text_embeddings"]
    image_emb = data["image_embeddings"]

    done = set(
        i for i in range(len(image_emb))
        if np.linalg.norm(image_emb[i]) > 0
    )

    print(f"Resuming checkpoint: {len(done)} studies processed")

    return study_keys, text_emb, image_emb, done


def save_checkpoint(keys, text_emb, image_emb):
    np.savez_compressed(
        CHECKPOINT_PATH,
        study_keys=np.array(keys, dtype=object),
        text_embeddings=text_emb,
        image_embeddings=image_emb,
    )
    print("Checkpoint saved.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def build_fusion_index():
    print("\n=== Phase 7: Build Fused Vector Store (View-Aware) ===")

    # ── 1. LOAD TEXT ─────────────────────────
    print("\n[1/5] Loading reports...")

    with open(DATA_DIR / "parsed_reports.json") as f:
        parsed_reports = json.load(f)

    with open(DATA_DIR / "temporal_patients.json") as f:
        patients = json.load(f)

    patients = {
        pid: p for pid, p in patients.items()
        if len(p["studies"]) >= MIN_STUDIES_PER_PATIENT
    }

    study_keys = []
    texts = []

    for pid, pdata in patients.items():
        for s in pdata["studies"]:
            sid = str(s["study_id"])
            key = f"{pid}_{sid}"

            report = parsed_reports.get(key, {})

            findings = report.get("findings_text", "")
            impression = report.get("impression_text", "")

            text = f"{findings} {impression}".strip()

            if not text:
                text = report.get("raw_text", "No report available.")

            study_keys.append(key)
            texts.append(text)

    print(f"Total studies: {len(study_keys):,}")

    # ── 2. LOAD IMAGE PATHS ───────────────────
    print("\n[2/5] Loading image paths...")

    with open(DATA_DIR / "view_index.json") as f:
        view_index = json.load(f)

    key_to_imgs = {}

    for key in study_keys:
        sid = key.split("_")[1]
        info = view_index.get(sid, {})

        paths = []

        for p in info.get("ap_paths", []):
            resolved = resolve_image_path(p)
            if resolved:
                paths.append(resolved)
                break

        for p in info.get("lv_paths", []):
            resolved = resolve_image_path(p)
            if resolved:
                paths.append(resolved)
                break

        if paths:
            key_to_imgs[key] = paths

    print(f"Studies with images: {len(key_to_imgs):,}")

    # ── 3. TEXT EMBEDDINGS ───────────────────
    print("\n[3/5] Encoding text...")

    ckpt = load_checkpoint()

    if ckpt and ckpt[0] == study_keys:
        study_keys, text_emb, image_emb, done = ckpt
    else:
        text_emb = encode_text_batch(texts, batch_size=256)
        image_emb = np.zeros((len(study_keys), CLAF_DIM), dtype=np.float32)
        done = set()

    print(f"Text embeddings: {text_emb.shape}")

    # ── 4. IMAGE EMBEDDINGS ───────────────────
    print("\n[4/5] Encoding images with view-aware fusion...")

    print("[View] Loading classifier...")
    view_clf = ViewClassifier()

    for i, key in enumerate(study_keys):

        if i in done:
            continue

        paths = key_to_imgs.get(key, [])

        if not paths:
            continue

        try:
            # Encode images
            embs = encode_image_batch(paths, batch_size=IMAGE_BATCH_SIZE)

            # Predict views
            views = view_clf.predict_batch(paths)

            # Assign weights
            weights = [
                AP_VIEW_WEIGHT if v == "ap" else LV_VIEW_WEIGHT
                for v in views
            ]

            weights = np.array(weights, dtype=np.float32)

            # Normalize weights
            weights = weights / weights.sum()

            # Weighted fusion
            img_emb = np.sum(embs * weights[:, None], axis=0)

            image_emb[i] = img_emb
            done.add(i)

        except Exception as e:
            print(f"[WARN] Failed image encoding: {key} ({e})")

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(study_keys, text_emb, image_emb)

        if (i + 1) % 500 == 0:
            print(f"{i+1:,}/{len(study_keys):,}")

    save_checkpoint(study_keys, text_emb, image_emb)

    # ── 5. FUSION ────────────────────────────
    print("\n[5/5] Fusing embeddings...")

    fused = []

    for i in range(len(study_keys)):
        t = text_emb[i]
        im = image_emb[i]

        has_t = np.linalg.norm(t) > 0
        has_i = np.linalg.norm(im) > 0

        if has_t and has_i:
            vec = CLAF_IMAGE_WEIGHT * im + CLAF_TEXT_WEIGHT * t
        elif has_t:
            vec = t
        elif has_i:
            vec = im
        else:
            vec = np.zeros(CLAF_DIM, dtype=np.float32)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        fused.append(vec)

    fused = np.vstack(fused)

    # ── SAVE ────────────────────────────────
    store = VectorStore("fused_reports", CLAF_DIM)
    store.add_batch(study_keys, fused)
    store.save(DATA_DIR / "vector_store")

    print("\nFused vector store built successfully.")

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    return store


# ─────────────────────────────────────────────

if __name__ == "__main__":
    build_fusion_index()