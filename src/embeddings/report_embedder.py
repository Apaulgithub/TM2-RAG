"""
Phase 4A: Report Embedder — Generate dense vector embeddings for each
radiology report using BiomedCLIP's text encoder.

Uses augmented_text
Incorporates structured pathology signals
Falls back safely when missing
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DATA_DIR


# ─────────────────────────────────────────────
# TEXT PREPARATION
# ─────────────────────────────────────────────

def prepare_texts(parsed_reports):
    study_keys = []
    texts = []

    for study_key in sorted(parsed_reports.keys()):
        report = parsed_reports[study_key]

        # ─────────────────────────────────────
        # PRIORITY 1: augmented_text (BEST)
        # ─────────────────────────────────────
        text = report.get("augmented_text", "")

        # ─────────────────────────────────────
        # FALLBACK 1: findings + impression
        # ─────────────────────────────────────
        if not text.strip():
            findings = report.get("findings_text", "")
            impression = report.get("impression_text", "")
            text = f"{findings} {impression}".strip()

        # ─────────────────────────────────────
        # FALLBACK 2: raw text
        # ─────────────────────────────────────
        if not text.strip():
            text = report.get("raw_text", "")

        # ─────────────────────────────────────
        # FINAL fallback
        # ─────────────────────────────────────
        if not text.strip():
            text = "No report available."

        study_keys.append(study_key)
        texts.append(text)

    return study_keys, texts


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def embed_reports():
    """
    Embed all reports using BiomedCLIP.

    Input:
        data/parsed_reports.json

    Output:
        data/report_embeddings.npz
    """

    print("\n=== Phase 4A: Report Embeddings ===")

    # ─────────────────────────────────────
    # LOAD
    # ─────────────────────────────────────
    print("\n[1/4] Loading parsed reports...")
    with open(DATA_DIR / "parsed_reports.json", "r") as f:
        parsed_reports = json.load(f)

    print(f"  {len(parsed_reports):,} reports loaded")

    # ─────────────────────────────────────
    # PREPARE TEXT
    # ─────────────────────────────────────
    print("\n[2/4] Preparing text...")
    study_keys, texts = prepare_texts(parsed_reports)

    # Debug sample
    print("\nSample text preview:")
    for i in range(min(2, len(texts))):
        print(f"\n--- {study_keys[i]} ---")
        print(texts[i][:300], "...")

    # ─────────────────────────────────────
    # EMBEDDING
    # ─────────────────────────────────────
    print("\n[3/4] Encoding reports...")

    from src.fusion.claf_fusion import encode_text_batch

    batch_size = 64
    embeddings_list = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        emb = encode_text_batch(batch, device=None)
        embeddings_list.append(emb)

        if (i + batch_size) % 500 == 0 or i + batch_size >= len(texts):
            print(f"  {min(i + batch_size, len(texts)):,}/{len(texts):,}")

    # ─────────────────────────────────────
    # STACK + NORMALIZE
    # ─────────────────────────────────────
    embeddings = np.concatenate(embeddings_list, axis=0).astype(np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embeddings = embeddings / norms

    print(f"  Shape: {embeddings.shape}")

    # ─────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────
    output_path = DATA_DIR / "report_embeddings.npz"

    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        study_keys=np.array(study_keys, dtype=object),
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved -> {output_path} ({size_mb:.1f} MB)")

    # ─────────────────────────────────────
    # SANITY CHECK
    # ─────────────────────────────────────
    if len(embeddings) >= 5:
        print("\nSanity check (first 5 cosine sims):")
        sim = embeddings[:5] @ embeddings[:5].T
        for i in range(5):
            row = " ".join(f"{sim[i,j]:.3f}" for j in range(5))
            print(f"[{row}]  {study_keys[i]}")

    return embeddings, study_keys


if __name__ == "__main__":
    embed_reports()