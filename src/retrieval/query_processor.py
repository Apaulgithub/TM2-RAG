"""
Phase 5A: Query Processor — Process an incoming query for retrieval.

Supports THREE query modes:
  1. text_only  — user provides report text (no image)
  2. image_only — user provides X-ray image (no text)
  3. both       — user provides text + image -> CLAF (BiomedCLIP) fusion
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.nlp.report_parser import parse_report, labels_to_text
from src.nlp.temporal_change_detector import detect_temporal_changes
from src.fusion.claf_fusion import (
    encode_text_batch,
    encode_image_batch,
    fuse_embeddings,
)
from src.vision.image_classifier import classify_images_batch
from src.config import PATHOLOGIES


# ─────────────────────────────────────────────
# DATA STRUCTURE
# ─────────────────────────────────────────────

@dataclass
class ProcessedQuery:
    raw_text: str
    study_key: str
    query_mode: str

    findings_text: str = ""
    impression_text: str = ""
    pathology_labels: Dict[str, int] = None
    temporal_status: str = "unknown"

    report_embedding: Optional[np.ndarray] = None
    window_query: Optional[np.ndarray] = None
    trajectory_query: Optional[np.ndarray] = None

    image_embedding: Optional[np.ndarray] = None
    fused_query: Optional[np.ndarray] = None

    image_path: Optional[str] = None


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def convert_probs_to_labels(prob_dict, threshold=0.5):
    return {
        p: 1 if prob_dict.get(p, 0.0) >= threshold else 0
        for p in PATHOLOGIES
    }


def merge_labels(text_labels, image_labels):
    return {
        p: 1 if (text_labels.get(p, 0) == 1 or image_labels.get(p, 0) == 1) else 0
        for p in PATHOLOGIES
    }


def build_augmented_query(report_text, labels):
    structured = labels_to_text(labels)
    return f"{report_text}\n\n[Structured Findings]\n{structured}"


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def process_query(
    report_text: str = None,
    image_path: str = None,
    patient_history: Optional[List[Dict]] = None,
    study_key: str = "query",
) -> ProcessedQuery:

    has_text = report_text and report_text.strip()
    has_image = image_path is not None

    if has_text and has_image:
        mode = "both"
    elif has_text:
        mode = "text_only"
    elif has_image:
        mode = "image_only"
    else:
        raise ValueError("Provide text or image")

    findings = ""
    impression = ""
    labels = {}
    temporal_status = "unknown"

    report_emb = None
    window_emb = None
    traj_emb = None
    image_emb = None
    fused = None

    # ─────────────────────────────────────
    # IMAGE → LABELS (NEW)
    # ─────────────────────────────────────
    image_labels = {}
    image_probs = {}

    if has_image:
        try:
            preds = classify_images_batch([image_path])
            probs = preds[0].pathology_probabilities
            image_probs = probs
            image_labels = convert_probs_to_labels(probs)
        except Exception as e:
            print(f"[Image Error] {e}")

    # ─────────────────────────────────────
    # TEXT PIPELINE
    # ─────────────────────────────────────
    if has_text:

        parsed = parse_report(study_key, report_text, {})
        temporal = detect_temporal_changes(report_text, parsed.pathology_labels)

        findings = parsed.findings_text
        impression = parsed.impression_text

        text_labels = parsed.pathology_labels

        # MERGE TEXT + IMAGE LABELS
        labels = merge_labels(text_labels, image_labels)

        temporal_status = temporal.temporal_status

        # AUGMENTED TEXT (CRITICAL)
        augmented_text = build_augmented_query(report_text, labels)

        # ── REPORT EMBEDDING
        report_emb = encode_text_batch([augmented_text])[0]

        # ── WINDOW QUERY (514 dim)
        trend_map = {"improving": 1.0, "stable": 0.0, "worsening": -1.0}
        trend_val = trend_map.get(temporal_status, 0.0)
        intensity_val = 0.5  # placeholder for query-time intensity

        window_emb = np.concatenate([
            report_emb,
            np.array([trend_val, intensity_val], dtype=np.float32)
        ])
        window_emb /= (np.linalg.norm(window_emb) + 1e-8)

        # ── TRAJECTORY QUERY (520 dim)
        # Using 8 features to match TRAJECTORY_EMBEDDING_DIM
        traj_feat = np.array([
            0.1,        # duration placeholder
            trend_val,  # trend
            0.0, 0.0,   # transitions
            0.0, 0.0,   # findings
            0.5, 0.0    # severity mean/delta placeholder
        ], dtype=np.float32)

        traj_emb = np.concatenate([report_emb, traj_feat])
        traj_emb /= (np.linalg.norm(traj_emb) + 1e-8)

    # ─────────────────────────────────────
    # IMAGE EMBEDDING
    # ─────────────────────────────────────
    if has_image:
        image_emb = encode_image_batch([image_path])[0]

    # ─────────────────────────────────────
    # FUSION
    # ─────────────────────────────────────
    if mode == "both":
        fused = fuse_embeddings(report_emb, image_emb)
    elif mode == "text_only":
        fused = report_emb
    elif mode == "image_only":
        fused = image_emb

    return ProcessedQuery(
        raw_text=report_text or "",
        study_key=study_key,
        query_mode=mode,
        findings_text=findings,
        impression_text=impression,
        pathology_labels=labels,
        temporal_status=temporal_status,
        report_embedding=report_emb,
        window_query=window_emb,
        trajectory_query=traj_emb,
        image_embedding=image_emb,
        fused_query=fused,
        image_path=image_path,
    )