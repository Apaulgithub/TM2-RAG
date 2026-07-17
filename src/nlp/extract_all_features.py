"""
Phase 2C: Extract All Features — Orchestrator that runs report parsing
and temporal change detection on all reports in the temporal patients dataset.

Input:  data/temporal_patients.json
Output: data/parsed_reports.json
"""

import json
import sys
from pathlib import Path
import tempfile
import csv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import DATA_DIR, PROJECT_ROOT, PATHOLOGIES
from src.nlp.report_parser import parse_report
from src.nlp.temporal_change_detector import detect_temporal_changes
from src.vision.image_classifier import classify_images_batch


CHEXBERT_CHECKPOINT = PROJECT_ROOT / "data" / "checkpoints" / "chexbert.pth"
METADATA_PATH = PROJECT_ROOT / "mimic-cxr-2.0.0-metadata.csv.gz"

# ─────────────────────────────────────────────
# CHECKPOINTS
# ─────────────────────────────────────────────

CHEXBERT_LABELS_CHECKPOINT = (
    DATA_DIR / "chexbert_labels.checkpoint.json"
)

PARSED_REPORTS_CHECKPOINT = (
    DATA_DIR / "parsed_reports.checkpoint.json"
)

SAVE_INTERVAL = 500


# ─────────────────────────────────────────────
# SAFE SAVE
# ─────────────────────────────────────────────

def safe_json_save(data, path: Path):

    tmp_path = path.with_suffix(".tmp")

    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)

    tmp_path.replace(path)


# ─────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────

_metadata_cache = None


def load_metadata():

    global _metadata_cache

    if _metadata_cache is not None:
        return _metadata_cache

    import pandas as pd

    print("Loading metadata...")

    df = pd.read_csv(METADATA_PATH)

    mapping = {}

    for _, row in df.iterrows():
        mapping[str(row["dicom_id"])] = str(
            row.get("ViewPosition", "")
        ).upper()

    _metadata_cache = mapping

    return mapping


def extract_dicom_id(path):
    return Path(path).stem


def split_by_view(image_paths, metadata):

    frontal = []
    lateral = []

    for p in image_paths:

        dicom = extract_dicom_id(p)
        view = metadata.get(dicom, "")

        if view in ("AP", "PA"):
            frontal.append(p)

        elif view in ("LATERAL", "LL"):
            lateral.append(p)

        else:
            frontal.append(p)

    return frontal, lateral


# ─────────────────────────────────────────────
# IMAGE AGGREGATION
# ─────────────────────────────────────────────

def avg_predictions(image_paths):

    if not image_paths:
        return None

    preds = classify_images_batch(image_paths)

    agg = {p: 0.0 for p in PATHOLOGIES}

    for pred in preds:
        for pathology in PATHOLOGIES:
            agg[pathology] += pred.pathology_probabilities[pathology]

    for pathology in agg:
        agg[pathology] /= max(len(preds), 1)

    return agg


def aggregate_view_predictions(image_paths, metadata):

    frontal, lateral = split_by_view(image_paths, metadata)

    frontal_pred = avg_predictions(frontal)
    lateral_pred = avg_predictions(lateral)

    if frontal_pred and lateral_pred:

        return {
            p: (frontal_pred[p] + lateral_pred[p]) / 2
            for p in PATHOLOGIES
        }

    elif frontal_pred:
        return frontal_pred

    elif lateral_pred:
        return lateral_pred

    else:
        return {}


def convert_probs_to_labels(prob_dict, threshold=0.5):

    return {
        p: 1 if prob_dict.get(p, 0.0) >= threshold else 0
        for p in PATHOLOGIES
    }


def merge_labels(text_labels, image_labels):

    merged = {}

    for p in PATHOLOGIES:

        text_value = text_labels.get(p, 0)
        image_value = image_labels.get(p, 0)

        # conservative multimodal fusion
        merged[p] = 1 if (text_value == 1 or image_value == 1) else 0

    return merged


# ─────────────────────────────────────────────
# CHEXBERT
# ─────────────────────────────────────────────

def run_chexbert(study_keys, texts):

    chexbert_labels = {}

    # ─────────────────────────────────────
    # RESUME FROM CHECKPOINT
    # ─────────────────────────────────────
    if CHEXBERT_LABELS_CHECKPOINT.exists():

        print("\n[Checkpoint] Loading CheXbert labels...")

        with open(CHEXBERT_LABELS_CHECKPOINT, "r") as f:
            chexbert_labels = json.load(f)

    completed = set(chexbert_labels.keys())

    remaining_keys = []
    remaining_texts = []

    for key, text in zip(study_keys, texts):

        if key not in completed:
            remaining_keys.append(key)
            remaining_texts.append(text)

    print(f"[CheXbert] Already completed: {len(completed)}")
    print(f"[CheXbert] Remaining: {len(remaining_texts)}")

    if not remaining_texts:
        return chexbert_labels

    # ─────────────────────────────────────
    # RUN CHEXBERT
    # ─────────────────────────────────────
    if CHEXBERT_CHECKPOINT.exists():

        from label import label
        import os

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False
        ) as tmp:

            writer = csv.writer(tmp)
            writer.writerow(["Report Impression"])

            for text in remaining_texts:
                writer.writerow([text])

            tmp_csv = tmp.name

        print("\nRunning CheXbert...")

        y_pred = label(str(CHEXBERT_CHECKPOINT), tmp_csv)

        for i, key in enumerate(remaining_keys):

            chexbert_labels[key] = {
                PATHOLOGIES[j]: int(y_pred[j][i])
                for j in range(len(PATHOLOGIES))
            }

            # checkpoint save
            if len(chexbert_labels) % SAVE_INTERVAL == 0:

                print(
                    f"[Checkpoint] Saving CheXbert labels "
                    f"({len(chexbert_labels)})"
                )

                safe_json_save(
                    chexbert_labels,
                    CHEXBERT_LABELS_CHECKPOINT
                )

        os.remove(tmp_csv)

    # final save
    safe_json_save(
        chexbert_labels,
        CHEXBERT_LABELS_CHECKPOINT
    )

    return chexbert_labels


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def extract_all_features():

    print("\n=== Phase 2: Extracting NLP + Image Features ===")

    metadata = load_metadata()

    input_path = DATA_DIR / "temporal_patients.json"

    with open(input_path, "r") as f:
        temporal_patients = json.load(f)

    study_keys = []
    texts = []
    study_meta = {}

    # ─────────────────────────────────────
    # COLLECT DATA
    # ─────────────────────────────────────
    for pid, pdata in temporal_patients.items():

        for study in pdata["studies"]:

            key = f"{pid}_{study['study_id']}"

            text = study.get("report_text", "")
            images = study.get("image_paths", [])

            study_keys.append(key)
            texts.append(text)

            study_meta[key] = {
                "images": images,
                "patient_id": pid,
            }

    # ─────────────────────────────────────
    # CHEXBERT
    # ─────────────────────────────────────
    chexbert_labels = run_chexbert(
        study_keys,
        texts
    )

    # ─────────────────────────────────────
    # LOAD PARSED REPORT CHECKPOINT
    # ─────────────────────────────────────
    parsed_reports = {}

    if PARSED_REPORTS_CHECKPOINT.exists():

        print("\n[Checkpoint] Loading parsed reports...")

        with open(PARSED_REPORTS_CHECKPOINT, "r") as f:
            parsed_reports = json.load(f)

    completed = set(parsed_reports.keys())

    print(f"[Parsed Reports] Already completed: {len(completed)}")

    # ─────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────
    print("\nProcessing reports with view-aware image signals...")

    for idx, key in enumerate(study_keys):

        # resume support
        if key in completed:
            continue

        report_text = texts[idx]

        text_labels = chexbert_labels.get(key, {})

        images = study_meta[key]["images"]

        image_probs = {}
        image_labels = {}

        # ─────────────────────────────────
        # IMAGE PROCESSING
        # ─────────────────────────────────
        if images:

            try:

                image_probs = aggregate_view_predictions(
                    images,
                    metadata
                )

                image_labels = convert_probs_to_labels(
                    image_probs
                )

            except Exception as e:

                print(f"[Image Error] {key}: {e}")

        # ─────────────────────────────────
        # MULTIMODAL LABEL FUSION
        # ─────────────────────────────────
        final_labels = merge_labels(
            text_labels,
            image_labels
        )

        # ─────────────────────────────────
        # REPORT PARSING
        # ─────────────────────────────────
        parsed = parse_report(
            key,
            report_text,
            final_labels
        )

        temporal = detect_temporal_changes(
            report_text,
            final_labels
        )

        # ─────────────────────────────────
        # BUILD RECORD
        # ─────────────────────────────────
        record = parsed.to_dict()

        record["temporal_analysis"] = temporal.to_dict()

        record["image_probabilities"] = image_probs
        record["image_labels"] = image_labels

        parsed_reports[key] = record

        # ─────────────────────────────────
        # CHECKPOINT SAVE
        # ─────────────────────────────────
        if len(parsed_reports) % SAVE_INTERVAL == 0:

            print(
                f"[Checkpoint] Saving parsed reports "
                f"({len(parsed_reports)})"
            )

            safe_json_save(
                parsed_reports,
                PARSED_REPORTS_CHECKPOINT
            )

        # progress log
        if (idx + 1) % 100 == 0:

            print(
                f"{idx+1:,}/{len(study_keys):,}"
            )

    # ─────────────────────────────────────
    # FINAL SAVE
    # ─────────────────────────────────────
    out_path = DATA_DIR / "parsed_reports.json"

    safe_json_save(
        parsed_reports,
        out_path
    )

    print(f"\nSaved -> {out_path}")

    return parsed_reports


# ─────────────────────────────────────────────

if __name__ == "__main__":
    extract_all_features()