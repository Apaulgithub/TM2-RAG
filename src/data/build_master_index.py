"""
Phase 1A: Build a master index of all patients, studies, reports, and images.

Each patient entry contains their studies in chronological order (using
study_id as a proxy for temporal ordering), with paths to reports and images.
"""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    IMAGES_DIR, REPORTS_DIR,
    STUDY_LIST_CSV, RECORD_LIST_CSV,
    DATA_DIR, MAX_STUDIES,
)

# CSV LOADERS
def load_study_list(path):
    """
    Parse cxr-study-list.csv to get (subject_id, study_id, report_path)
    mappings for all studies.
    """
    data = defaultdict(list)

    with open(path, "r") as f:
        for row in csv.DictReader(f):
            sid = int(row["subject_id"])
            stid = int(row["study_id"])
            data[sid].append((stid, row["path"]))

    return data


def load_record_list(path):
    """
    Parse cxr-record-list.csv to get (subject_id, study_id, dicom_id, image_path)
    mappings for all images.
    """
    data = defaultdict(list)

    with open(path, "r") as f:
        for row in csv.DictReader(f):
            key = (int(row["subject_id"]), int(row["study_id"]))
            data[key].append(row["dicom_id"])

    return data

# HELPERS
def read_report(path):
    try:
        text = Path(path).read_text(encoding="utf-8")
        return " ".join(text.split())
    except Exception:
        return ""


def resolve_report_path(rel_path):
    if rel_path.startswith("files/"):
        rel_path = rel_path[len("files/"):]
    return REPORTS_DIR / rel_path


def resolve_image_paths(subject_id, study_id, dicom_ids):
    prefix = f"p{str(subject_id)[:2]}"
    base = IMAGES_DIR / prefix / f"p{subject_id}" / f"s{study_id}"

    paths = []
    for d in dicom_ids:
        p = base / f"{d}.jpg"
        if p.exists():
            paths.append(str(p))
    return paths


# MAIN
def build_master_index():
    """
    Main function: Build the master patient-study index.

    For each patient:
    1. Gather all their studies from the CSV
    2. Sort studies by study_id (ascending) as temporal proxy
    3. Read the report text
    4. Find corresponding image paths
    5. Assemble into a structured dict

    Returns:
        dict: The master index, keyed by patient_id string
    """
    print("\n=== Phase 1A: Building Master Index ===")

    # Load metadata
    patient_studies = load_study_list(STUDY_LIST_CSV)
    study_images = load_record_list(RECORD_LIST_CSV)

    print(f"Patients: {len(patient_studies):,}")
    print(f"Studies: {sum(len(v) for v in patient_studies.values()):,}")

    # Optional subset
    if MAX_STUDIES:
        all_studies = [
            (sid, stid, path)
            for sid, studies in patient_studies.items()
            for stid, path in studies
        ]
        all_studies.sort()
        all_studies = all_studies[:MAX_STUDIES]

        patient_studies = defaultdict(list)
        for sid, stid, path in all_studies:
            patient_studies[sid].append((stid, path))

        print(f"[Subset] Patients: {len(patient_studies):,}")

    # Build index
    master_index = {}

    for i, (sid, studies) in enumerate(sorted(patient_studies.items())):
        studies = sorted(studies)

        entries = []
        for order, (stid, rel_path) in enumerate(studies):
            report_path = resolve_report_path(rel_path)
            report_text = read_report(report_path)

            dicoms = study_images.get((sid, stid), [])
            images = resolve_image_paths(sid, stid, dicoms)

            entries.append({
                "study_id": f"s{stid}",
                "study_id_num": stid,
                "order_index": order,
                "report_path": str(report_path),
                "report_text": report_text,
                "image_paths": images,
                "num_images": len(images),
            })

        master_index[f"p{sid}"] = {
            "subject_id": sid,
            "num_studies": len(entries),
            "studies": entries,
        }

        if (i + 1) % 500 == 0:
            print(f"Processed {i+1:,} patients")

    # Save
    out = DATA_DIR / "master_index.json"
    with open(out, "w") as f:
        json.dump(master_index, f, indent=2)

    print(f"\nSaved -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")
    return master_index


if __name__ == "__main__":
    build_master_index()
