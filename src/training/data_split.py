"""
Patient-level dataset splitting for T-RAG

Creates a deterministic 70/20/10 split at PATIENT level
to avoid leakage across temporal studies.

Stratified by number of studies per patient.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    DATA_DIR,
    MIN_STUDIES_PER_PATIENT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    SPLIT_SEED,
)

OUTPUT_PATH = DATA_DIR / "splits.json"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_bin(n):
    """Group patients by temporal richness."""
    if n < 15:
        return "10-14"
    elif n < 20:
        return "15-19"
    elif n < 30:
        return "20-29"
    elif n < 50:
        return "30-49"
    else:
        return "50+"


def compute_stats(pids, patients):
    total_studies = sum(len(patients[p]["studies"]) for p in pids)
    avg = total_studies / len(pids) if pids else 0

    return {
        "patients": len(pids),
        "studies": total_studies,
        "avg_studies_per_patient": round(avg, 1),
    }


# ─────────────────────────────────────────────
# MAIN SPLIT FUNCTION
# ─────────────────────────────────────────────

def create_splits(force=False):
    if OUTPUT_PATH.exists() and not force:
        print("splits.json exists -> loading")
        with open(OUTPUT_PATH) as f:
            return json.load(f)

    print("\n=== Creating Patient-Level Splits ===")

    # ── Load patients ───────────────────────
    with open(DATA_DIR / "temporal_patients.json") as f:
        all_patients = json.load(f)

    # filter
    patients = {
        pid: p for pid, p in all_patients.items()
        if len(p["studies"]) >= MIN_STUDIES_PER_PATIENT
    }

    print(f"Patients (≥{MIN_STUDIES_PER_PATIENT} studies): {len(patients):,}")

    # ── Stratify by study count ─────────────
    bins = defaultdict(list)

    for pid, pdata in patients.items():
        b = get_bin(len(pdata["studies"]))
        bins[b].append(pid)

    print("\nStudy-count bins:")
    for b in sorted(bins.keys()):
        print(f"  {b}: {len(bins[b]):,}")

    rng = np.random.RandomState(SPLIT_SEED)

    train, val, test = [], [], []

    # ── Split per bin ───────────────────────
    for b, group in bins.items():
        rng.shuffle(group)

        n = len(group)
        n_train = int(n * TRAIN_SPLIT)
        n_val = int(n * VAL_SPLIT)

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    # shuffle final splits
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    # ── Sanity checks ───────────────────────
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))

    # ── Stats ──────────────────────────────
    train_stats = compute_stats(train, patients)
    val_stats = compute_stats(val, patients)
    test_stats = compute_stats(test, patients)

    print("\nSplit summary:")
    for name, stats in [
        ("train", train_stats),
        ("val", val_stats),
        ("test", test_stats),
    ]:
        print(
            f"{name:6} | {stats['patients']:5} patients | "
            f"{stats['studies']:7} studies | "
            f"{stats['avg_studies_per_patient']:4} avg"
        )

    # ── Save ───────────────────────────────
    result = {
        "train": train,
        "val": val,
        "test": test,
        "metadata": {
            "total_patients": len(patients),
            "seed": SPLIT_SEED,
            "min_studies": MIN_STUDIES_PER_PATIENT,
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        }
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved -> {OUTPUT_PATH}")
    return result


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

def load_splits():
    if not OUTPUT_PATH.exists():
        return create_splits()
    with open(OUTPUT_PATH) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# GET STUDY KEYS
# ─────────────────────────────────────────────

def get_study_keys_for_split(split):
    splits = load_splits()
    pids = set(splits[split])

    with open(DATA_DIR / "temporal_patients.json") as f:
        patients = json.load(f)

    keys = []
    for pid in pids:
        pdata = patients.get(pid)
        if not pdata:
            continue

        for s in pdata["studies"]:
            keys.append(f"{pid}_{s['study_id']}")

    return keys


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    create_splits(force=args.force)