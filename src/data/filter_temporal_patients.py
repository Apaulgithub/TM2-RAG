"""
Phase 1B: Filter the master index to keep only patients with temporal data.

A patient has "temporal data" if they have >= MIN_STUDIES_PER_PATIENT studies.
These are the patients whose disease trajectories we can analyze.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DATA_DIR, MIN_STUDIES_PER_PATIENT


def filter_temporal_patients():
    """
    Read master index, keep only patients with multiple studies,
    and save as temporal_patients.json.
    """
    print("\n=== Phase 1B: Filtering Temporal Patients ===")

    # Load
    path = DATA_DIR / "master_index.json"
    with open(path, "r") as f:
        master = json.load(f)

    print(f"Loaded {len(master):,} patients")

    # Filter
    temporal = {
        pid: data
        for pid, data in master.items()
        if data["num_studies"] >= MIN_STUDIES_PER_PATIENT
    }

    # Save
    out = DATA_DIR / "temporal_patients.json"
    with open(out, "w") as f:
        json.dump(temporal, f, indent=2)

    # Stats
    counts = [v["num_studies"] for v in temporal.values()]
    total = sum(counts)

    print(f"\nTemporal patients: {len(temporal):,}")
    print(f"Retention: {len(temporal)/len(master)*100:.1f}%")
    print(f"Avg studies: {total/len(counts):.1f}")
    print(f"Saved -> {out}")

    # Distribution
    print("\nStudy distribution:")
    for lo, hi in [(2,2),(3,5),(6,10),(11,20),(21,50),(51,1000)]:
        c = sum(lo <= x <= hi for x in counts)
        if c:
            label = f"{lo}" if lo == hi else f"{lo}-{hi}"
            print(f"{label:>6}: {c:,}")

    return temporal


if __name__ == "__main__":
    filter_temporal_patients()
