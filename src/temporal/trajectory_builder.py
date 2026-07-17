"""
Phase 3A: Trajectory Builder — Construct temporal patient trajectories
from parsed reports.

Build temporal trajectories using:
merged pathology labels (text + image)
image-derived signals 
continuous severity score 
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DATA_DIR, PATHOLOGIES


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class StudyState:
    """
    A single snapshot of a patient's condition at one study.
    """

    study_id: str
    order_index: int

    # core labels
    pathology_labels: Dict[str, int]

    # temporal signals
    temporal_status: str
    temporal_confidence: float
    per_pathology_trends: Dict[str, str]

    # text
    findings_text: str
    impression_text: str

    # metadata
    has_comparison: bool
    num_positive_findings: int = 0
    num_images: int = 0

    # NEW: image-aware signals
    image_labels: Dict[str, int] = field(default_factory=dict)
    image_probabilities: Dict[str, float] = field(default_factory=dict)
    avg_pathology_score: float = 0.0

    def to_dict(self):
        return {
            "study_id": self.study_id,
            "order_index": self.order_index,
            "pathology_labels": self.pathology_labels,
            "temporal_status": self.temporal_status,
            "temporal_confidence": self.temporal_confidence,
            "per_pathology_trends": self.per_pathology_trends,
            "findings_text": self.findings_text,
            "impression_text": self.impression_text,
            "has_comparison": self.has_comparison,
            "num_positive_findings": self.num_positive_findings,
            "num_images": self.num_images,
            "image_labels": self.image_labels,
            "image_probabilities": self.image_probabilities,
            "avg_pathology_score": round(self.avg_pathology_score, 4),
        }


@dataclass
class Trajectory:
    """
    A patient's complete temporal trajectory.
    """

    patient_id: str
    states: List[StudyState]

    transitions: List[Dict] = field(default_factory=list)
    overall_trend: str = "unknown"
    trend_confidence: float = 0.0
    duration_studies: int = 0

    def to_dict(self):
        return {
            "patient_id": self.patient_id,
            "states": [s.to_dict() for s in self.states],
            "transitions": self.transitions,
            "overall_trend": self.overall_trend,
            "trend_confidence": round(self.trend_confidence, 3),
            "duration_studies": self.duration_studies,
        }


# ─────────────────────────────────────────────
# BUILD SINGLE STATE
# ─────────────────────────────────────────────

def build_study_state(patient_id: str, study_data: Dict, parsed_reports: Dict) -> StudyState:

    study_id = study_data["study_id"]
    study_key = f"{patient_id}_{study_id}"

    parsed = parsed_reports.get(study_key)
    if not parsed:
        return None

    # ─────────────────────────────────────
    # LABELS
    # ─────────────────────────────────────
    pathology_labels = parsed.get("pathology_labels", {p: 0 for p in PATHOLOGIES})
    image_labels = parsed.get("image_labels", {})
    image_probs = parsed.get("image_probabilities", {})

    # ─────────────────────────────────────
    # TEMPORAL
    # ─────────────────────────────────────
    temporal = parsed.get("temporal_analysis", {})

    # ─────────────────────────────────────
    # POSITIVE COUNT (exclude "No Finding")
    # ─────────────────────────────────────
    num_positive = sum(
        1 for k, v in pathology_labels.items()
        if v == 1 and k != "No Finding"
    )

    # ─────────────────────────────────────
    # CONTINUOUS SEVERITY SCORE
    # ─────────────────────────────────────
    if image_probs:
        avg_score = sum(image_probs.values()) / len(image_probs)
    else:
        avg_score = num_positive / max(len(PATHOLOGIES), 1)

    # ─────────────────────────────────────
    # BUILD STATE
    # ─────────────────────────────────────
    return StudyState(
        study_id=study_id,
        order_index=study_data["order_index"],

        pathology_labels=pathology_labels,

        temporal_status=temporal.get("temporal_status", "unknown"),
        temporal_confidence=temporal.get("confidence", 0.0),
        per_pathology_trends=temporal.get("per_pathology_trends", {}),

        findings_text=parsed.get("findings_text", ""),
        impression_text=parsed.get("impression_text", ""),

        has_comparison=temporal.get("has_comparison", False),
        num_positive_findings=num_positive,
        num_images=study_data.get("num_images", 0),

        image_labels=image_labels,
        image_probabilities=image_probs,
        avg_pathology_score=avg_score,
    )


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def build_trajectories():

    print("\n=== Phase 3A: Building Patient Trajectories ===")

    # ─────────────────────────────────────
    # LOAD DATA
    # ─────────────────────────────────────
    print("\n[1/3] Loading data...")

    with open(DATA_DIR / "temporal_patients.json", "r") as f:
        temporal_patients = json.load(f)

    with open(DATA_DIR / "parsed_reports.json", "r") as f:
        parsed_reports = json.load(f)

    print(f"{len(temporal_patients):,} patients | {len(parsed_reports):,} reports")

    # ─────────────────────────────────────
    # BUILD TRAJECTORIES
    # ─────────────────────────────────────
    print("\n[2/3] Building trajectories...")

    trajectories = {}

    for i, (patient_id, pdata) in enumerate(temporal_patients.items()):

        states = []

        for study_data in pdata["studies"]:
            state = build_study_state(patient_id, study_data, parsed_reports)
            if state:
                states.append(state)

        # ensure temporal ordering
        states.sort(key=lambda s: s.order_index)

        trajectories[patient_id] = Trajectory(
            patient_id=patient_id,
            states=states,
            duration_studies=len(states),
        )

        if (i + 1) % 500 == 0:
            print(f"Built {i+1:,}/{len(temporal_patients):,} trajectories")

    # ─────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────
    print("\n[3/3] Saving...")

    output_path = DATA_DIR / "trajectories.json"

    with open(output_path, "w") as f:
        json.dump(
            {pid: t.to_dict() for pid, t in trajectories.items()},
            f,
            indent=2
        )

    print(f"Saved → {output_path}")

    # ─────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────
    lengths = [t.duration_studies for t in trajectories.values()]

    print("\n=== Summary ===")
    print(f"Total trajectories: {len(trajectories):,}")
    print(f"Total states: {sum(lengths):,}")
    print(f"Avg length: {sum(lengths)/len(lengths):.1f}")
    print(f"Max length: {max(lengths)}")
    print(f"Median length: {sorted(lengths)[len(lengths)//2]}")

    return trajectories


if __name__ == "__main__":
    build_trajectories()