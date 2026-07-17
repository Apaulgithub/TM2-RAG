"""
Dynamic Temporal Windowing — the core TD-RAG innovation adapted for
clinical radiology time-series.

We adapt this for MIMIC-CXR:
  • A "measurement" = one radiology study (report + images).
  • A "window"      = W consecutive studies for the same patient.
  • A "step"        = S studies forward (overlap = W - S).

A patient with N studies produces max(1, N - W + 1) windows when S = 1.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    DATA_DIR, PATHOLOGIES,
    WINDOW_SIZE, WINDOW_STEP, RECENCY_MAX_WINDOWS
)


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class TemporalWindow:
    """
    One temporal window — a contiguous slice of a patient's trajectory.
    """
    window_key: str            # e.g. "p10000032_w0_2"
    patient_id: str
    start_idx: int             # index of first study in trajectory
    end_idx: int               # index of last study (inclusive)
    states: list               # the W study-state dicts in order
    window_size: int           # actual number of studies in this window

    # Computed within the window
    local_trend: str           # improving / worsening / stable / mixed
    pathology_deltas: dict     # net change per pathology across window
    serialised_text: str       # structured text representation for embedding

    def to_dict(self):
        return {
            "window_key": self.window_key,
            "patient_id": self.patient_id,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "window_size": self.window_size,
            "local_trend": self.local_trend,
            "pathology_deltas": self.pathology_deltas,
            "study_ids": [s["study_id"] for s in self.states],
            "serialised_text": self.serialised_text,
        }


def compute_local_trend(states):
    score = 0

    for s in states:
        trends = s.get("per_pathology_trends", {})
        score += len([t for t in trends.values() if t == "worsening"])
        score -= len([t for t in trends.values() if t == "improving"])

    if score > 0:
        return "worsening"
    elif score < 0:
        return "improving"
    else:
        return "stable"


def compute_pathology_deltas(states):
    first = states[0].get("pathology_labels", {})
    last = states[-1].get("pathology_labels", {})

    deltas = {}
    for p in PATHOLOGIES:
        prev = max(first.get(p, 0), 0)
        curr = max(last.get(p, 0), 0)
        deltas[p] = curr - prev

    return deltas



def serialise_window(states, trend):
    """
    Convert a window of study states into a structured text template
    preserving temporal ordering and pathology labels.
    """
    lines = [f"[Window: {len(states)} studies | trend: {trend}]"]

    for i, s in enumerate(states):
        study_id = s["study_id"]
        labels = s.get("pathology_labels", {})

        positives = [p for p, v in labels.items() if v == 1 and p != "No Finding"]
        text = s.get("impression_text") or s.get("findings_text") or "(no report)"

        if len(text) > 200:
            text = text[:200] + "..."

        if positives:
            p_str = ", ".join(positives)
        elif labels.get("No Finding", 0) == 1:
            p_str = "No acute finding"
        else:
            p_str = "Unremarkable"

        lines.append(f"Study {i+1} [{study_id}]: {p_str}. {text}")

    lines.append(f"[Trend: {trend}]")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# WINDOW CREATION
# ─────────────────────────────────────────────

def create_windows(trajectory, window_size=WINDOW_SIZE, step=WINDOW_STEP):
    """
    Slide a window of `window_size` consecutive studies across a patient
    trajectory with `window_step` offset, producing overlapping sub-trajectory
    windows.
    """
    patient_id = trajectory["patient_id"]
    states = trajectory["states"]
    n = len(states)

    if n == 0:
        return []

    windows = []

    # Single window case
    if n < window_size:
        trend = compute_local_trend(states)
        deltas = compute_pathology_deltas(states)
        text = serialise_window(states, trend)

        windows.append(
            TemporalWindow(
                window_key=f"{patient_id}_w0_{n-1}",
                patient_id=patient_id,
                start_idx=0,
                end_idx=n - 1,
                states=states,
                window_size=n,
                local_trend=trend,
                pathology_deltas=deltas,
                serialised_text=text,
            )
        )
        return windows

    # Sliding windows
    start = 0
    while start + window_size <= n:
        end = start + window_size - 1
        window_states = states[start:end + 1]

        trend = compute_local_trend(window_states)
        deltas = compute_pathology_deltas(window_states)
        text = serialise_window(window_states, trend)

        windows.append(
            TemporalWindow(
                window_key=f"{patient_id}_w{start}_{end}",
                patient_id=patient_id,
                start_idx=start,
                end_idx=end,
                states=window_states,
                window_size=window_size,
                local_trend=trend,
                pathology_deltas=deltas,
                serialised_text=text,
            )
        )

        start += step

    return windows


# ──────────────────────────────────────────────
# Build all windows
# ──────────────────────────────────────────────

def build_all_windows():
    """
    Phase 3C: Create dynamic temporal windows for every trajectory.

    Input:  data/trajectories.json
    Output: data/temporal_windows.json
    """
    print("\n=== Phase 3C: Building Temporal Windows ===")

    with open(DATA_DIR / "trajectories.json", "r") as f:
        trajectories = json.load(f)

    all_windows = {}
    total = 0

    for i, (pid, traj) in enumerate(trajectories.items()):
        wins = create_windows(traj)

        if RECENCY_MAX_WINDOWS:
            wins = wins[-RECENCY_MAX_WINDOWS:]

        for w in wins:
            all_windows[w.window_key] = w.to_dict()

        total += len(wins)

        if (i + 1) % 500 == 0:
            print(f"{i+1:,} patients → {total:,} windows")

    out = DATA_DIR / "temporal_windows.json"
    with open(out, "w") as f:
        json.dump(all_windows, f, indent=2)

    print(f"\nSaved -> {out}")
    print(f"Total windows: {total:,}")

    return all_windows


if __name__ == "__main__":
    build_all_windows()
