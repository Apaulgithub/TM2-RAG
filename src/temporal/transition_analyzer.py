"""
Phase 3B: Transition Analyzer — Analyze transitions between consecutive
study states within each patient trajectory.

For each pair of consecutive studies (S_t → S_{t+1}), we compute:
1. Which pathologies appeared (new finding)
2. Which pathologies disappeared (resolved)
3. Which pathologies persisted
4. The overall transition direction (improving/worsening/stable)

Then we aggregate transitions across the full trajectory to determine
the patient's overall disease trend.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DATA_DIR, PATHOLOGIES

def compute_transition(state_t, state_t1):
    """
    Compute the transition between two consecutive study states.

    For each pathology, determine what happened:
    - "new":       0 → 1  (appeared)
    - "resolved":  1 → 0  (disappeared)
    - "persisted": 1 → 1  (still present)
    - "absent":    0 → 0  (still absent)
    - "worsened":  Inferred from temporal language + pathology count increase
    - "improved":  Inferred from temporal language + pathology count decrease
    """
    labels_t = state_t.get("pathology_labels", {})
    labels_t1 = state_t1.get("pathology_labels", {})
    trends = state_t1.get("per_pathology_trends", {})

    new_findings = []
    resolved_findings = []
    persisted_findings = []
    improved = []
    worsened = []
    transitions = {}

    for p in PATHOLOGIES:
        prev = max(labels_t.get(p, 0), 0)
        curr = max(labels_t1.get(p, 0), 0)
        trend = trends.get(p)

        if prev == 0 and curr == 1:
            transitions[p] = "new"
            new_findings.append(p)

        elif prev == 1 and curr == 0:
            transitions[p] = "resolved"
            resolved_findings.append(p)

        elif prev == 1 and curr == 1:
            transitions[p] = "persisted"
            persisted_findings.append(p)

            if trend == "improving":
                improved.append(p)
            elif trend == "worsening":
                worsened.append(p)

        else:
            transitions[p] = "absent"

    # simple direction logic (clean + interpretable)
    score = (
        len(new_findings)
        - len(resolved_findings)
        + len(worsened)
        - len(improved)
    )

    if score > 0:
        direction = "worsening"
    elif score < 0:
        direction = "improving"
    else:
        direction = state_t1.get("temporal_status", "stable")

    return {
        "from_study": state_t["study_id"],
        "to_study": state_t1["study_id"],
        "direction": direction,
        "new_findings": new_findings,
        "resolved_findings": resolved_findings,
        "persisted_findings": persisted_findings,
        "improved_pathologies": improved,
        "worsened_pathologies": worsened,
    }



def compute_overall_trend(transitions):
    """
    Aggregate individual transitions to determine the patient's
    overall disease trajectory.
    """
    if not transitions:
        return "unknown", 0.0

    counts = Counter(t["direction"] for t in transitions)
    total = len(transitions)

    direction = max(counts, key=counts.get)
    confidence = counts[direction] / total

    if len(counts) > 1:
        direction = "mixed"

    return direction, round(confidence, 3)


def analyze_transitions():
    """
    Main function: Compute transitions for all trajectories and
    update trajectories.json with transition data and overall trends.
    """
    print("\n=== Phase 3B: Analyzing Transitions ===")

    # Load trajectories
    with open(DATA_DIR / "trajectories.json", "r") as f:
        trajectories = json.load(f)

    total_transitions = 0
    trend_stats = Counter()

    for pid, traj in trajectories.items():
        states = traj["states"]
        transitions = []

        for i in range(len(states) - 1):
            t = compute_transition(states[i], states[i + 1])
            transitions.append(t)

        overall_trend, confidence = compute_overall_trend(transitions)

        traj["transitions"] = transitions
        traj["overall_trend"] = overall_trend
        traj["trend_confidence"] = confidence

        trend_stats[overall_trend] += 1
        total_transitions += len(transitions)

    # Save
    with open(DATA_DIR / "trajectories.json", "w") as f:
        json.dump(trajectories, f, indent=2)

    # Summary
    print("\n=== Summary ===")
    print(f"Total transitions: {total_transitions:,}")
    print(f"Avg per patient: {total_transitions/len(trajectories):.1f}")

    print("\nTrend distribution:")
    for k, v in trend_stats.items():
        print(f"{k:>12}: {v:,}")

    return trajectories


if __name__ == "__main__":
    analyze_transitions()