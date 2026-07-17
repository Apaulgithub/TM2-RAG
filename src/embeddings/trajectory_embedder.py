"""
Phase 4B: Trajectory & Window Embedder

Uses improved report embeddings
Adds continuous temporal signals
Adds severity trend features
Keeps simple & fast design
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DATA_DIR, EMBEDDING_DIM


# ─────────────────────────────────────────────
# TEMPORAL FEATURES (UPDATED)
# ─────────────────────────────────────────────

def build_temporal_features(traj):
    """
    Build richer temporal features using:
     transitions
     overall trend
     severity progression
    """

    transitions = traj.get("transitions", [])
    states = traj.get("states", [])

    n = max(len(transitions), 1)

    improving = sum(len(t.get("improved_pathologies", [])) for t in transitions)
    worsening = sum(len(t.get("worsened_pathologies", [])) for t in transitions)
    new = sum(len(t.get("new_findings", [])) for t in transitions)
    resolved = sum(len(t.get("resolved_findings", [])) for t in transitions)

    # ─────────────────────────────────────
    # TREND SCORE
    # ─────────────────────────────────────
    trend_map = {
        "improving": 1.0,
        "stable": 0.0,
        "mixed": -0.2,
        "worsening": -1.0,
        "unknown": 0.0,
    }

    trend = traj.get("overall_trend", "unknown")
    trend_score = trend_map.get(trend, 0.0) * traj.get("trend_confidence", 0.5)

    # ─────────────────────────────────────
    # NEW: SEVERITY SIGNAL
    # ─────────────────────────────────────
    scores = [s.get("avg_pathology_score", 0.0) for s in states]

    if scores:
        mean_score = np.mean(scores)
        delta_score = scores[-1] - scores[0] if len(scores) > 1 else 0.0
    else:
        mean_score = 0.0
        delta_score = 0.0

    # ─────────────────────────────────────
    # FINAL FEATURE VECTOR
    # ─────────────────────────────────────
    features = np.array([
        min(traj.get("duration_studies", 2) / 20.0, 1.0),
        trend_score,
        improving / n,
        worsening / n,
        new / n,
        resolved / n,

        mean_score,
        delta_score,
    ], dtype=np.float32)

    return features


# ─────────────────────────────────────────────
# TRAJECTORY EMBEDDINGS
# ─────────────────────────────────────────────

def embed_trajectories():

    print("\n=== Phase 4B: Trajectory Embeddings ===")

    # Load report embeddings
    data = np.load(DATA_DIR / "report_embeddings.npz", allow_pickle=True)
    report_embs = data["embeddings"]
    report_keys = data["study_keys"].tolist()

    key_to_emb = {k: report_embs[i] for i, k in enumerate(report_keys)}

    # Load trajectories
    with open(DATA_DIR / "trajectories.json") as f:
        trajectories = json.load(f)

    patient_ids = []
    traj_embs = []

    for pid, traj in trajectories.items():

        states = traj["states"]

        # ─────────────────────────────────────
        # MEAN REPORT EMBEDDING
        # ─────────────────────────────────────
        embs = []

        for s in states:
            key = f"{pid}_{s['study_id']}"
            if key in key_to_emb:
                embs.append(key_to_emb[key])

        if embs:
            mean_emb = np.mean(embs, axis=0)
        else:
            mean_emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        # ─────────────────────────────────────
        # TEMPORAL FEATURES (UPDATED)
        # ─────────────────────────────────────
        temp = build_temporal_features(traj)

        combined = np.concatenate([mean_emb, temp])

        # normalize
        combined = combined / (np.linalg.norm(combined) + 1e-8)

        traj_embs.append(combined)
        patient_ids.append(pid)

    traj_matrix = np.stack(traj_embs).astype(np.float32)

    np.savez_compressed(
        DATA_DIR / "trajectory_embeddings.npz",
        embeddings=traj_matrix,
        patient_ids=np.array(patient_ids, dtype=object),
    )

    print(f"Saved trajectory embeddings: {traj_matrix.shape}")

    return traj_matrix, patient_ids


# ─────────────────────────────────────────────
# WINDOW EMBEDDINGS (MINOR UPDATE)
# ─────────────────────────────────────────────

def embed_windows():

    print("\n=== Phase 4B-W: Window Embeddings ===")

    with open(DATA_DIR / "temporal_windows.json") as f:
        windows = json.load(f)

    from src.fusion.claf_fusion import encode_text_batch

    keys = sorted(windows.keys())
    texts = [windows[k]["serialised_text"] for k in keys]

    batch_size = 64
    text_embs = []

    for i in range(0, len(texts), batch_size):
        text_embs.append(encode_text_batch(texts[i:i+batch_size]))

    text_matrix = np.concatenate(text_embs, axis=0)

    final = []

    for i, k in enumerate(keys):
        w = windows[k]

        trend_map = {
            "improving": 1.0,
            "stable": 0.0,
            "worsening": -1.0,
        }

        trend = trend_map.get(w.get("local_trend", "stable"), 0.0)

        # NEW: optional intensity
        intensity = w.get("avg_score", 0.0)

        vec = np.concatenate([
            text_matrix[i],
            np.array([trend, intensity], dtype=np.float32)
        ])

        vec = vec / (np.linalg.norm(vec) + 1e-8)
        final.append(vec)

    final_matrix = np.stack(final).astype(np.float32)

    np.savez_compressed(
        DATA_DIR / "window_embeddings.npz",
        embeddings=final_matrix,
        window_keys=np.array(keys, dtype=object),
    )

    print(f"Saved window embeddings: {final_matrix.shape}")

    return final_matrix, keys


if __name__ == "__main__":
    embed_trajectories()
    embed_windows()