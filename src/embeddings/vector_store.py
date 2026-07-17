"""
Phase 4C: Vector Store (Simple NumPy-based)

Supports:
1. Report index     — report embeddings         (EMBEDDING_DIM)
2. Window index     — temporal windows          (EMBEDDING_DIM + 1)
3. Trajectory index — patient trajectories      (EMBEDDING_DIM + 6)

Image stores (dual-view):
4. AP image store     — AP view embeddings       (AP_EMBEDDING_DIM)
5. LV image store     — LV view embeddings       (LV_EMBEDDING_DIM)
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class VectorStore:
    """
    A simple, from-scratch vector store using NumPy.

    Stores embeddings in a dense matrix and uses dot product (on L2-normalized
    vectors = cosine similarity) for nearest neighbor search.
    """

    def __init__(self, name: str, embedding_dim: int):
        self.name = name
        self.embedding_dim = embedding_dim
        self.embeddings: Optional[np.ndarray] = None
        self.keys: List[str] = []
        self.metadata: Dict[str, dict] = {}
        self._is_built = False

    def add_batch(self, keys: List[str], embeddings: np.ndarray,
                  metadata: Optional[Dict[str, dict]] = None):

        assert len(keys) == embeddings.shape[0], "Key/embedding mismatch"
        assert embeddings.shape[1] == self.embedding_dim, "Wrong embedding dimension"

        embeddings = embeddings.astype(np.float32)

        # Ensure normalization (important for cosine similarity)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings = embeddings / norms

        self.keys = list(keys)
        self.embeddings = embeddings

        if metadata:
            self.metadata = metadata

        self._is_built = True

    def search(self, query: np.ndarray, top_k: int = 5):
        if not self._is_built:
            raise RuntimeError("Store not built")

        query = query.astype(np.float32)
        query = query / (np.linalg.norm(query) + 1e-8)

        sims = self.embeddings @ query

        idx = np.argsort(sims)[::-1][:top_k]

        return [
            (self.keys[i], float(sims[i]), self.metadata.get(self.keys[i], {}))
            for i in idx
        ]

    def append(self, keys: List[str], embeddings: np.ndarray,
               metadata: Optional[Dict[str, dict]] = None):
        """
        Append new vectors to an existing store at runtime.

        Used for on-the-fly ingestion of unseen patient data.
        Skips keys that already exist in the store.
        """

        assert embeddings.shape[1] == self.embedding_dim, "Wrong embedding dimension"

        embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings = embeddings / norms

        # Filter out duplicates
        new_idx = [i for i, k in enumerate(keys) if k not in set(self.keys)]

        if not new_idx:
            return 0

        new_keys = [keys[i] for i in new_idx]
        new_embs = embeddings[new_idx]

        if self._is_built and self.embeddings is not None:
            self.embeddings = np.vstack([self.embeddings, new_embs])
            self.keys.extend(new_keys)
        else:
            self.embeddings = new_embs
            self.keys = list(new_keys)
            self._is_built = True

        if metadata:
            for k in new_keys:
                if k in metadata:
                    self.metadata[k] = metadata[k]

        return len(new_keys)

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)

        np.save(directory / f"{self.name}_embeddings.npy", self.embeddings)

        with open(directory / f"{self.name}_meta.json", "w") as f:
            json.dump({
                "name": self.name,
                "dim": self.embedding_dim,
                "keys": self.keys
            }, f)

        if self.metadata:
            with open(directory / f"{self.name}_metadata.pkl", "wb") as f:
                pickle.dump(self.metadata, f)

        print(f"Saved {self.name}: {len(self.keys)} vectors")

    def load(self, directory: Path):
        self.embeddings = np.load(directory / f"{self.name}_embeddings.npy")

        with open(directory / f"{self.name}_meta.json") as f:
            meta = json.load(f)

        self.keys = meta["keys"]
        self.embedding_dim = meta["dim"]

        meta_path = directory / f"{self.name}_metadata.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)

        self._is_built = True

        print(f"Loaded {self.name}: {len(self.keys)} vectors")


# ─────────────────────────────────────────────
# BUILD TEXT STORES
# ─────────────────────────────────────────────

def build_text_vector_stores():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.config import DATA_DIR, EMBEDDING_DIM, WINDOW_EMBEDDING_DIM, TRAJECTORY_EMBEDDING_DIM

    print("\n=== Building TEXT Vector Stores ===")

    store_dir = DATA_DIR / "vector_store"

    # ── REPORT STORE ─────────────────────────
    data = np.load(DATA_DIR / "report_embeddings.npz", allow_pickle=True)

    report_store = VectorStore("reports", EMBEDDING_DIM)
    report_store.add_batch(
        data["study_keys"].tolist(),
        data["embeddings"]
    )
    report_store.save(store_dir)

    # ── WINDOW STORE ─────────────────────────
    data = np.load(DATA_DIR / "window_embeddings.npz", allow_pickle=True)

    with open(DATA_DIR / "temporal_windows.json") as f:
        windows = json.load(f)

    metadata = {
        k: {
            "patient_id": windows.get(k, {}).get("patient_id"),
            "trend": windows.get(k, {}).get("local_trend"),
            "size": windows.get(k, {}).get("window_size"),
        }
        for k in data["window_keys"]
    }

    window_store = VectorStore("windows", WINDOW_EMBEDDING_DIM)
    window_store.add_batch(
        data["window_keys"].tolist(),
        data["embeddings"],
        metadata
    )
    window_store.save(store_dir)

    # ── TRAJECTORY STORE ─────────────────────
    data = np.load(DATA_DIR / "trajectory_embeddings.npz", allow_pickle=True)

    with open(DATA_DIR / "trajectories.json") as f:
        trajectories = json.load(f)

    metadata = {
        pid: {
            "trend": trajectories.get(pid, {}).get("overall_trend"),
            "length": trajectories.get(pid, {}).get("duration_studies"),
        }
        for pid in data["patient_ids"]
    }

    traj_store = VectorStore("trajectories", TRAJECTORY_EMBEDDING_DIM)
    traj_store.add_batch(
        data["patient_ids"].tolist(),
        data["embeddings"],
        metadata
    )
    traj_store.save(store_dir)

    return report_store, window_store, traj_store


# ─────────────────────────────────────────────
# BUILD IMAGE STORES (DUAL)
# ─────────────────────────────────────────────

def build_image_vector_stores():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.config import DATA_DIR, AP_EMBEDDING_DIM, LV_EMBEDDING_DIM

    print("\n=== Building IMAGE Vector Stores (Dual) ===")

    store_dir = DATA_DIR / "vector_store"

    data = np.load(DATA_DIR / "dual_image_embeddings.npz", allow_pickle=True)

    study_keys = data["study_keys"].tolist()
    ap = data["ap_features"]
    lv = data["lv_features"]

    # AP store
    ap_store = VectorStore("img_ap", AP_EMBEDDING_DIM)
    ap_store.add_batch(study_keys, ap)
    ap_store.save(store_dir)

    # LV store
    lv_store = VectorStore("img_lv", LV_EMBEDDING_DIM)
    lv_store.add_batch(study_keys, lv)
    lv_store.save(store_dir)

    return ap_store, lv_store


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def build_all_vector_stores():
    print("\n=== Phase 6: Building All Vector Stores ===")

    text_stores = build_text_vector_stores()
    image_stores = build_image_vector_stores()

    print("\nDone building all vector stores.")

    return (*text_stores, *image_stores)


if __name__ == "__main__":
    build_all_vector_stores()