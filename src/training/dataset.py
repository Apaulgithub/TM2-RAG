import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DATA_DIR, PATHOLOGIES
from src.training.data_split import load_splits


class ClassifierDataset(Dataset):
    """
    Multi-label pathology classification dataset using BiomedCLIP embeddings.

    Each item:
      features: (D,) float tensor — embedding
      labels:   (14,) float tensor
      mask:     (14,) float tensor
      key:      str
    """

    def __init__(self, split="train", embedding_type="report"):
        splits = load_splits()
        split_pids = set(splits[split])

        # ── Load embeddings ──────────────────
        if embedding_type == "report":
            data = np.load(DATA_DIR / "report_embeddings.npz", allow_pickle=True)
            all_keys = list(data["study_keys"])
            all_embs = data["embeddings"]

        elif embedding_type == "fused":
            data = np.load(DATA_DIR / "vector_store" / "fused_reports_embeddings.npy")
            meta = json.load(open(DATA_DIR / "vector_store" / "fused_reports_meta.json"))
            all_keys = meta["keys"]
            all_embs = data

        else:
            raise ValueError("embedding_type must be 'report' or 'fused'")

        # ── Load labels ──────────────────────
        with open(DATA_DIR / "parsed_reports.json") as f:
            parsed = json.load(f)

        self.keys = []
        self.features = []
        self.labels = []
        self.masks = []

        for i, sk in enumerate(all_keys):
            pid = sk.rsplit("_", 1)[0]

            if pid not in split_pids:
                continue
            if sk not in parsed:
                continue

            path_labels = parsed[sk].get("pathology_labels", {})

            label = np.zeros(len(PATHOLOGIES), dtype=np.float32)
            mask = np.ones(len(PATHOLOGIES), dtype=np.float32)

            for j, p in enumerate(PATHOLOGIES):
                val = path_labels.get(p, 0)

                if val == 1:
                    label[j] = 1.0
                elif val == -1:
                    mask[j] = 0.0
                else:
                    label[j] = 0.0

            self.keys.append(sk)
            self.features.append(all_embs[i])
            self.labels.append(label)
            self.masks.append(mask)

        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        self.masks = np.array(self.masks, dtype=np.float32)

        print(f"{split}: {len(self.keys):,} samples")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return {
            "features": torch.from_numpy(self.features[idx]),
            "labels": torch.from_numpy(self.labels[idx]),
            "mask": torch.from_numpy(self.masks[idx]),
            "key": self.keys[idx],
        }