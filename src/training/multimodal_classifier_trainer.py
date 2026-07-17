"""
Multimodal Classifier (BiomedCLIP Fused Embeddings)

Input: 512-dim fused embeddings
Output: multi-label pathology predictions
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import PATHOLOGIES, CHECKPOINT_DIR, DATA_DIR, FUSION_DIM


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

EPOCHS = 50
BATCH_SIZE = 256
LR = 3e-4
PATIENCE = 10
WARMUP = 3
CKPT_PATH = CHECKPOINT_DIR / "multimodal_classifier.pt"


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class MultimodalDataset(Dataset):

    def __init__(self, split="train"):

        with open(DATA_DIR / "splits.json") as f:
            split_patients = set(json.load(f)[split])

        emb = np.load(DATA_DIR / "vector_store" / "fused_reports_embeddings.npy", mmap_mode="r")
        meta = json.load(open(DATA_DIR / "vector_store" / "fused_reports_meta.json"))
        keys = meta["keys"]

        reports = json.load(open(DATA_DIR / "parsed_reports.json"))

        self.features = []
        self.labels = []
        self.masks = []

        pos_counts = np.zeros(len(PATHOLOGIES))
        neg_counts = np.zeros(len(PATHOLOGIES))

        for i, k in enumerate(keys):
            pid = k.split("_")[0]
            if pid not in split_patients:
                continue

            vec = emb[i]
            if np.linalg.norm(vec) == 0:
                continue

            lbl_dict = reports.get(k, {}).get("pathology_labels", {})

            lbl, msk = [], []

            for p in PATHOLOGIES:
                v = lbl_dict.get(p, 0)

                if v == 1:
                    lbl.append(1.0); msk.append(1.0)
                elif v == -1:
                    lbl.append(0.0); msk.append(0.0)
                else:
                    lbl.append(0.0); msk.append(1.0)

            lbl = np.array(lbl, dtype=np.float32)
            msk = np.array(msk, dtype=np.float32)

            pos_counts += (lbl * msk > 0.5)
            neg_counts += ((1 - lbl) * msk > 0.5)

            self.features.append(vec.astype(np.float32))
            self.labels.append(lbl)
            self.masks.append(msk)

        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        self.masks = np.array(self.masks)

        self.pos_weights = np.clip(neg_counts / np.maximum(pos_counts, 1), 0.5, 20.0)

        print(f"{split}: {len(self.features):,} samples | dim={FUSION_DIM}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.features[idx]),
            "labels": torch.tensor(self.labels[idx]),
            "mask": torch.tensor(self.masks[idx]),
        }


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class MultimodalClassifier(nn.Module):

    def __init__(self, in_dim=FUSION_DIM):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, len(PATHOLOGIES))
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nTraining Multimodal Classifier\n")

    train_ds = MultimodalDataset("train")
    val_ds = MultimodalDataset("val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = MultimodalClassifier().to(device)

    pos_weights = torch.tensor(train_ds.pos_weights).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val = float("inf")
    patience = 0

    for epoch in range(EPOCHS):

        t0 = time.time()

        # ── TRAIN ───────────────────────────
        model.train()
        train_loss = 0

        for batch in train_loader:
            x = batch["features"].to(device)
            y = batch["labels"].to(device)
            m = batch["mask"].to(device)

            logits = model(x)
            loss = (criterion(logits, y) * m).sum() / m.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── VAL ─────────────────────────────
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device)
                y = batch["labels"].to(device)
                m = batch["mask"].to(device)

                logits = model(x)
                loss = (criterion(logits, y) * m).sum() / m.sum().clamp(min=1)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f}")

        # ── EARLY STOP ──────────────────────
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), CKPT_PATH)
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    print("Done.")


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

def test():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultimodalClassifier().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    ds = MultimodalDataset("test")
    loader = DataLoader(ds, batch_size=512)

    labels_all = []
    probs_all = []
    masks_all = []

    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            probs = torch.sigmoid(model(x)).cpu().numpy()

            probs_all.append(probs)
            labels_all.append(batch["labels"].numpy())
            masks_all.append(batch["mask"].numpy())

    labels_all = np.vstack(labels_all)
    probs_all = np.vstack(probs_all)

    masks_all = np.vstack(masks_all)
    probs_all = np.vstack(probs_all)
    labels_all = np.vstack(labels_all)

    preds_all = (probs_all >= 0.5).astype(int)

    per_label_metrics = {}
    aucs = []
    tp_total = fp_total = fn_total = 0

    for i, p in enumerate(PATHOLOGIES):
        valid = masks_all[:, i] == 1
        if valid.sum() == 0:
            per_label_metrics[p] = {"auc": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
            continue

        y = labels_all[valid, i]
        probs = probs_all[valid, i]
        preds = preds_all[valid, i]

        try:
            auc = roc_auc_score(y, probs) if (y.sum() > 0 and y.sum() < len(y)) else float("nan")
        except Exception:
            auc = float("nan")

        prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)

        per_label_metrics[p] = {"auc": auc, "precision": float(prec), "recall": float(rec), "f1": float(f1)}

        if not np.isnan(auc):
            aucs.append(auc)

        tp_total += int(((y == 1) & (preds == 1)).sum())
        fp_total += int(((y == 0) & (preds == 1)).sum())
        fn_total += int(((y == 1) & (preds == 0)).sum())

    print("\nPer-label results:")
    for p, m in per_label_metrics.items():
        print(f"{p:25} AUC={m['auc'] if m['auc']==m['auc'] else 'nan':6}  P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    auc_mean = np.nanmean([v for v in aucs]) if len(aucs) > 0 else float("nan")
    precs = [v["precision"] for v in per_label_metrics.values() if not np.isnan(v["precision"])]
    recs = [v["recall"] for v in per_label_metrics.values() if not np.isnan(v["recall"])]
    f1s = [v["f1"] for v in per_label_metrics.values() if not np.isnan(v["f1"])]

    macro_prec = np.mean(precs) if len(precs) > 0 else float("nan")
    macro_rec = np.mean(recs) if len(recs) > 0 else float("nan")
    macro_f1 = np.mean(f1s) if len(f1s) > 0 else float("nan")

    micro_prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else float("nan")
    micro_rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else float("nan")
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec == micro_prec and micro_rec == micro_rec and (micro_prec + micro_rec) > 0) else float("nan")

    print(f"\nMacro — AUC: {auc_mean:.4f}  P: {macro_prec:.4f}  R: {macro_rec:.4f}  F1: {macro_f1:.4f}")
    print(f"Micro — P: {micro_prec:.4f}  R: {micro_rec:.4f}  F1: {micro_f1:.4f}")

    if __name__ == "__main__":
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--test-only", action="store_true")
        args = parser.parse_args()
        if args.test_only:
            test()
        else:
            train()
            test()