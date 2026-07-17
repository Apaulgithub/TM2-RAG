"""
Temporal Classifier (T-RAG Core Model)

Uses:
- report embeddings (512)
- window embeddings (513)
- trajectory embeddings (518)

Goal:
Predict pathology using temporal + multimodal context.
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

from src.config import PATHOLOGIES, CHECKPOINT_DIR, DATA_DIR


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

EPOCHS = 50
BATCH_SIZE = 128
LR = 3e-4
PATIENCE = 10

CKPT_PATH = CHECKPOINT_DIR / "temporal_classifier.pt"


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class TemporalDataset(Dataset):

    def __init__(self, split="train"):

        with open(DATA_DIR / "splits.json") as f:
            split_patients = set(json.load(f)[split])

        # load embeddings
        report = np.load(DATA_DIR / "report_embeddings.npz", allow_pickle=True)
        window = np.load(DATA_DIR / "window_embeddings.npz", allow_pickle=True)
        traj = np.load(DATA_DIR / "trajectory_embeddings.npz", allow_pickle=True)

        report_keys = report["study_keys"]
        window_map = dict(zip(window["window_keys"], window["embeddings"]))
        traj_map = dict(zip(traj["patient_ids"], traj["embeddings"]))

        reports = json.load(open(DATA_DIR / "parsed_reports.json"))

        self.r, self.w, self.t = [], [], []
        self.labels, self.masks = [], []

        pos_counts = np.zeros(len(PATHOLOGIES))
        neg_counts = np.zeros(len(PATHOLOGIES))

        for i, k in enumerate(report_keys):
            pid = k.split("_")[0]

            if pid not in split_patients:
                continue

            r_emb = report["embeddings"][i]
            w_emb = window_map.get(k)
            t_emb = traj_map.get(pid)

            if w_emb is None or t_emb is None:
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

            self.r.append(r_emb)
            self.w.append(w_emb)
            self.t.append(t_emb)
            self.labels.append(lbl)
            self.masks.append(msk)

        self.r = np.array(self.r, dtype=np.float32)
        self.w = np.array(self.w, dtype=np.float32)
        self.t = np.array(self.t, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        self.masks = np.array(self.masks, dtype=np.float32)

        self.pos_weights = np.clip(neg_counts / np.maximum(pos_counts, 1), 0.5, 20.0)

        print(f"{split}: {len(self.r):,} samples")

    def __len__(self):
        return len(self.r)

    def __getitem__(self, idx):
        return {
            "report": torch.tensor(self.r[idx]),
            "window": torch.tensor(self.w[idx]),
            "trajectory": torch.tensor(self.t[idx]),
            "labels": torch.tensor(self.labels[idx]),
            "mask": torch.tensor(self.masks[idx]),
        }


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class TemporalClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        # projections to same space
        self.r_proj = nn.Linear(512, 256)
        self.w_proj = nn.Linear(513, 256)
        self.t_proj = nn.Linear(518, 256)

        # gating (importance weighting)
        self.gate = nn.Sequential(
            nn.Linear(256 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )

        # classifier
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(PATHOLOGIES))
        )

    def forward(self, r, w, t):

        r = self.r_proj(r)
        w = self.w_proj(w)
        t = self.t_proj(t)

        concat = torch.cat([r, w, t], dim=-1)
        weights = self.gate(concat)  # (B, 3)

        fused = (
            weights[:, 0:1] * r +
            weights[:, 1:2] * w +
            weights[:, 2:3] * t
        )

        return self.head(fused)


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nTraining Temporal Classifier\n")

    train_ds = TemporalDataset("train")
    val_ds = TemporalDataset("val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = TemporalClassifier().to(device)

    pos_weights = torch.tensor(train_ds.pos_weights).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val = float("inf")
    patience = 0

    for epoch in range(EPOCHS):

        t0 = time.time()

        model.train()
        tl = 0

        for batch in train_loader:
            r = batch["report"].to(device)
            w = batch["window"].to(device)
            t = batch["trajectory"].to(device)
            y = batch["labels"].to(device)
            m = batch["mask"].to(device)

            logits = model(r, w, t)
            loss = (criterion(logits, y) * m).sum() / m.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tl += loss.item()

        tl /= len(train_loader)

        model.eval()
        vl = 0

        with torch.no_grad():
            for batch in val_loader:
                r = batch["report"].to(device)
                w = batch["window"].to(device)
                t = batch["trajectory"].to(device)
                y = batch["labels"].to(device)
                m = batch["mask"].to(device)

                logits = model(r, w, t)
                loss = (criterion(logits, y) * m).sum() / m.sum().clamp(min=1)

                vl += loss.item()

        vl /= len(val_loader)

        print(f"Epoch {epoch+1}: train={tl:.4f} val={vl:.4f}")

        if vl < best_val:
            best_val = vl
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

    model = TemporalClassifier().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    ds = TemporalDataset("test")
    loader = DataLoader(ds, batch_size=256)

    labels_all = []
    probs_all = []
    masks_all = []

    with torch.no_grad():
        for batch in loader:
            r = batch["report"].to(device)
            w = batch["window"].to(device)
            t = batch["trajectory"].to(device)

            probs = torch.sigmoid(model(r, w, t)).cpu().numpy()

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