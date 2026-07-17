"""
Unified Embedding Classifier Trainer
Trains on report / fused embeddings (BiomedCLIP-based)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    PATHOLOGIES,
    CHECKPOINT_DIR,
    CLASSIFIER_LR,
    CLASSIFIER_EPOCHS,
    CLASSIFIER_BATCH_SIZE,
    CLASSIFIER_PATIENCE,
)
from src.training.dataset import ClassifierDataset


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class EmbeddingClassifier(nn.Module):
    def __init__(self, in_dim=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, len(PATHOLOGIES)),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────

def train_classifier(embedding_type="fused"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"Training Classifier ({embedding_type} embeddings)")
    print("=" * 60)

    train_ds = ClassifierDataset("train", embedding_type)
    val_ds = ClassifierDataset("val", embedding_type)

    train_loader = DataLoader(train_ds, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CLASSIFIER_BATCH_SIZE)

    model = EmbeddingClassifier(in_dim=train_ds.features.shape[1]).to(device)

    # pos weights
    pos = (train_ds.labels * train_ds.masks).sum(axis=0)
    neg = ((1 - train_ds.labels) * train_ds.masks).sum(axis=0)
    pos_weights = np.clip(neg / np.maximum(pos, 1), 0.5, 20.0)
    pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=CLASSIFIER_LR)

    best_val = float("inf")
    patience = 0

    for epoch in range(CLASSIFIER_EPOCHS):

        t0 = time.time()

        # train
        model.train()
        tl = 0

        for batch in train_loader:
            x = batch["features"].to(device)
            y = batch["labels"].to(device)
            m = batch["mask"].to(device)

            logits = model(x)
            loss = (criterion(logits, y) * m).sum() / m.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tl += loss.item()

        tl /= len(train_loader)

        # val
        model.eval()
        vl = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device)
                y = batch["labels"].to(device)
                m = batch["mask"].to(device)

                logits = model(x)
                loss = (criterion(logits, y) * m).sum() / m.sum().clamp(min=1)

                vl += loss.item()

        vl /= len(val_loader)

        print(f"Epoch {epoch+1} | train={tl:.4f} val={vl:.4f} ({time.time()-t0:.1f}s)")

        if vl < best_val:
            best_val = vl
            patience = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"{embedding_type}_classifier.pt")
        else:
            patience += 1
            if patience >= CLASSIFIER_PATIENCE:
                break

    print("Training done.")


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

def test_classifier(embedding_type="fused"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ClassifierDataset("test", embedding_type)
    model = EmbeddingClassifier(in_dim=ds.features.shape[1]).to(device)
    model.load_state_dict(
        torch.load(CHECKPOINT_DIR / f"{embedding_type}_classifier.pt", map_location=device)
    )
    model.eval()

    loader = DataLoader(ds, batch_size=512)

    labels_all = []
    probs_all = []
    masks_all = []

    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()

            probs_all.append(probs)
            labels_all.append(batch["labels"].numpy())
            masks_all.append(batch["mask"].numpy())

    labels_all = np.vstack(labels_all)
    probs_all = np.vstack(probs_all)

    masks_all = np.vstack(masks_all)

    probs_all = np.vstack(probs_all)
    labels_all = np.vstack(labels_all)

    per_label_metrics = {}
    aucs = []

    tp_total = fp_total = fn_total = 0

    preds_all = (probs_all >= 0.5).astype(int)

    for i, p in enumerate(PATHOLOGIES):
        valid = masks_all[:, i] == 1
        if valid.sum() == 0:
            per_label_metrics[p] = {"auc": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
            continue

        y = labels_all[valid, i]
        probs = probs_all[valid, i]
        preds = preds_all[valid, i]

        # AUC
        try:
            if y.sum() > 0 and y.sum() < len(y):
                auc = roc_auc_score(y, probs)
            else:
                auc = float("nan")
        except Exception:
            auc = float("nan")

        # Precision / Recall / F1 (binary)
        prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)

        per_label_metrics[p] = {"auc": auc, "precision": float(prec), "recall": float(rec), "f1": float(f1)}

        if not np.isnan(auc):
            aucs.append(auc)

        # accumulate for micro
        tp_total += int(((y == 1) & (preds == 1)).sum())
        fp_total += int(((y == 0) & (preds == 1)).sum())
        fn_total += int(((y == 1) & (preds == 0)).sum())

    # Print per-label metrics
    print("\nPer-label results:")
    for p, m in per_label_metrics.items():
        print(f"{p:25} AUC={m['auc'] if m['auc']==m['auc'] else 'nan':6}  P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    # Macro averages (mean over labels that have values)
    auc_mean = np.nanmean([v for v in aucs]) if len(aucs) > 0 else float("nan")
    precs = [v["precision"] for v in per_label_metrics.values() if not np.isnan(v["precision"])]
    recs = [v["recall"] for v in per_label_metrics.values() if not np.isnan(v["recall"])]
    f1s = [v["f1"] for v in per_label_metrics.values() if not np.isnan(v["f1"])]

    macro_prec = np.mean(precs) if len(precs) > 0 else float("nan")
    macro_rec = np.mean(recs) if len(recs) > 0 else float("nan")
    macro_f1 = np.mean(f1s) if len(f1s) > 0 else float("nan")

    # Micro averages from accumulated counts
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
        test_classifier()
    else:
        train_classifier()
        test_classifier()
