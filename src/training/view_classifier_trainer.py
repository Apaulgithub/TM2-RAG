"""
View Classifier Trainer
"""

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    DataLoader,
    random_split,
)

from src.training.view_dataset import ViewDataset

from src.vision.view_classifier import (
    ViewClassifierHead,
)

from src.vision.image_classifier import (
    load_model,
)
from src.config import CHECKPOINT_DIR, PROJECT_ROOT

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────

def train():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\n[Trainer] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[Trainer] CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"[Trainer] GPU Count: {torch.cuda.device_count()}")

    # ─────────────────────────────────────────
    # LOAD DATASET
    # ─────────────────────────────────────────

    print("\n[Trainer] Loading dataset...")
    dataset = ViewDataset(
        PROJECT_ROOT / "mimic-cxr-2.0.0-metadata.csv.gz"
    )
    print(f"[Trainer] Dataset loaded. Total samples: {len(dataset):,}")

    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset contains 0 samples."
        )

    # deterministic split
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(42)

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    print(
        f"[Trainer] Train samples: {len(train_ds):,}"
    )

    print(
        f"[Trainer] Val samples: {len(val_ds):,}"
    )

    # ─────────────────────────────────────────
    # DATALOADERS
    # ─────────────────────────────────────────

    train_loader = DataLoader(
        train_ds,
        batch_size=256,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    # ─────────────────────────────────────────
    # LOAD DENSENET BACKBONE
    # ─────────────────────────────────────────

    print("\n[Trainer] Loading DenseNet backbone...")

    backbone, _, _ = load_model()

    backbone = backbone.to(device)
    backbone.eval()

    # freeze backbone
    for p in backbone.parameters():
        p.requires_grad = False

    # feature extractor
    def extract_features(imgs):

        with torch.no_grad():

            feats = backbone.features(imgs)

            feats = torch.relu(feats)

            # global average pooling
            feats = feats.mean(dim=[2, 3])

        return feats

    # ─────────────────────────────────────────
    # VIEW HEAD
    # ─────────────────────────────────────────

    model = ViewClassifierHead().to(device)

    # ─────────────────────────────────────────
    # CLASS BALANCING
    # ─────────────────────────────────────────

    labels = torch.tensor(
        [s[1] for s in dataset.samples]
    )

    ap_count = (labels == 0).sum().item()
    lv_count = (labels == 1).sum().item()

    print("\n[Trainer] Class Distribution")
    print(f"AP/PA: {ap_count:,}")
    print(f"LV:    {lv_count:,}")

    weights = torch.tensor([
        1.0,
        ap_count / max(lv_count, 1)
    ], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    scaler = torch.cuda.amp.GradScaler()

    best_val = float("inf")

    patience = 0
    PATIENCE = 3

    print("\n[Trainer] Starting training...\n")

    # ─────────────────────────────────────────
    # TRAIN LOOP
    # ─────────────────────────────────────────

    for epoch in range(10):

        # ─────────────────────────────────────
        # TRAIN
        # ─────────────────────────────────────

        model.train()

        train_loss = 0
        correct = 0
        total = 0
        
        print(f"\n[Epoch {epoch+1}/10] Training...")

        for batch_idx, (imgs, labels) in enumerate(train_loader):

            imgs = imgs.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():

                feats = extract_features(imgs)

                logits = model(feats)

                loss = criterion(
                    logits,
                    labels,
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            train_loss += loss.item()

            preds = logits.argmax(dim=1)

            correct += (
                preds == labels
            ).sum().item()

            total += labels.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}: loss={loss.item():.4f}")

        train_acc = correct / total

        # ─────────────────────────────────────
        # VALIDATION
        # ─────────────────────────────────────

        print(f"[Epoch {epoch+1}/10] Validating...")
        model.eval()

        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():

            for val_batch_idx, (imgs, labels) in enumerate(val_loader):

                imgs = imgs.to(device)

                labels = labels.to(device)

                feats = extract_features(imgs)

                logits = model(feats)

                loss = criterion(
                    logits,
                    labels,
                )

                val_loss += loss.item()

                preds = logits.argmax(dim=1)

                correct += (
                    preds == labels
                ).sum().item()

                total += labels.size(0)
                
                if (val_batch_idx + 1) % 10 == 0:
                    print(f"  Val Batch {val_batch_idx + 1}/{len(val_loader)}")

        val_acc = correct / total

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"\n✓ Epoch {epoch+1}/{10} completed:\n"
            f"  Train: loss={avg_train_loss:.4f}, acc={train_acc:.4f}\n"
            f"  Val:   loss={avg_val_loss:.4f}, acc={val_acc:.4f}\n"
        )

        # ─────────────────────────────────────
        # SAVE BEST
        # ─────────────────────────────────────

        if avg_val_loss < best_val:

            best_val = val_loss
            patience = 0

            CHECKPOINT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR /
                "view_classifier.pt"
            )

            print(
                "[Checkpoint] "
                "Saved best model"
            )

        else:

            patience += 1

            if patience >= PATIENCE:
                print(
                    "\n[Trainer] Early stopping."
                )
                break

    print("\n[Trainer] Training complete.")
    if torch.cuda.is_available():
        print(f"[Trainer] Final GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")

# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    train()