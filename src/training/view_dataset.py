"""
Dataset for View Classification (AP vs Lateral)
"""

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import (
    IMAGES_DIR,
    TEMPORAL_PATIENTS_PATH,
)

from src.vision.image_classifier import (
    preprocess_image,
    load_model,
)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class ViewDataset(Dataset):

    def __init__(self, metadata_csv):

        print("\n[ViewDataset] Loading metadata...")

        self.df = pd.read_csv(metadata_csv)

        print(f"[ViewDataset] Raw rows: {len(self.df):,}")

        # ─────────────────────────────────────
        # LOAD TEMPORAL PATIENTS
        # ─────────────────────────────────────

        print("[ViewDataset] Loading temporal cohort...")

        with open(TEMPORAL_PATIENTS_PATH, "r") as f:
            temporal_patients = set(
                json.load(f).keys()
            )

        print(
            f"[ViewDataset] Temporal patients: "
            f"{len(temporal_patients):,}"
        )

        # ─────────────────────────────────────
        # FILTER TO TEMPORAL PATIENTS
        # ─────────────────────────────────────

        self.df["subject_id"] = (
            "p" +
            self.df["subject_id"]
            .astype(str)
        )

        self.df = self.df[
            self.df["subject_id"]
            .isin(temporal_patients)
        ]

        print(
            f"[ViewDataset] After temporal filtering: "
            f"{len(self.df):,}"
        )

        # ─────────────────────────────────────
        # BUILD SAMPLES
        # ─────────────────────────────────────

        self.samples = []

        ap_count = 0
        lv_count = 0
        missing_count = 0

        for _, row in self.df.iterrows():

            view = str(
                row["ViewPosition"]
            ).upper()

            # AP / PA
            if view in ["AP", "PA"]:
                label = 0
                ap_count += 1

            # Lateral
            elif "LAT" in view:
                label = 1
                lv_count += 1

            else:
                continue

            subject_id = str(
                row["subject_id"]
            ).replace("p", "")

            study_id = str(
                row["study_id"]
            )

            dicom_id = str(
                row["dicom_id"]
            )

            # ─────────────────────────────
            # BUILD IMAGE PATH
            # ─────────────────────────────

            img_path = (
                IMAGES_DIR /
                f"p{subject_id[:2]}" /
                f"p{subject_id}" /
                f"s{study_id}" /
                f"{dicom_id}.jpg"
            )

            if img_path.exists():

                self.samples.append(
                    (
                        str(img_path),
                        label,
                    )
                )

            else:
                missing_count += 1

        # ─────────────────────────────────────
        # SUMMARY
        # ─────────────────────────────────────

        print("\n[ViewDataset] Dataset Summary")
        print("-" * 40)

        print(
            f"Final samples: "
            f"{len(self.samples):,}"
        )

        print(
            f"AP/PA samples: "
            f"{ap_count:,}"
        )

        print(
            f"Lateral samples: "
            f"{lv_count:,}"
        )

        print(
            f"Missing images skipped: "
            f"{missing_count:,}"
        )

        # ─────────────────────────────────────
        # LOAD SHARED TRANSFORMS
        # ─────────────────────────────────────

        print("\n[ViewDataset] Loading transforms...")

        _, self.transforms, _ = load_model()

        print("[ViewDataset] Ready\n")

    # ─────────────────────────────────────────
    # DATASET SIZE
    # ─────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    # ─────────────────────────────────────────
    # GET ITEM
    # ─────────────────────────────────────────

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        img = preprocess_image(
            path,
            self.transforms,
        )

        # remove extra batch dim
        img = img.squeeze(0)

        return (
            img,
            torch.tensor(
                label,
                dtype=torch.long,
            ),
        )