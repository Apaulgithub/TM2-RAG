"""
Simplified T-RAG Inference Pipeline
----------------------------------

Supports:
- text-only queries
- image-only queries
- multimodal queries
- temporal retrieval
- AP/LV-aware fusion
- VLM report generation
"""

import json
import sys

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    DATA_DIR,
    AP_VIEW_WEIGHT,
    LV_VIEW_WEIGHT,
)

from src.retrieval.query_processor import (
    process_query,
)

from src.retrieval.time_aware_retriever import (
    TimeAwareRetriever,
)

from src.generation.generator import (
    generate_report,
)

from src.vision.image_classifier import (
    classify_images_batch,
)

from src.vision.view_classifier import (
    ViewClassifier,
)

from src.fusion.claf_fusion import (
    encode_image_batch,
)


# ============================================================
# RESULT STRUCTURE
# ============================================================

@dataclass
class PredictionResult:

    patient_id: str = ""
    study_key: str = ""

    source: str = "text_only"

    views: List[str] = field(default_factory=list)

    pathology_probs: Dict[str, float] = (
        field(default_factory=dict)
    )

    pathology_positive: List[str] = (
        field(default_factory=list)
    )

    retrieval_hits: int = 0

    generated_report: str = ""

    def summary(self):

        positives = (
            self.pathology_positive
            if self.pathology_positive
            else ["None"]
        )

        views = (
            ", ".join(self.views)
            if self.views
            else "unknown"
        )

        return (
            f"\nPatient: {self.patient_id}\n"
            f"Study: {self.study_key}\n"
            f"Source: {self.source}\n"
            f"Views: {views}\n"
            f"Pathologies: {', '.join(positives)}\n"
            f"Retrieval Hits: {self.retrieval_hits}\n\n"
            f"{self.generated_report}"
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

class TRAGInference:

    def __init__(
        self,
        threshold: float = 0.5,
        generation_backend: str = "qwen2.5-vl-3b",
    ):

        self.threshold = threshold
        self.backend = generation_backend

        # ----------------------------------------------------
        # LOAD STATIC RESOURCES ONCE
        # ----------------------------------------------------

        self.retriever = TimeAwareRetriever(
            DATA_DIR
        )

        self.retriever.load()

        self.view_classifier = (
            ViewClassifier()
        )

        with open(DATA_DIR / "parsed_reports.json") as f:
            self.parsed_reports = json.load(f)

        with open(DATA_DIR / "temporal_patients.json") as f:
            self.temporal_patients = json.load(f)

    # ========================================================
    # IMAGE PATHOLOGY PREDICTION
    # ========================================================

    def predict_pathologies(
        self,
        image_path: str,
    ):

        pred = classify_images_batch(
            [image_path]
        )[0]

        probs = pred.pathology_probabilities

        positives = [

            pathology

            for pathology, prob
            in probs.items()

            if prob >= self.threshold
        ]

        return probs, positives

    # ========================================================
    # VIEW-AWARE IMAGE EMBEDDING
    # ========================================================

    def build_image_embedding(
        self,
        image_paths: List[str],
    ):

        image_paths = [
            p for p in image_paths
            if Path(p).exists()
        ]

        if not image_paths:
            return None, []

        embeddings = encode_image_batch(
            image_paths,
            batch_size=len(image_paths),
        )

        views = self.view_classifier.predict_batch(
            image_paths
        )

        weights = []

        for view in views:

            if view == "ap":
                weights.append(AP_VIEW_WEIGHT)

            else:
                weights.append(LV_VIEW_WEIGHT)

        weights = np.array(
            weights,
            dtype=np.float32,
        )

        weights /= weights.sum()

        fused = np.sum(
            embeddings * weights[:, None],
            axis=0,
        )

        return fused, views

    # ========================================================
    # MAIN PREDICTION
    # ========================================================

    def predict(
        self,
        patient_id: Optional[str] = None,
        study_key: Optional[str] = None,
        query_text: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
    ):

        if not patient_id and not study_key:

            raise ValueError(
                "Provide patient_id or study_key"
            )

        # ----------------------------------------------------
        # RESOLVE STUDY
        # ----------------------------------------------------

        if patient_id and not study_key:

            pdata = self.temporal_patients.get(
                patient_id
            )

            if not pdata:
                raise ValueError(
                    f"Patient not found: {patient_id}"
                )

            latest = pdata["studies"][-1]

            study_key = (
                f"{patient_id}_"
                f"{latest['study_id']}"
            )

        if study_key and not patient_id:

            patient_id = study_key.split("_")[0]

        result = PredictionResult(
            patient_id=patient_id,
            study_key=study_key,
        )

        # ====================================================
        # IMAGE PROCESSING
        # ====================================================

        probs = {}
        positives = []

        image_embedding = None
        views = []

        if image_paths:

            image_paths = [
                p for p in image_paths
                if Path(p).exists()
            ]

            if image_paths:

                probs, positives = (
                    self.predict_pathologies(
                        image_paths[0]
                    )
                )

                image_embedding, views = (
                    self.build_image_embedding(
                        image_paths
                    )
                )

                result.views = views

                result.source = (
                    "multimodal"
                    if query_text
                    else "image_only"
                )

        result.pathology_probs = probs
        result.pathology_positive = positives

        # ====================================================
        # TEXT QUERY
        # ====================================================

        if query_text is None:

            report_info = (
                self.parsed_reports.get(
                    study_key,
                    {}
                )
            )

            query_text = report_info.get(
                "raw_text",
                ""
            )

        if not query_text:

            query_text = (

                "Findings: "
                + ", ".join(positives)

                if positives

                else "Chest X-ray"
            )

        # ====================================================
        # BUILD QUERY
        # ====================================================

        query = process_query(
            report_text=query_text,
            study_key=study_key,
            image_path=(
                image_paths[0]
                if image_paths
                else None
            ),
        )

        # ----------------------------------------------------
        # INJECT IMAGE SIGNALS
        # ----------------------------------------------------

        if probs:

            for pathology, prob in probs.items():

                if prob >= self.threshold:

                    query.pathology_labels[
                        pathology
                    ] = 1

        if image_embedding is not None:

            query.image_embedding = (
                image_embedding
            )

        # ====================================================
        # RETRIEVAL
        # ====================================================

        retrieval_result = (
            self.retriever.retrieve(query)
        )

        result.retrieval_hits = len(
            retrieval_result.reports
        )

        # ====================================================
        # GENERATION
        # ====================================================

        result.generated_report = (
            generate_report(
                query=query,
                retrieval_result=retrieval_result,
                backend=self.backend,
            )
        )

        return result

    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup(self):

        try:
            import torch

            torch.cuda.empty_cache()

        except Exception:
            pass


# ============================================================
# SIMPLE CLI
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--patient")

    parser.add_argument("--study")

    parser.add_argument("--text")

    parser.add_argument("--image", nargs="+")

    parser.add_argument(
        "--model",
        default="qwen2.5-vl-3b",
    )

    args = parser.parse_args()

    pipeline = TRAGInference(
        generation_backend=args.model
    )

    result = pipeline.predict(
        patient_id=args.patient,
        study_key=args.study,
        query_text=args.text,
        image_paths=args.image,
    )

    print("\n" + "=" * 60)

    print(result.summary())

    pipeline.cleanup()