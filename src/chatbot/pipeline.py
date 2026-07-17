"""
Chatbot Pipeline — wraps the full T-RAG pipeline for Streamlit.

Loads once via @st.cache_resource, serves many queries.
Supports: text-only, image-only, and text+image queries.
Optional Grad-CAM heatmap generation on demand.
"""

import re
import csv
import json
import io
import sys
import tempfile
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    DATA_DIR,
    LLM_BACKEND,
    PATHOLOGIES,
    PROJECT_ROOT,
    VLM_MODELS,
)


# ─────────────────────────────────────────────
# HEATMAP KEYWORDS
# ─────────────────────────────────────────────

_HEATMAP_TRIGGERS = [
    "heatmap", "heat map", "grad-cam", "gradcam",
    "grad cam", "show me where", "highlight",
    "saliency", "attention map", "visualize",
    "show findings", "overlay",
]


def wants_heatmap(text: str) -> bool:
    """Check if user query asks for Grad-CAM heatmaps."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in _HEATMAP_TRIGGERS)


def wants_heatmap_only(text: str) -> bool:
    """Check if the user wants only the Grad-CAM image, not a report."""
    text_lower = text.lower()
    heatmap_requested = wants_heatmap(text_lower)
    only_requested = any(
        phrase in text_lower
        for phrase in [
            "only",
            "image only",
            "heatmap only",
            "grad-cam only",
            "gradcam only",
            "just",
            "without report",
            "no report",
            "don't generate report",
            "do not generate report",
        ]
    )
    report_requested = any(
        phrase in text_lower
        for phrase in [
            "report",
            "findings",
            "impression",
            "diagnosis",
        ]
    )

    return heatmap_requested and (only_requested or not report_requested)


# ─────────────────────────────────────────────
# METADATA PARSER
# ─────────────────────────────────────────────

def parse_metadata_file(file_bytes: bytes, filename: str) -> str:
    """
    Parse uploaded patient metadata (CSV / JSON / TXT)
    and return a text summary for the prompt.
    """

    ext = Path(filename).suffix.lower()
    text = file_bytes.decode("utf-8", errors="replace")

    if ext == ".csv":
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            return ""

        lines = [f"Patient metadata ({len(rows)} studies):"]
        for i, row in enumerate(rows[:10]):  # cap at 10 studies
            parts = []
            for col in ["subject_id", "study_id", "dicom_id",
                        "view_position", "study_date"]:
                if col in row and row[col]:
                    parts.append(f"{col}={row[col]}")
            if parts:
                lines.append(f"  Study {i+1}: {', '.join(parts)}")

        if len(rows) > 10:
            lines.append(f"  ... and {len(rows) - 10} more studies")

        return "\n".join(lines)

    elif ext == ".json":
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                # Could be a single report or patient record
                return json.dumps(data, indent=2)[:2000]
            elif isinstance(data, list):
                return json.dumps(data[:5], indent=2)[:2000]
        except json.JSONDecodeError:
            return text[:2000]

    else:  # .txt or unknown
        return text[:2000]


# ─────────────────────────────────────────────
# PROMPT BUILDER (matches to-do spec)
# ─────────────────────────────────────────────

def build_rag_prompt(
    user_query: str,
    retrieval_result,
    metadata_text: str = "",
) -> str:
    """Build the temporal context prompt from retrieval results.

    Uses BiomedCLIP CLAF multimodal retrieval context when available.
    """

    # -- Patient timeline / metadata
    patient_timeline_text = metadata_text or "No patient metadata provided."

    # -- Reports (text-only retrieval)
    reports_parts = []
    for r in (retrieval_result.reports or [])[:3]:
        findings = r.get("findings", "")
        impression = r.get("impression", "")
        if findings or impression:
            reports_parts.append(
                f"Findings: {findings[:400]}\n"
                f"Impression: {impression[:250]}"
            )
    reports_text = "\n---\n".join(reports_parts) if reports_parts else "None"

    # -- Fused/multimodal context (BiomedCLIP CLAF retrieval)
    # The retriever builds a full context string that includes
    # multimodal-fused retrieval results from BiomedCLIP-aligned
    # embeddings (text + image → shared 512-dim space).
    fused_text = retrieval_result.context if retrieval_result.context else "None"

    # -- Temporal windows
    windows_parts = []
    for w in (retrieval_result.windows or [])[:3]:
        trend = w.get("trend", "unknown")
        text = w.get("text", "")
        windows_parts.append(f"({trend}) {text[:300]}")
    windows_text = "\n".join(windows_parts) if windows_parts else "None"

    # -- Longitudinal trajectories
    traj_parts = []
    for t in (retrieval_result.trajectories or [])[:2]:
        trend = t.get("trend", "unknown")
        length = t.get("length", 0)
        traj_parts.append(f"Trend: {trend} | Length: {length}")
    trajectory_text = "\n".join(traj_parts) if traj_parts else "None"

    prompt = f"""You are an expert thoracic radiologist AI specialized in
longitudinal chest X-ray interpretation.

PATIENT TIMELINE CONTEXT:
{patient_timeline_text}

RETRIEVED PRIOR REPORTS:
{reports_text}

RETRIEVED MULTIMODAL CASES (BiomedCLIP CLAF):
{fused_text}

TEMPORAL PROGRESSION MEMORY:
{windows_text}

LONGITUDINAL TRAJECTORY MEMORY:
{trajectory_text}

USER QUERY:
{user_query}

TASK:
Generate a clinically grounded chest X-ray report
for the MOST RECENT study using temporal reasoning
across the provided timeline and retrieved memories.

Focus on:
- interval progression
- worsening vs improvement
- persistent findings
- disease evolution
- temporal consistency

Output ONLY:

FINDINGS:
...

IMPRESSION:
...

Rules:
- Use concise radiology-report style
- No bullet points
- No conversational text
- No explanations
- Keep longitudinal reasoning clinically grounded
"""
    return prompt.strip()


# ─────────────────────────────────────────────
# EXTRACT PATHOLOGIES FROM REPORT TEXT
# ─────────────────────────────────────────────

def extract_pathologies_from_text(text: str) -> List[str]:
    """Extract pathology labels mentioned in the generated report."""
    text_lower = text.lower()
    found = []
    for p in PATHOLOGIES:
        name = p.lower()
        if p == "No Finding" or name not in text_lower:
            continue

        negated_patterns = [
            rf"\bno\s+{re.escape(name)}\b",
            rf"\bwithout\s+{re.escape(name)}\b",
            rf"\bno\s+evidence\s+of\s+{re.escape(name)}\b",
            rf"\bno\s+overt\s+{re.escape(name)}\b",
        ]
        if any(re.search(pattern, text_lower) for pattern in negated_patterns):
            continue

        found.append(p)

    priority = [
        "Atelectasis",
        "Lung Opacity",
        "Consolidation",
        "Pneumonia",
        "Edema",
        "Pleural Effusion",
        "Cardiomegaly",
    ]
    return sorted(
        found,
        key=lambda item: priority.index(item) if item in priority else len(priority),
    )


# ─────────────────────────────────────────────
# CHAT PIPELINE
# ─────────────────────────────────────────────

class ChatPipeline:
    """
    Main pipeline: loads once, serves many queries.
    Wraps retriever + VLM + optional Grad-CAM.
    """

    def __init__(self):
        self.retriever = None
        self.vlm = None
        self._gradcam = None
        self._loaded = False

    # ─────────────────────────────────────
    # LOAD
    # ─────────────────────────────────────

    def load(self):
        """Load retriever. Qwen is loaded lazily after query retrieval."""
        if self._loaded:
            return

        # -- Retriever
        from src.retrieval.time_aware_retriever import TimeAwareRetriever
        self.retriever = TimeAwareRetriever(data_dir=DATA_DIR)
        self.retriever.load()

        self._loaded = True

    def _release_retrieval_gpu_models(self):
        """Free GPU memory used by retrieval encoders before loading Qwen."""
        try:
            from src.fusion.claf_fusion import unload_biomedclip
            unload_biomedclip()
        except Exception as e:
            print(f"[WARN] Could not unload BiomedCLIP: {e}")

        try:
            from src.vision.image_classifier import unload_model
            unload_model()
        except Exception as e:
            print(f"[WARN] Could not unload image classifier: {e}")

    def _release_vlm_for_retrieval(self):
        """Free Qwen before retrieval-side encoders need CUDA memory."""
        if self.vlm is None:
            return

        self.vlm = None
        try:
            from src.generation.vlm_backends import clear_backend_cache
            clear_backend_cache()
        except Exception as e:
            print(f"[WARN] Could not clear VLM cache: {e}")

    def _load_vlm(self):
        """Lazy-load Qwen2.5-VL-3B for report generation."""
        if self.vlm is None:
            from src.generation.vlm_backends import get_vlm_backend
            model_name = (
                LLM_BACKEND
                if LLM_BACKEND in VLM_MODELS
                else "qwen2.5-vl-3b"
            )
            self.vlm = get_vlm_backend(model_name)
        return self.vlm

    @property
    def gradcam(self):
        """Lazy-load Grad-CAM only when needed."""
        if self._gradcam is None:
            from src.vision.gradcam import GradCAM
            self._gradcam = GradCAM()
        return self._gradcam

    # ─────────────────────────────────────
    # PROCESS QUERY
    # ─────────────────────────────────────

    def _process_query(
        self,
        report_text: Optional[str],
        image_paths: Optional[List[str]],
    ):
        from src.retrieval.query_processor import process_query
        return process_query(
            report_text=report_text,
            image_path=image_paths[0] if image_paths else None,
        )

    # ─────────────────────────────────────
    # GENERATE (streaming)
    # ─────────────────────────────────────

    def generate_stream(
        self,
        user_query: str,
        image_paths: Optional[List[str]] = None,
        metadata_text: str = "",
    ) -> Generator[str, None, None]:
        """
        Full RAG cycle with streaming output.

        Yields tokens one by one.
        """

        # 1. Free Qwen first so retrieval encoders can use CUDA.
        self._release_vlm_for_retrieval()
        retrieval_query_text = user_query if user_query.strip() else ""
        if metadata_text:
            retrieval_query_text = (
                retrieval_query_text + "\n\n" + metadata_text[:2000]
            ).strip()

        # 2. Process query
        query = self._process_query(
            report_text=retrieval_query_text or None,
            image_paths=image_paths,
        )

        # 3. Retrieve
        retrieval_result = self.retriever.retrieve(query)

        # 4. Build prompt
        prompt = build_rag_prompt(
            user_query=user_query,
            retrieval_result=retrieval_result,
            metadata_text=metadata_text,
        )

        # 5. Release retrieval-side GPU models, then stream from Qwen.
        self._release_retrieval_gpu_models()
        vlm = self._load_vlm()
        yield from vlm.generate_stream(
            prompt=prompt,
            image_paths=image_paths,
        )

    # ─────────────────────────────────────
    # GENERATE (non-streaming)
    # ─────────────────────────────────────

    def generate(
        self,
        user_query: str,
        image_paths: Optional[List[str]] = None,
        metadata_text: str = "",
    ) -> str:
        """Full RAG cycle, returns complete text."""

        self._release_vlm_for_retrieval()
        retrieval_query_text = user_query if user_query.strip() else ""
        if metadata_text:
            retrieval_query_text = (
                retrieval_query_text + "\n\n" + metadata_text[:2000]
            ).strip()

        query = self._process_query(
            report_text=retrieval_query_text or None,
            image_paths=image_paths,
        )

        retrieval_result = self.retriever.retrieve(query)

        prompt = build_rag_prompt(
            user_query=user_query,
            retrieval_result=retrieval_result,
            metadata_text=metadata_text,
        )

        self._release_retrieval_gpu_models()
        vlm = self._load_vlm()
        return vlm.generate(
            prompt=prompt,
            image_paths=image_paths,
        )

    # ─────────────────────────────────────
    # GRAD-CAM
    # ─────────────────────────────────────

    def generate_heatmaps(
        self,
        image_paths: List[str],
        report_text: str = "",
    ) -> List[dict]:
        """
        Generate Grad-CAM overlays for uploaded images.
        Generates one overlay per relevant pathology target.
        """

        self._release_vlm_for_retrieval()

        pathologies = extract_pathologies_from_text(report_text)
        targets = pathologies[:4] if pathologies else [None]
        print(
            f"[GradCAM] Generating overlays for {len(image_paths)} image(s); "
            f"targets={[t or 'model-top-class' for t in targets]}"
        )

        overlays = []
        for img_path in image_paths:
            combined_heatmap = None
            completed_targets = []
            max_confidence = 0.0

            for target_label in targets:
                try:
                    heatmap, _, label, confidence = self.gradcam.generate(
                        image_path=img_path,
                        target_label=target_label,
                    )
                    print(
                        f"[GradCAM] Overlay ready for {img_path}; "
                        f"target={label}, confidence={confidence:.4f}"
                    )
                    weighted_heatmap = heatmap.astype(np.float32) * max(confidence, 0.05)
                    if combined_heatmap is None:
                        combined_heatmap = weighted_heatmap
                    else:
                        combined_heatmap = np.maximum(combined_heatmap, weighted_heatmap)
                    completed_targets.append(label)
                    max_confidence = max(max_confidence, confidence)
                except Exception as e:
                    print(
                        f"[GradCAM] Failed for {img_path} "
                        f"target={target_label or 'model-top-class'}: {e}"
                    )

            if combined_heatmap is not None:
                combined_heatmap -= combined_heatmap.min()
                combined_heatmap /= (combined_heatmap.max() + 1e-8)
                combined_heatmap = (combined_heatmap * 255).astype(np.uint8)

                original = np.array(Image.open(img_path).convert("RGB"))
                overlay = self.gradcam._overlay(original, combined_heatmap)
                label = "Combined: " + ", ".join(dict.fromkeys(completed_targets))
                overlays.append({
                    "image": Image.fromarray(overlay),
                    "label": label,
                    "confidence": max_confidence,
                })

        return overlays
