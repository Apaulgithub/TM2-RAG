"""
Phase 5B: Time-Aware Retriever with Multimodal Support (TD-RAG + CLAF)

Implements TRIPLE RETRIEVAL across THREE MODALITIES:

Multimodal retrieval (text / image / fused)
Temporal-aware reranking
Clinical-aware reranking
Improved context building
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (
    DATA_DIR,
    TOP_K_REPORTS,
    TOP_K_WINDOWS,
    TOP_K_TRAJECTORIES,
)
from src.embeddings.vector_store import VectorStore
from src.retrieval.query_processor import ProcessedQuery


# ──────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────

@dataclass
class RetrievalResult:
    query_key: str
    reports: List = field(default_factory=list)
    windows: List = field(default_factory=list)
    trajectories: List = field(default_factory=list)
    context: str = ""


# ──────────────────────────────────────────────
# RETRIEVER
# ──────────────────────────────────────────────

class TimeAwareRetriever:

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self._loaded = False

    # ─────────────────────────────────────────
    # LOAD
    # ─────────────────────────────────────────

    def load(self):
        print("Loading Retriever...")

        store_dir = self.data_dir / "vector_store"

        from src.config import (
            EMBEDDING_DIM,
            WINDOW_EMBEDDING_DIM,
            TRAJECTORY_EMBEDDING_DIM
        )

        # vector stores
        self.report_store = VectorStore("reports", EMBEDDING_DIM)
        self.report_store.load(store_dir)

        self.window_store = VectorStore("windows", WINDOW_EMBEDDING_DIM)
        self.window_store.load(store_dir)

        self.trajectory_store = VectorStore("trajectories", TRAJECTORY_EMBEDDING_DIM)
        self.trajectory_store.load(store_dir)

        # fused (optional)
        self.fused_store = None
        if (store_dir / "fused_reports_meta.json").exists():
            self.fused_store = VectorStore("fused_reports", 512)
            self.fused_store.load(store_dir)

        # metadata
        with open(self.data_dir / "parsed_reports.json") as f:
            self.parsed_reports = json.load(f)

        with open(self.data_dir / "temporal_windows.json") as f:
            self.temporal_windows = json.load(f)

        with open(self.data_dir / "trajectories.json") as f:
            self.trajectories = json.load(f)

        self._loaded = True
        print("Retriever ready.")

    # ─────────────────────────────────────────
    # MAIN RETRIEVAL
    # ─────────────────────────────────────────

    def retrieve(self, query: ProcessedQuery) -> RetrievalResult:

        if not self._loaded:
            self.load()

        # ── SELECT REPORT QUERY VECTOR ─────────
        if query.fused_query is not None and self.fused_store:
            rpt_store = self.fused_store
            rpt_query = query.fused_query
        else:
            rpt_store = self.report_store
            rpt_query = query.report_embedding

        # ─────────────────────────────────────
        # REPORT RETRIEVAL
        # ─────────────────────────────────────
        reports = []
        if rpt_query is not None:
            results = rpt_store.search(rpt_query, TOP_K_REPORTS)

            for key, score, _ in results:
                r = self.parsed_reports.get(key, {})
                reports.append({
                    "key": key,
                    "score": score,
                    "findings": r.get("findings_text", ""),
                    "impression": r.get("impression_text", ""),
                })

        # ─────────────────────────────────────
        # WINDOW RETRIEVAL
        # ─────────────────────────────────────
        windows = []
        if query.window_query is not None:
            results = self.window_store.search(query.window_query, TOP_K_WINDOWS)

            for key, score, meta in results:
                w = self.temporal_windows.get(key, {})
                windows.append({
                    "key": key,
                    "score": score,
                    "trend": meta.get("trend", w.get("local_trend")),
                    "text": w.get("serialised_text", ""),
                })

        # ─────────────────────────────────────
        # TRAJECTORY RETRIEVAL
        # ─────────────────────────────────────
        trajectories = []
        if query.trajectory_query is not None:
            results = self.trajectory_store.search(
                query.trajectory_query,
                TOP_K_TRAJECTORIES
            )

            for pid, score, meta in results:
                traj = self.trajectories.get(pid, {})
                trajectories.append({
                    "patient_id": pid,
                    "score": score,
                    "trend": meta.get("trend", traj.get("overall_trend")),
                    "length": meta.get("length", traj.get("duration_studies")),
                })

        # ─────────────────────────────────────
        # RE-RANK
        # ─────────────────────────────────────
        reports = self._rerank_reports(query, reports)
        windows = self._rerank_windows(query, windows)
        trajectories = self._rerank_trajectories(query, trajectories)

        # ─────────────────────────────────────
        # CONTEXT
        # ─────────────────────────────────────
        context = self._build_context(query, reports, windows, trajectories)

        return RetrievalResult(
            query_key=query.study_key,
            reports=reports,
            windows=windows,
            trajectories=trajectories,
            context=context,
        )

    # ─────────────────────────────────────────
    # HELPER: LABEL OVERLAP
    # ─────────────────────────────────────────

    def _label_overlap(self, q_labels, r_labels):
        if not q_labels or not r_labels:
            return 0.0

        match = 0
        total = 0

        for k in q_labels:
            if k == "No Finding":
                continue
            total += 1
            if q_labels.get(k, 0) == 1 and r_labels.get(k, 0) == 1:
                match += 1

        return match / max(total, 1)

    # ─────────────────────────────────────────
    # RE-RANK REPORTS
    # ─────────────────────────────────────────

    def _rerank_reports(self, query, reports):

        for r in reports:
            bonus = 0.0

            # temporal hint
            if query.temporal_status == "worsening" and "worsening" in r["impression"]:
                bonus += 0.05

            # NEW: pathology overlap
            r_labels = self.parsed_reports.get(r["key"], {}).get("pathology_labels", {})
            overlap = self._label_overlap(query.pathology_labels, r_labels)

            bonus += 0.15 * overlap

            r["score"] += bonus

        return sorted(reports, key=lambda x: x["score"], reverse=True)

    # ─────────────────────────────────────────
    # RE-RANK WINDOWS
    # ─────────────────────────────────────────

    def _rerank_windows(self, query, windows):

        for w in windows:
            if w["trend"] == query.temporal_status:
                w["score"] += 0.1

        return sorted(windows, key=lambda x: x["score"], reverse=True)

    # ─────────────────────────────────────────
    # RE-RANK TRAJECTORIES
    # ─────────────────────────────────────────

    def _rerank_trajectories(self, query, trajs):

        for t in trajs:
            if t["trend"] == query.temporal_status:
                t["score"] += 0.1
            elif t["trend"] in ["improving", "worsening"]:
                t["score"] += 0.05

        return sorted(trajs, key=lambda x: x["score"], reverse=True)

    # ─────────────────────────────────────────
    # CONTEXT BUILDER
    # ─────────────────────────────────────────

    def _build_context(self, query, reports, windows, trajs):

        lines = []

        # WINDOWS
        if windows:
            lines.append("=== SIMILAR TEMPORAL WINDOWS ===")
            for w in windows[:3]:
                lines.append(f"- ({w['trend']}) {w['text'][:200]}")

        # REPORTS
        if reports:
            lines.append("\n=== SIMILAR REPORTS ===")
            for r in reports[:3]:
                lines.append(f"- {r['impression'][:200]}")

                labels = self.parsed_reports.get(r["key"], {}).get("pathology_labels", {})
                positives = [k for k, v in labels.items() if v == 1]

                if positives:
                    lines.append(f"  → Findings: {', '.join(positives)}")

        # TRAJECTORIES
        if trajs:
            lines.append("\n=== SIMILAR TRAJECTORIES ===")
            for t in trajs[:2]:
                lines.append(f"- trend={t['trend']}, length={t['length']}")

        # QUERY
        lines.append("\n=== QUERY ===")

        if query.pathology_labels:
            q_pos = [k for k, v in query.pathology_labels.items() if v == 1]
            if q_pos:
                lines.append(f"Findings: {', '.join(q_pos)}")

        lines.append(query.raw_text)

        return "\n".join(lines)