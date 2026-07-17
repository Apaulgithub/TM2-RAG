

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import LLM_BACKEND

from src.retrieval.query_processor import ProcessedQuery
from src.retrieval.time_aware_retriever import RetrievalResult


# ============================================================
# TEMPORAL SUMMARY
# ============================================================

def build_temporal_summary(
    retrieval_result: RetrievalResult,
):

    windows = retrieval_result.windows
    trajectories = retrieval_result.trajectories

    trend_scores = {
        "worsening": 0,
        "improving": 0,
        "stable": 0,
    }

    emerging = set()
    resolving = set()

    # --------------------------------------------------------
    # WINDOWS (stronger local signal)
    # --------------------------------------------------------

    for w in windows:

        trend = w.get("trend", "unknown")

        if trend in trend_scores:
            trend_scores[trend] += 2

        deltas = w.get(
            "pathology_deltas",
            {}
        )

        for pathology, delta in deltas.items():

            if pathology == "No Finding":
                continue

            if delta > 0:
                emerging.add(pathology)

            elif delta < 0:
                resolving.add(pathology)

    # --------------------------------------------------------
    # TRAJECTORIES (global signal)
    # --------------------------------------------------------

    for t in trajectories:

        trend = t.get("trend", "unknown")

        if trend in trend_scores:
            trend_scores[trend] += 1

    # --------------------------------------------------------
    # DOMINANT TREND
    # --------------------------------------------------------

    if sum(trend_scores.values()) > 0:
        dominant_trend = max(
            trend_scores,
            key=trend_scores.get
        )
    else:
        dominant_trend = "unknown"

    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    if dominant_trend == "worsening":
        risk_level = "high"

    elif dominant_trend == "improving":
        risk_level = "low"

    elif dominant_trend == "stable":
        risk_level = "low"

    else:
        risk_level = "unknown"

    return {
        "trend": dominant_trend,
        "risk_level": risk_level,
        "emerging": list(emerging)[:5],
        "resolving": list(resolving)[:5],
    }


# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_context(
    query: ProcessedQuery,
    retrieval_result: RetrievalResult,
    top_k: int = 3,
):

    lines = []

    # ========================================================
    # CURRENT QUERY
    # ========================================================

    lines.append("[CURRENT CASE]")

    if query.findings_text:
        lines.append(
            f"Findings: {query.findings_text}"
        )

    if query.impression_text:
        lines.append(
            f"Impression: {query.impression_text}"
        )

    positives = [
        p for p, v
        in query.pathology_labels.items()
        if v == 1
    ]

    if positives:
        lines.append(
            f"Detected Pathologies: "
            f"{', '.join(positives)}"
        )

    lines.append(
        f"Temporal Status: "
        f"{query.temporal_status}"
    )

    lines.append(
        f"Query Mode: "
        f"{query.query_mode}"
    )

    # ========================================================
    # RETRIEVED REPORTS
    # ========================================================

    if retrieval_result.reports:

        lines.append("\n[SIMILAR REPORTS]")

        for r in retrieval_result.reports[:top_k]:

            findings = r.get(
                "findings",
                ""
            )

            impression = r.get(
                "impression",
                ""
            )

            if findings:
                lines.append(
                    f"Findings: {findings[:400]}"
                )

            if impression:
                lines.append(
                    f"Impression: {impression[:250]}"
                )

    # ========================================================
    # TEMPORAL WINDOWS
    # ========================================================

    if retrieval_result.windows:

        lines.append("\n[TEMPORAL PROGRESSION]")

        for w in retrieval_result.windows[:top_k]:

            trend = w.get(
                "trend",
                "unknown"
            )

            text = w.get(
                "text",
                ""
            )

            lines.append(
                f"({trend}) {text[:300]}"
            )

    # ========================================================
    # TRAJECTORIES
    # ========================================================

    if retrieval_result.trajectories:

        lines.append(
            "\n[LONGITUDINAL TRAJECTORIES]"
        )

        for t in retrieval_result.trajectories[:2]:

            trend = t.get(
                "trend",
                "unknown"
            )

            length = t.get(
                "length",
                0
            )

            lines.append(
                f"Trend: {trend} "
                f"| Length: {length}"
            )

    return "\n".join(lines)


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt(
    query: ProcessedQuery,
    retrieval_result: RetrievalResult,
):

    temporal = build_temporal_summary(
        retrieval_result
    )

    context = build_context(
        query,
        retrieval_result,
    )

    prompt = f"""
You are an expert thoracic radiologist AI.

Generate a clinically accurate chest X-ray report
using:
- current patient findings
- retrieved multimodal reports
- temporal progression patterns
- longitudinal disease trajectories

{context}

[TEMPORAL SUMMARY]

Dominant Trend: {temporal['trend']}
Risk Level: {temporal['risk_level']}

Emerging Findings:
{', '.join(temporal['emerging']) if temporal['emerging'] else 'None'}

Resolving Findings:
{', '.join(temporal['resolving']) if temporal['resolving'] else 'None'}

[INSTRUCTIONS]

Generate:
1. FINDINGS
2. IMPRESSION
3. TEMPORAL RISK ASSESSMENT
4. RECOMMENDED FOLLOW-UP

Rules:
- be concise
- avoid hallucinations
- reflect temporal progression
- ensure pathology consistency
- maintain radiology style
"""

    return prompt.strip()


# ============================================================
# VALIDATION
# ============================================================

def validate_report(text: str):

    if not text:
        return False

    if len(text.strip()) < 40:
        return False

    required = [
        "FINDINGS",
        "IMPRESSION",
    ]

    text_upper = text.upper()

    return all(
        r in text_upper
        for r in required
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_report(text: str):

    if not text:
        return text

    text = text.strip()

    replacements = {
        "FINDINGS:": "\nFINDINGS:\n",
        "IMPRESSION:": "\nIMPRESSION:\n",
        "TEMPORAL": "\nTEMPORAL",
        "RECOMMENDED": "\nRECOMMENDED",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.strip()


# ============================================================
# RETRIEVAL FALLBACK
# ============================================================

def build_retrieval_fallback(
    query: ProcessedQuery,
    retrieval_result: RetrievalResult,
):

    reports = retrieval_result.reports[:3]

    findings = []
    impressions = []

    for r in reports:

        if r.get("findings"):
            findings.append(r["findings"])

        if r.get("impression"):
            impressions.append(r["impression"])

    findings_text = (
        " ".join(findings[:3])
        if findings
        else "No acute cardiopulmonary abnormality."
    )

    impression_text = (
        " ".join(impressions[:2])
        if impressions
        else "No acute findings."
    )

    temporal = build_temporal_summary(
        retrieval_result
    )

    return f"""
FINDINGS:
{findings_text}

IMPRESSION:
{impression_text}

TEMPORAL RISK ASSESSMENT:
Trend: {temporal['trend']}
Risk: {temporal['risk_level']}

RECOMMENDED FOLLOW-UP:
Clinical correlation and follow-up imaging if indicated.
""".strip()


# ============================================================
# TEMPLATE FALLBACK
# ============================================================

def build_template_fallback(
    query: ProcessedQuery,
):

    positives = [
        p for p, v
        in query.pathology_labels.items()
        if v == 1
    ]

    if positives:

        findings = (
            f"Findings consistent with: "
            f"{', '.join(positives)}."
        )

        impression = (
            f"Abnormal chest radiograph "
            f"demonstrating "
            f"{', '.join(positives)}."
        )

    else:

        findings = (
            "No acute cardiopulmonary abnormality."
        )

        impression = "Normal chest radiograph."

    return f"""
FINDINGS:
{findings}

IMPRESSION:
{impression}

TEMPORAL RISK ASSESSMENT:
Insufficient temporal evidence.

RECOMMENDED FOLLOW-UP:
Clinical follow-up as indicated.
""".strip()


# ============================================================
# MAIN GENERATION
# ============================================================

def generate_report(
    query: ProcessedQuery,
    retrieval_result: RetrievalResult,
    backend: Optional[str] = None,
    debug: bool = False,
):

    backend = backend or LLM_BACKEND

    # ========================================================
    # TEMPLATE MODE
    # ========================================================

    if backend == "template":

        return build_template_fallback(
            query
        )

    # ========================================================
    # BUILD PROMPT
    # ========================================================

    prompt = build_prompt(
        query,
        retrieval_result,
    )

    if debug:

        print("\n[GENERATOR]")
        print(f"Backend: {backend}")

        print("\n[PROMPT PREVIEW]")
        print(prompt[:1200])

    # ========================================================
    # VLM GENERATION
    # ========================================================

    try:

        from src.generation.vlm_backends import (
            get_vlm_backend,
            list_available_models,
        )

        if backend not in list_available_models():

            print(
                f"[WARN] Unknown backend "
                f"{backend}"
            )

            return build_template_fallback(
                query
            )

        vlm = get_vlm_backend(backend)

        image_paths = (
            [query.image_path]
            if query.image_path
            else None
        )

        output = vlm.generate(
            prompt=prompt,
            image_paths=image_paths,
        )

    except Exception as e:

        print(
            f"[WARN] Generation failed: {e}"
        )

        output = None

    # ========================================================
    # VALIDATE
    # ========================================================

    if output and validate_report(output):

        return normalize_report(output)

    # ========================================================
    # FALLBACK 1
    # ========================================================

    fallback = build_retrieval_fallback(
        query,
        retrieval_result,
    )

    if validate_report(fallback):

        return fallback

    # ========================================================
    # FALLBACK 2
    # ========================================================

    return build_template_fallback(
        query
    )