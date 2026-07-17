"""
Phase 2B: Temporal Change Detector — Detect temporal language in radiology
reports that indicates improvement, worsening, stability, or new findings.
"""

import re
from dataclasses import dataclass


# ─────────────────────────────────────────────
# TEMPORAL CHANGE PATTERNS
# ─────────────────────────────────────────────

IMPROVE = [r"improv", r"resolv", r"decreas", r"reduced", r"less", r"clearing"]
WORSEN  = [r"worsen", r"increas", r"progress", r"larger", r"worse"]
STABLE  = [r"unchanged", r"stable", r"no\s+change", r"similar", r"persistent"]
NEW     = [r"new", r"developed", r"appeared", r"interval\s+development"]
RESOLVED= [r"resolved", r"no\s+longer", r"disappeared", r"cleared"]

# ─────────────────────────────────────────────
# PATHOLOGY KEYWORDS (minimal mapping)
# ─────────────────────────────────────────────

PATHOLOGY_TERMS = {
    "Atelectasis": ["atelectasis", "collapse"],
    "Cardiomegaly": ["cardiomegaly", "heart enlarged"],
    "Consolidation": ["consolidation"],
    "Edema": ["edema", "congestion"],
    "Enlarged Cardiomediastinum": ["mediastinum"],
    "Fracture": ["fracture"],
    "Lung Lesion": ["nodule", "mass", "lesion"],
    "Lung Opacity": ["opacity", "infiltrate", "ground glass"],
    "No Finding": ["normal", "clear"],
    "Pleural Effusion": ["effusion"],
    "Pleural Other": ["pleural thickening"],
    "Pneumonia": ["pneumonia"],
    "Pneumothorax": ["pneumothorax"],
    "Support Devices": ["tube", "line", "catheter", "device"],
}


# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────

def split_sentences(text):
    return [s.strip() for s in re.split(r"[.\n]+", text) if s.strip()]


def detect_trend(sentence):
    s = sentence.lower()

    if any(re.search(p, s) for p in RESOLVED):
        return "resolved"
    if any(re.search(p, s) for p in IMPROVE):
        return "improving"
    if any(re.search(p, s) for p in WORSEN):
        return "worsening"
    if any(re.search(p, s) for p in STABLE):
        return "stable"
    if any(re.search(p, s) for p in NEW):
        return "new_finding"

    return None


def match_pathologies(sentence, pathology_labels):
    s = sentence.lower()
    matched = []

    for p, label in pathology_labels.items():
        if label == 0:
            continue

        terms = PATHOLOGY_TERMS.get(p, [])
        for t in terms:
            if t in s:
                matched.append(p)
                break

    return matched


# ─────────────────────────────────────────────
# OUTPUT STRUCTURE
# ─────────────────────────────────────────────

@dataclass
class TemporalAnalysis:
    temporal_status: str
    confidence: float
    per_pathology_trends: dict
    has_comparison: bool
    change_phrases: list

    def to_dict(self):
        return self.__dict__


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────

def detect_temporal_changes(report_text: str, pathology_labels: dict):
    sentences = split_sentences(report_text)

    global_scores = {
        "improving": 0,
        "worsening": 0,
        "stable": 0,
        "new_finding": 0,
        "resolved": 0,
    }

    per_pathology = {}
    phrases = []

    for sent in sentences:
        trend = detect_trend(sent)
        if not trend:
            continue

        phrases.append(sent)

        # update global score
        global_scores[trend] += 1

        # match pathologies in same sentence
        matched = match_pathologies(sent, pathology_labels)

        for p in matched:
            per_pathology[p] = trend

    # ───────── GLOBAL STATUS ─────────
    total = sum(global_scores.values())

    if total == 0:
        return TemporalAnalysis(
            temporal_status="unknown",
            confidence=0.0,
            per_pathology_trends={},
            has_comparison=("comparison" in report_text.lower()),
            change_phrases=[],
        )

    dominant = max(global_scores, key=global_scores.get)
    confidence = global_scores[dominant] / total

    return TemporalAnalysis(
        temporal_status=dominant,
        confidence=round(confidence, 3),
        per_pathology_trends=per_pathology,
        has_comparison=("comparison" in report_text.lower()),
        change_phrases=list(set(phrases)),
    )