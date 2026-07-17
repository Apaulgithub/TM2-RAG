"""
Phase 2A: Report Parser — Extract structured sections and pathology labels
from free-text MIMIC-CXR radiology reports.

This module provides:
1. Section extraction: Parse FINDINGS, IMPRESSION, INDICATION, etc.
2. Structured Pathology labeling: Detect 14 CheXpert pathologies by
    using the CheXbert model on the report text. 
3. Structured text augmentation: Store raw text, sections, and labels in a unified format.
"""

import re
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# SECTION EXTRACTION
# ─────────────────────────────────────────────

SECTION_HEADERS = [
    "EXAMINATION","INDICATION","HISTORY","TECHNIQUE","COMPARISON",
    "FINDINGS","IMPRESSION","CONCLUSION","RECOMMENDATION",
    "NOTIFICATION","WET READ","FINAL REPORT",
]

HEADER_REGEX = re.compile(
    rf"(?:^|\s)({'|'.join(map(re.escape, SECTION_HEADERS))})\s*[:.]?\s*",
    re.IGNORECASE | re.MULTILINE,
)


def extract_sections(text: str):
    text = text.strip()
    sections = {"full_text": text}

    if not text:
        return sections

    matches = list(HEADER_REGEX.finditer(text))

    if not matches:
        sections["findings"] = text
        return sections

    for i, m in enumerate(matches):
        name = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        if name in ("conclusion", "impression"):
            name = "impression"
        elif name in ("history", "indication"):
            name = "indication"

        sections[name] = content

    return sections


# ─────────────────────────────────────────────
# STRUCTURED LABEL TEXT
# ─────────────────────────────────────────────

def labels_to_text(labels: dict):
    """
    Convert pathology labels -> text string
    """
    parts = []

    for k, v in labels.items():
        if v == 1:
            parts.append(f"{k}: present")
        else:
            parts.append(f"{k}: absent")

    return "; ".join(parts)


# ─────────────────────────────────────────────
# OUTPUT STRUCTURE
# ─────────────────────────────────────────────

@dataclass
class ParsedReport:
    study_key: str
    sections: dict = field(default_factory=dict)
    pathology_labels: dict = field(default_factory=dict)

    findings_text: str = ""
    impression_text: str = ""
    raw_text: str = ""

    structured_label_text: str = ""
    augmented_text: str = ""

    def to_dict(self):
        return self.__dict__


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def parse_report(study_key: str, report_text: str, pathology_labels: dict):

    sections = extract_sections(report_text)

    findings = sections.get("findings", "")
    impression = sections.get("impression", "")

    # ─────────────────────────────────────
    # NEW: structured label text
    # ─────────────────────────────────────
    structured_text = labels_to_text(pathology_labels)

    # ─────────────────────────────────────
    # Augmented text
    # ─────────────────────────────────────
    augmented = f"{report_text}\n\n[Structured Findings]\n{structured_text}"

    return ParsedReport(
        study_key=study_key,
        sections=sections,
        pathology_labels=pathology_labels,
        findings_text=findings,
        impression_text=impression,
        raw_text=report_text,
        structured_label_text=structured_text,
        augmented_text=augmented,
    )