"""
Evaluation module for T-RAG report generation.

Stronger clinical evaluation
Temporal consistency metric 
Improved retrieval evaluation
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DATA_DIR, CHECKPOINT_DIR, PATHOLOGIES


# ─────────────────────────────────────────────
# TEXT -> LABELS (IMPROVED)
# ─────────────────────────────────────────────

NEGATIONS = ["no ", "without ", "absence of "]


def _extract_labels(text):
    text = text.lower()
    labels = {}

    for p in PATHOLOGIES:
        pname = p.lower()

        if pname in text:
            # check simple negation
            negated = any(neg + pname in text for neg in NEGATIONS)
            labels[p] = 0 if negated else 1
        else:
            labels[p] = 0

    return labels


# ─────────────────────────────────────────────
# CHEXPERT F1 (IMPROVED)
# ─────────────────────────────────────────────

def compute_chexpert_f1(refs, hyps):

    stats = {p: {"tp": 0, "fp": 0, "fn": 0} for p in PATHOLOGIES}

    for ref, hyp in zip(refs, hyps):
        rl = _extract_labels(ref)
        hl = _extract_labels(hyp)

        for p in PATHOLOGIES:
            if rl[p] == 1 and hl[p] == 1:
                stats[p]["tp"] += 1
            elif rl[p] == 0 and hl[p] == 1:
                stats[p]["fp"] += 1
            elif rl[p] == 1 and hl[p] == 0:
                stats[p]["fn"] += 1

    f1s = {}
    for p in PATHOLOGIES:
        tp, fp, fn = stats[p]["tp"], stats[p]["fp"], stats[p]["fn"]

        if tp + fp + fn == 0:
            f1s[p] = float("nan")
        else:
            prec = tp / (tp + fp) if (tp + fp) else 0
            rec = tp / (tp + fn) if (tp + fn) else 0
            f1s[p] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    macro = np.nanmean(list(f1s.values()))
    return float(macro)


# ─────────────────────────────────────────────
# TEMPORAL METRIC (NEW)
# ─────────────────────────────────────────────

def _extract_trend(text):
    text = text.lower()

    if "worsen" in text or "increase" in text:
        return "worsening"
    if "improve" in text or "decrease" in text:
        return "improving"
    if "stable" in text:
        return "stable"

    return "unknown"


def temporal_consistency(query, generated_text):
    pred = _extract_trend(generated_text)
    true = query.temporal_status

    return 1.0 if pred == true else 0.0


# ─────────────────────────────────────────────
# RETRIEVAL SCORE (IMPROVED)
# ─────────────────────────────────────────────

def retrieval_score(query_labels, retrieved_reports, top_k=3):

    query_pos = {p for p, v in query_labels.items() if v == 1}

    if not query_pos:
        return 0.0

    score = 0

    for r in retrieved_reports[:top_k]:
        txt = r.get("impression", "").lower()
        matches = sum(1 for p in query_pos if p.lower() in txt)
        score += matches / max(len(query_pos), 1)

    return score / top_k


# ─────────────────────────────────────────────
# BLEU / ROUGE (UNCHANGED)
# ─────────────────────────────────────────────

def compute_bleu(refs, hyps):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    r = [[r.split()] for r in refs]
    h = [h.split() for h in hyps]

    return corpus_bleu(r, h, smoothing_function=smooth)


def compute_rouge_l(refs, hyps):
    try:
        from rouge_score import rouge_scorer
    except:
        return None

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)]
    return float(np.mean(scores))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def evaluate(max_samples=None):

    from src.training.data_split import load_splits
    from src.retrieval.query_processor import process_query
    from src.retrieval.time_aware_retriever import TimeAwareRetriever
    from src.generation.generator import generate_report

    print("\n=== Evaluation (UPDATED) ===")

    splits = load_splits()
    test_pids = splits["test"]

    tp = json.load(open(DATA_DIR / "temporal_patients.json"))
    parsed = json.load(open(DATA_DIR / "parsed_reports.json"))

    pairs = []

    for pid in test_pids:
        pdata = tp.get(pid)
        if not pdata:
            continue

        study = pdata["studies"][-1]
        key = f"{pid}_{study['study_id']}"

        ref = parsed.get(key)
        if not ref:
            continue

        text = (ref.get("findings_text", "") + " " +
                ref.get("impression_text", "")).strip()

        if text:
            pairs.append((pid, key, text))

    if max_samples:
        pairs = pairs[:max_samples]

    print(f"Samples: {len(pairs)}")

    retriever = TimeAwareRetriever(DATA_DIR)
    retriever.load()

    refs, hyps = [], []
    temporal_scores = []
    retrieval_scores = []

    for pid, key, ref_text in pairs:

        try:
            query = process_query(report_text=ref_text, study_key=key)
            result = retriever.retrieve(query)

            hyp = generate_report(query, result)

            refs.append(ref_text)
            hyps.append(hyp)

            temporal_scores.append(
                temporal_consistency(query, hyp)
            )

            retrieval_scores.append(
                retrieval_score(query.pathology_labels, result.reports)
            )

        except Exception as e:
            print(f"Error on {key}: {e}")

    bleu = compute_bleu(refs, hyps)
    rouge = compute_rouge_l(refs, hyps)
    chex = compute_chexpert_f1(refs, hyps)

    print("\nResults")
    print("=" * 40)

    print(f"BLEU: {bleu:.4f}")
    print(f"ROUGE-L: {rouge:.4f}" if rouge else "ROUGE unavailable")
    print(f"CheXpert F1: {chex:.4f}")
    print(f"Temporal Consistency: {np.mean(temporal_scores):.4f}")
    print(f"Retrieval Score: {np.mean(retrieval_scores):.4f}")

    out = {
        "bleu": bleu,
        "rouge": rouge,
        "chexpert_f1": chex,
        "temporal_consistency": float(np.mean(temporal_scores)),
        "retrieval_score": float(np.mean(retrieval_scores)),
    }

    with open(CHECKPOINT_DIR / "eval_results.json", "w") as f:
        json.dump(out, f, indent=2)

    return out

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    evaluate(max_samples=args.max_samples)