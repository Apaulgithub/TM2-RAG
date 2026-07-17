"""
VLM Comparative Benchmark — T-RAG Report Generation

Runs each VLM model on the test split and collects NLG + clinical metrics:
    - BLEU
    - ROUGE-L
    - CheXpert F1 (pathology accuracy)
    - Temporal Consistency

Usage:
    # All models
    python run_pipeline.py --benchmark

    # Single model
    python run_pipeline.py --benchmark --backend qwen2.5-vl-3b

    # With sample limit
    python run_pipeline.py --benchmark --max-samples 50

    # Standalone
    python -m src.evaluation.benchmark --max-samples 20
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import DATA_DIR, CHECKPOINT_DIR, VLM_MODELS, VLM_QUANTIZE


# ─────────────────────────────────────────────
# SAMPLE BUILDER
# ─────────────────────────────────────────────

def _build_test_samples(max_samples: Optional[int] = None):
    """
    Build (patient_id, study_key, ref_text, image_path) tuples from the test split.
    """

    from src.training.data_split import load_splits

    splits = load_splits()
    test_pids = splits["test"]

    tp = json.load(open(DATA_DIR / "temporal_patients.json"))
    parsed = json.load(open(DATA_DIR / "parsed_reports.json"))

    samples = []

    for pid in test_pids:
        pdata = tp.get(pid)
        if not pdata:
            continue

        study = pdata["studies"][-1]
        key = f"{pid}_{study['study_id']}"

        ref = parsed.get(key)
        if not ref:
            continue

        ref_text = (
            ref.get("findings_text", "") + " " +
            ref.get("impression_text", "")
        ).strip()

        if not ref_text:
            continue

        # Try to find image path for multimodal testing
        image_path = None
        images = study.get("images", [])
        if images:
            from src.config import IMAGES_DIR
            first_img = images[0]
            dicom_id = first_img.get("dicom_id", "")
            if dicom_id:
                prefix = pid[:3] if pid.startswith("p") else f"p{pid[:2]}"
                candidate = IMAGES_DIR / prefix / pid / f"s{study['study_id']}" / f"{dicom_id}.jpg"
                if candidate.exists():
                    image_path = str(candidate)

        samples.append({
            "patient_id": pid,
            "study_key": key,
            "ref_text": ref_text,
            "image_path": image_path,
        })

    if max_samples:
        samples = samples[:max_samples]

    return samples


# ─────────────────────────────────────────────
# SINGLE MODEL EVALUATION
# ─────────────────────────────────────────────

def _evaluate_model(
    model_name: str,
    samples: List[Dict],
    quantize: str = VLM_QUANTIZE,
) -> Dict:
    """Run a single VLM model on all samples, return metrics dict."""

    from src.retrieval.query_processor import process_query
    from src.retrieval.time_aware_retriever import TimeAwareRetriever
    from src.generation.generator import generate_report
    from src.training.evaluate import (
        compute_bleu,
        compute_rouge_l,
        compute_chexpert_f1,
        temporal_consistency,
    )

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name} [{quantize}]")
    print(f"Samples: {len(samples)}")
    print(f"{'='*60}")

    # Load retriever once
    retriever = TimeAwareRetriever(DATA_DIR)
    retriever.load()

    refs = []
    hyps = []
    temporal_scores = []
    times = []
    errors = 0

    for i, sample in enumerate(samples):
        try:
            t0 = time.time()

            query = process_query(
                report_text=sample["ref_text"],
                study_key=sample["study_key"],
                image_path=sample.get("image_path"),
            )

            result = retriever.retrieve(query)

            hyp = generate_report(
                query,
                result,
                backend=model_name,
            )

            elapsed = time.time() - t0
            times.append(elapsed)

            refs.append(sample["ref_text"])
            hyps.append(hyp)

            temporal_scores.append(
                temporal_consistency(query, hyp)
            )

            if (i + 1) % 10 == 0:
                print(f"  [{model_name}] {i+1}/{len(samples)} done  "
                      f"({elapsed:.1f}s/sample)")

        except Exception as e:
            errors += 1
            print(f"  [{model_name}] Error on {sample['study_key']}: {e}")

    # Compute metrics
    if not refs:
        print(f"  [{model_name}] No successful samples!")
        return {"error": "no successful samples"}

    import numpy as np

    bleu = compute_bleu(refs, hyps)
    rouge = compute_rouge_l(refs, hyps)
    chex = compute_chexpert_f1(refs, hyps)
    temp_con = float(np.mean(temporal_scores)) if temporal_scores else 0.0
    avg_time = float(np.mean(times)) if times else 0.0

    metrics = {
        "model": model_name,
        "quantization": quantize,
        "bleu": round(bleu, 4),
        "rouge_l": round(rouge, 4) if rouge is not None else None,
        "chexpert_f1": round(chex, 4),
        "temporal_consistency": round(temp_con, 4),
        "num_samples": len(refs),
        "num_errors": errors,
        "avg_time_per_sample_s": round(avg_time, 2),
    }

    print(f"\n  [{model_name}] Results:")
    for k, v in metrics.items():
        if k != "model":
            print(f"    {k}: {v}")

    return metrics


# ─────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────

def _print_comparison_table(all_results: Dict[str, Dict]):
    """Print a formatted comparison table of all model results."""

    print("\n" + "=" * 80)
    print("VLM COMPARATIVE BENCHMARK RESULTS")
    print("=" * 80)

    metric_keys = ["bleu", "rouge_l", "chexpert_f1", "temporal_consistency",
                   "avg_time_per_sample_s"]

    # Header
    header = f"{'Model':<22}"
    for mk in metric_keys:
        label = mk.replace("_", " ").title()
        if len(label) > 12:
            label = label[:12]
        header += f" {label:>12}"
    print(header)
    print("-" * 80)

    # Rows
    for model_name, metrics in all_results.items():
        row = f"{model_name:<22}"
        for mk in metric_keys:
            val = metrics.get(mk)
            if val is None:
                row += f" {'N/A':>12}"
            elif isinstance(val, float):
                row += f" {val:>12.4f}"
            else:
                row += f" {str(val):>12}"
        print(row)

    print("=" * 80)

    # Also include the retrieval baseline
    print("\nNote: Run with --backend retrieval for the retrieval-only baseline.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_benchmark(
    models: Optional[List[str]] = None,
    views: Optional[List[str]] = None,
    approaches: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    quantize: str = VLM_QUANTIZE,
):
    """
    Run the full VLM comparative benchmark.

    Args:
        models:      List of model names to evaluate (default: all in VLM_MODELS)
        views:       Reserved for future view-type filtering
        approaches:  Reserved for future approach-level filtering
        max_samples: Limit number of test samples
        quantize:    Quantization level ("fp16", "8bit", "4bit")
    """

    print("\n" + "=" * 60)
    print("T-RAG VLM Comparative Benchmark")
    print("=" * 60)

    # Determine which models to benchmark
    if models:
        model_list = [m for m in models if m in VLM_MODELS]
        if not model_list:
            # Maybe user passed "retrieval" as a baseline model
            model_list = models
    else:
        model_list = list(VLM_MODELS.keys())

    print(f"Models: {model_list}")
    print(f"Quantization: {quantize}")
    print(f"Max samples: {max_samples or 'all'}")

    # Build test samples
    samples = _build_test_samples(max_samples)
    print(f"Test samples loaded: {len(samples)}")

    if not samples:
        print("[ERROR] No test samples found. Run data_split first.")
        return

    # Always include retrieval baseline
    all_results = {}

    if "retrieval" in model_list:
        model_list.remove("retrieval")
        all_results["retrieval"] = _evaluate_model_baseline(samples)

    # Run each VLM
    for model_name in model_list:
        try:
            metrics = _evaluate_model(model_name, samples, quantize)
            all_results[model_name] = metrics
        except Exception as e:
            print(f"\n[ERROR] Failed to evaluate {model_name}: {e}")
            all_results[model_name] = {"error": str(e)}

    # Save results
    output_path = DATA_DIR / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print comparison table
    _print_comparison_table(all_results)

    return all_results


def _evaluate_model_baseline(samples: List[Dict]) -> Dict:
    """Run the retrieval-only baseline (no LLM) for comparison."""

    return _evaluate_model("retrieval", samples, quantize="none")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="T-RAG VLM Comparative Benchmark"
    )
    parser.add_argument("--backend", type=str, default=None,
                        help="Single model to evaluate (default: all)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of test samples")
    parser.add_argument("--quantize", type=str, default=VLM_QUANTIZE,
                        choices=["fp16", "8bit", "4bit"],
                        help="Quantization level")

    args = parser.parse_args()

    models = [args.backend] if args.backend else None

    run_benchmark(
        models=models,
        max_samples=args.max_samples,
        quantize=args.quantize,
    )
