#!/usr/bin/env python3
"""Run the institutional investor casebook benchmark.

Loads PM cases, runs LLM inference (or mock), scores outputs against
golden answers, and prints a results table with Likert ratings.

Usage:
    python run_benchmark.py              # requires GPU + model
    python run_benchmark.py --mock       # no GPU needed (testing)
    python run_benchmark.py --model X    # custom HuggingFace model ID
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from investor_casebook.data.loader import CasebookLoader
from investor_casebook.runner import CasebookRunner
from investor_casebook.reasoning.scorer import CaseScorer


def print_results(report: dict) -> None:
    """Pretty-print per-case scores and aggregate summary."""
    per_case = report["per_case"]
    aggregate = report["aggregate"]

    print("\n" + "=" * 78)
    print("BENCHMARK RESULTS")
    print("=" * 78)

    # Per-case table
    header = f"{'ID':<16} {'Category':<22} {'Comp':>5} {'Num':>5} {'Str':>5} {'Ovr':>5} {'Rating'}"
    print(header)
    print("-" * 78)

    for case in per_case:
        print(
            f"{case['id']:<16} "
            f"{case['category']:<22} "
            f"{case['completeness']:>5.2f} "
            f"{case['numerical_accuracy']:>5.2f} "
            f"{case['structure']:>5.2f} "
            f"{case['overall']:>5.2f} "
            f"{case['likert']}/5 {case['likert_label']}"
        )

    # Aggregate
    print("-" * 78)
    print(f"{'AGGREGATE':<40} "
          f"{'':>5} {'':>5} {'':>5} "
          f"{aggregate['mean']:>5.2f} "
          f"{aggregate['likert_mean']:.1f}/5")

    print(f"\n  Cases: {aggregate['count']}  |  "
          f"Mean: {aggregate['mean']:.3f}  |  "
          f"Median: {aggregate['median']:.3f}  |  "
          f"Min: {aggregate['min']:.3f}  |  "
          f"Max: {aggregate['max']:.3f}")

    # Likert distribution
    dist = aggregate.get("likert_distribution", {})
    if dist:
        print("\n  Likert Distribution:")
        for label, count in dist.items():
            bar = "#" * count
            print(f"    {label:<22} {bar} ({count})")

    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="Run the institutional investor casebook benchmark."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("src/investor_casebook/data"),
        help="Path to data directory (default: src/investor_casebook/data)",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="sample_cases.jsonl",
        help="JSONL filename within data directory (default: sample_cases.jsonl)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (no GPU, placeholder outputs)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model ID (default: Meta-Llama-3-8B-Instruct)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark_results.jsonl"),
        help="Output JSONL path (default: results/benchmark_results.jsonl)",
    )

    args = parser.parse_args()

    # 1. Load cases
    print(f"Loading cases from {args.data / args.cases}...")
    loader = CasebookLoader(args.data)
    cases = loader.load_cases(args.cases)
    print(f"Loaded {len(cases)} cases.")

    # 2. Initialize runner
    runner = CasebookRunner(model_id=args.model, mock=args.mock)

    # 3. Run inference
    print(f"\nRunning inference on {len(cases)} cases...")
    results = runner.run_all_cases(cases)

    # 4. Score results
    scorer = CaseScorer()
    report = scorer.score_all(results)

    # 5. Print results
    print_results(report)

    # 6. Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for case_result, case_score in zip(results, report["per_case"]):
            record = {
                "timestamp": datetime.now().isoformat(),
                "model": args.model,
                "mock": args.mock,
                **case_result,
                "scores": case_score,
            }
            f.write(json.dumps(record) + "\n")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
