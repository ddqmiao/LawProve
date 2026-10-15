#!/usr/bin/env python
"""Fuse retrieval outputs with RRF and evaluate recall."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from law_grounding.data import read_jsonl, write_jsonl
from law_grounding.law_retrieval import DEFAULT_CUTOFFS, compute_retrieval_metrics, fuse_retrieval_rows


def parse_cutoffs(value: str) -> list[int]:
    cutoffs = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not cutoffs:
        raise argparse.ArgumentTypeError("at least one cutoff is required")
    return cutoffs


def parse_weights(value: str | None) -> list[float] | None:
    if not value:
        return None
    weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not weights:
        raise argparse.ArgumentTypeError("at least one weight is required")
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval-files",
        nargs="+",
        required=True,
        help="Retrieved JSONL files with aligned rows, e.g. bm25.top50.jsonl embedding.top100.jsonl",
    )
    parser.add_argument("--weights", type=parse_weights, help="Comma-separated source weights")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--cutoffs",
        type=parse_cutoffs,
        default=list(DEFAULT_CUTOFFS),
        help="Comma-separated recall cutoffs, e.g. 5,10,20,50",
    )
    parser.add_argument("--output", help="Optional metrics JSON output path")
    parser.add_argument("--retrieved-output", help="Optional fused JSONL output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [read_jsonl(path) for path in args.retrieval_files]
    rows = fuse_retrieval_rows(runs, top_k=args.top_k, rrf_k=args.rrf_k, weights=args.weights)
    metrics = compute_retrieval_metrics(rows, cutoffs=args.cutoffs)
    metrics.update(
        {
            "retriever": "hybrid_rrf",
            "retrieval_files": [str(path) for path in args.retrieval_files],
            "weights": args.weights or [1.0] * len(args.retrieval_files),
            "top_k": args.top_k,
            "rrf_k": args.rrf_k,
            "cutoffs": args.cutoffs,
        }
    )
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if args.retrieved_output:
        write_jsonl(args.retrieved_output, rows)


if __name__ == "__main__":
    main()
