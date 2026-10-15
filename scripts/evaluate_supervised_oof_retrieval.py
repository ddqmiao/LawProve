#!/usr/bin/env python
"""Evaluate out-of-fold supervised sparse law-profile retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from law_grounding.data import load_law_map, read_jsonl, write_jsonl
from law_grounding.law_retrieval import (
    DEFAULT_CUTOFFS,
    assign_group_folds,
    build_oof_fold_manifest,
    build_law_corpus,
    compute_retrieval_metrics,
    retrieve_supervised_oof_for_samples,
)


DEFAULT_LAW_MAP = "data/criminal_law_map.json"


def parse_cutoffs(value: str) -> list[int]:
    cutoffs = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not cutoffs:
        raise argparse.ArgumentTypeError("at least one cutoff is required")
    return cutoffs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--law-map", default=DEFAULT_LAW_MAP, help="criminal_law_map.json path")
    parser.add_argument("--samples", required=True, help="Prepared train samples JSONL path")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--group-field", default="case_id", help="Keep all rows with this field in one OOF fold")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic grouped-fold assignment seed")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--cutoffs",
        type=parse_cutoffs,
        default=list(DEFAULT_CUTOFFS),
        help="Comma-separated recall cutoffs, e.g. 5,10,20,50",
    )
    parser.add_argument(
        "--query-facts",
        choices=("all", "none"),
        default="none",
        help="Whether retrieval query includes all fact candidates or only defendant/court view",
    )
    parser.add_argument("--max-fact-chars", type=int, default=12000)
    parser.add_argument("--law-text-weight", type=float, default=0.15)
    parser.add_argument("--output", help="Optional metrics JSON output path")
    parser.add_argument("--retrieved-output", help="Optional retrieved law JSONL output path")
    parser.add_argument("--fold-manifest-output", help="Optional grouped OOF fold manifest JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    law_map = load_law_map(args.law_map)
    samples = read_jsonl(args.samples)
    documents = build_law_corpus(law_map)
    max_fact_chars = None if args.max_fact_chars <= 0 else args.max_fact_chars
    rows = retrieve_supervised_oof_for_samples(
        samples,
        documents,
        num_folds=args.num_folds,
        top_k=args.top_k,
        fact_mode=args.query_facts,
        max_fact_chars=max_fact_chars,
        law_text_weight=args.law_text_weight,
        group_field=args.group_field,
        seed=args.seed,
    )
    metrics = compute_retrieval_metrics(rows, cutoffs=args.cutoffs)
    metrics.update(
        {
            "retriever": "supervised_law_profile_oof",
            "samples": str(args.samples),
            "num_folds": args.num_folds,
            "group_field": args.group_field,
            "seed": args.seed,
            "top_k": args.top_k,
            "cutoffs": args.cutoffs,
            "query_facts": args.query_facts,
            "law_text_weight": args.law_text_weight,
            "num_law_documents": len(documents),
            "law_map": str(args.law_map),
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
    if args.fold_manifest_output:
        assignments = assign_group_folds(
            samples,
            num_folds=args.num_folds,
            group_field=args.group_field,
            seed=args.seed,
        )
        manifest = build_oof_fold_manifest(
            samples,
            assignments,
            num_folds=args.num_folds,
            group_field=args.group_field,
            seed=args.seed,
        )
        manifest_output = Path(args.fold_manifest_output)
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
