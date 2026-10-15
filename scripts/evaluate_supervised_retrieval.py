#!/usr/bin/env python
"""Train and evaluate a supervised sparse law-profile retriever."""

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
    SupervisedLawProfileRetriever,
    build_law_corpus,
    compute_retrieval_metrics,
    retrieve_supervised_for_samples,
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
    parser.add_argument("--train-samples", required=True, help="Prepared train samples JSONL path")
    parser.add_argument("--eval-samples", required=True, help="Prepared eval samples JSONL path")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    law_map = load_law_map(args.law_map)
    train_samples = read_jsonl(args.train_samples)
    eval_samples = read_jsonl(args.eval_samples)
    documents = build_law_corpus(law_map)
    retriever = SupervisedLawProfileRetriever.fit(
        documents,
        train_samples,
        fact_mode=args.query_facts,
        law_text_weight=args.law_text_weight,
    )
    max_fact_chars = None if args.max_fact_chars <= 0 else args.max_fact_chars
    rows = retrieve_supervised_for_samples(
        eval_samples,
        retriever,
        top_k=args.top_k,
        fact_mode=args.query_facts,
        max_fact_chars=max_fact_chars,
    )
    metrics = compute_retrieval_metrics(rows, cutoffs=args.cutoffs)
    metrics.update(
        {
            "retriever": "supervised_law_profile",
            "train_samples": str(args.train_samples),
            "eval_samples": str(args.eval_samples),
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


if __name__ == "__main__":
    main()
