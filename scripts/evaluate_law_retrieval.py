#!/usr/bin/env python
"""Evaluate BM25 criminal-law retrieval recall on prepared grounding samples."""

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
    BM25LawRetriever,
    build_law_corpus,
    compute_retrieval_metrics,
    retrieve_for_samples,
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
    parser.add_argument("--samples", required=True, help="Prepared samples JSONL path")
    parser.add_argument("--top-k", type=int, default=50, help="Number of laws to retrieve per sample")
    parser.add_argument(
        "--cutoffs",
        type=parse_cutoffs,
        default=list(DEFAULT_CUTOFFS),
        help="Comma-separated recall cutoffs, e.g. 5,10,20,50",
    )
    parser.add_argument(
        "--max-fact-chars",
        type=int,
        default=12000,
        help="Maximum fact text chars included in retrieval query; <=0 means all facts",
    )
    parser.add_argument(
        "--query-facts",
        choices=("all", "none"),
        default="all",
        help="Whether retrieval query includes all fact candidates or only defendant/court view",
    )
    parser.add_argument("--output", help="Optional metrics JSON output path")
    parser.add_argument("--retrieved-output", help="Optional retrieved law JSONL output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    law_map = load_law_map(args.law_map)
    samples = read_jsonl(args.samples)
    corpus = build_law_corpus(law_map)
    retriever = BM25LawRetriever(corpus)
    max_fact_chars = None if args.max_fact_chars <= 0 else args.max_fact_chars
    rows = retrieve_for_samples(
        samples,
        retriever,
        top_k=args.top_k,
        fact_mode=args.query_facts,
        max_fact_chars=max_fact_chars,
    )
    metrics = compute_retrieval_metrics(rows, cutoffs=args.cutoffs)
    metrics.update(
        {
            "retriever": "bm25",
            "top_k": args.top_k,
            "cutoffs": args.cutoffs,
            "query_facts": args.query_facts,
            "num_law_documents": len(corpus),
            "samples": str(args.samples),
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
