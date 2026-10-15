#!/usr/bin/env python
"""Aggregate listwise law selector predictions into retrieval candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from law_grounding.data import read_jsonl, write_jsonl
from law_grounding.law_listwise_selector import aggregate_listwise_predictions, candidate_pool_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Listwise group prediction JSONL")
    parser.add_argument("--output", required=True, help="Retrieval candidate JSONL output")
    parser.add_argument("--metrics-output", help="Optional JSON path for candidate pool metrics")
    parser.add_argument("--no-include-maybe", action="store_true", help="Only aggregate keep, excluding maybe")
    parser.add_argument("--max-candidates", type=int, help="Optional cap for aggregated candidates per sample")
    parser.add_argument(
        "--always-keep-top-n",
        type=int,
        default=0,
        help="Always retain the first N original retrieval candidates before model selections",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = aggregate_listwise_predictions(
        read_jsonl(args.predictions),
        include_maybe=not args.no_include_maybe,
        max_candidates=args.max_candidates,
        always_keep_top_n=args.always_keep_top_n,
    )
    write_jsonl(args.output, rows)
    metrics = candidate_pool_metrics(rows)
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if args.metrics_output:
        path = Path(args.metrics_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
