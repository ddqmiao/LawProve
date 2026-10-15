#!/usr/bin/env python
"""Aggregate joint ranker group logits into bounded retrieval candidate pools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from law_grounding.data import read_jsonl, write_jsonl
from law_grounding.joint_rank_grounding import aggregate_joint_rank_scores
from law_grounding.law_listwise_selector import candidate_pool_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs="+", required=True, help="One or more score JSONL shards")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output")
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--always-keep-top-n", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = [row for path in args.predictions for row in read_jsonl(path)]
    pools = aggregate_joint_rank_scores(
        predictions,
        max_candidates=args.max_candidates,
        always_keep_top_n=args.always_keep_top_n,
    )
    write_jsonl(args.output, pools)
    metrics = candidate_pool_metrics(pools)
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if args.metrics_output:
        path = Path(args.metrics_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
