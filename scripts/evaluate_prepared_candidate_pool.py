#!/usr/bin/env python
"""Evaluate the existing candidate order in prepared grounding samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from law_grounding.data import read_jsonl
from law_grounding.law_listwise_selector import candidate_pool_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    rows = [
        {
            "gold_laws": [str(item) for item in sample.get("gold_laws", [])],
            "retrieved_law_ids": [
                str(candidate["id"])
                for candidate in (sample.get("law_candidates") or [])[: args.top_k]
            ],
        }
        for sample in read_jsonl(args.samples)
    ]
    metrics = {"top_k": args.top_k, **candidate_pool_metrics(rows)}
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
