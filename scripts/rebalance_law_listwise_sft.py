#!/usr/bin/env python
"""Rebalance listwise SFT records for high-recall selector training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from law_grounding.data import read_jsonl, write_jsonl
from law_grounding.law_listwise_sampling import rebalance_listwise_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-output")
    parser.add_argument("--max-negative-groups-per-unit", type=int, default=1)
    parser.add_argument("--empty-gold-unit-fraction", type=float, default=0.1)
    parser.add_argument("--late-positive-copies", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, stats = rebalance_listwise_records(
        read_jsonl(args.input),
        max_negative_groups_per_unit=args.max_negative_groups_per_unit,
        empty_gold_unit_fraction=args.empty_gold_unit_fraction,
        late_positive_copies=args.late_positive_copies,
        seed=args.seed,
    )
    write_jsonl(args.output, records)
    text = json.dumps(stats, ensure_ascii=False, indent=2)
    print(text)
    if args.stats_output:
        path = Path(args.stats_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
