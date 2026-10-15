#!/usr/bin/env python
"""Resample prepared joint-ranking records without rebuilding prompt text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from law_grounding.data import read_jsonl, write_jsonl
from law_grounding.joint_rank_grounding import joint_rank_record_stats, select_joint_rank_training_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prepared train ranking records")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-output")
    parser.add_argument("--max-positive-groups-per-unit", type=int, default=1)
    parser.add_argument("--max-negative-groups-per-unit", type=int, default=1)
    parser.add_argument("--negative-unit-fraction", type=float, default=0.25)
    parser.add_argument("--empty-groups-per-unit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    records = read_jsonl(args.input)
    selected, selection_stats = select_joint_rank_training_records(
        records,
        max_positive_groups_per_unit=args.max_positive_groups_per_unit,
        max_negative_groups_per_unit=args.max_negative_groups_per_unit,
        negative_unit_fraction=args.negative_unit_fraction,
        empty_groups_per_unit=args.empty_groups_per_unit,
        seed=args.seed,
    )
    write_jsonl(output, selected)
    stats = {
        "configuration": vars(args),
        "input": joint_rank_record_stats(records),
        "selected": joint_rank_record_stats(selected),
        "selection": selection_stats,
    }
    text = json.dumps(stats, ensure_ascii=False, indent=2)
    stats_path = Path(args.stats_output) if args.stats_output else output.with_suffix(".stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
