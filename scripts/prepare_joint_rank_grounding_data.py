#!/usr/bin/env python
"""Prepare first-token listwise ranking records for joint rank-grounding training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from law_grounding.data import load_law_map, read_jsonl, split_name_from_path, write_jsonl
from law_grounding.joint_rank_grounding import (
    build_all_joint_rank_records,
    joint_rank_record_stats,
    select_joint_rank_training_records,
)


DEFAULT_LAW_MAP = "data/criminal_law_map.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", required=True, help="Directory containing prepared {split}.jsonl samples")
    parser.add_argument("--law-map", default=DEFAULT_LAW_MAP)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=10)
    parser.add_argument("--max-fact-chars", type=int, default=3000)
    parser.add_argument("--max-positive-groups-per-unit", type=int, default=1)
    parser.add_argument("--max-negative-groups-per-unit", type=int, default=1)
    parser.add_argument("--negative-unit-fraction", type=float, default=1.0)
    parser.add_argument("--empty-groups-per-unit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1 or args.group_size < 1:
        raise ValueError("top-k and group-size must be positive")
    samples_dir = Path(args.samples_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    law_map = load_law_map(args.law_map)
    stats: dict[str, object] = {
        "configuration": vars(args),
        "splits": {},
    }
    for path in sorted(samples_dir.glob("*.jsonl")):
        split = split_name_from_path(path)
        samples = read_jsonl(path)
        all_records = build_all_joint_rank_records(
            samples,
            law_map,
            top_k=args.top_k,
            group_size=args.group_size,
            max_fact_chars=args.max_fact_chars,
            seed=args.seed,
        )
        write_jsonl(output_dir / "all" / f"{split}.jsonl", all_records)
        split_stats: dict[str, object] = {
            "samples": len(samples),
            "all": joint_rank_record_stats(all_records),
        }
        if split == "train":
            train_records, selection_stats = select_joint_rank_training_records(
                all_records,
                max_positive_groups_per_unit=args.max_positive_groups_per_unit,
                max_negative_groups_per_unit=args.max_negative_groups_per_unit,
                negative_unit_fraction=args.negative_unit_fraction,
                empty_groups_per_unit=args.empty_groups_per_unit,
                seed=args.seed,
            )
            write_jsonl(output_dir / "train.jsonl", train_records)
            split_stats["selected"] = joint_rank_record_stats(train_records)
            split_stats["selection"] = selection_stats
        stats["splits"][split] = split_stats

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "stats_path": str(stats_path), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
