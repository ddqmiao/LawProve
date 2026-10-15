#!/usr/bin/env python
"""Prepare grouped listwise law selector SFT data from prepared samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from law_grounding.data import load_law_map, read_jsonl, split_name_from_path, write_jsonl
from law_grounding.law_listwise_selector import build_listwise_selector_records


DEFAULT_LAW_MAP = "data/criminal_law_map.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", required=True, help="Directory containing prepared samples/{split}.jsonl")
    parser.add_argument("--law-map", default=DEFAULT_LAW_MAP)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=10)
    parser.add_argument("--max-fact-chars", type=int, default=3000)
    return parser.parse_args()


def _record_stats(records: list[dict[str, object]]) -> dict[str, object]:
    keep = maybe = reject = 0
    group_sizes: list[int] = []
    for record in records:
        answer = json.loads(record["messages"][-1]["content"])
        keep += len(answer["keep"])
        maybe += len(answer["maybe"])
        reject += len(answer["reject"])
        group_sizes.append(len(record["meta"]["candidate_law_ids"]))
    avg_group_size = sum(group_sizes) / len(group_sizes) if group_sizes else 0.0
    return {
        "records": len(records),
        "avg_group_size": round(avg_group_size, 4),
        "keep_labels": keep,
        "maybe_labels": maybe,
        "reject_labels": reject,
    }


def main() -> None:
    args = parse_args()
    samples_dir = Path(args.samples_dir)
    output_dir = Path(args.output_dir)
    law_map = load_law_map(args.law_map)
    stats: dict[str, object] = {"splits": {}}

    for path in sorted(samples_dir.glob("*.jsonl")):
        split = split_name_from_path(path)
        samples = read_jsonl(path)
        records = build_listwise_selector_records(
            samples,
            law_map,
            top_k=args.top_k,
            group_size=args.group_size,
            max_fact_chars=args.max_fact_chars,
        )
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", records)
        stats["splits"][split] = {"samples": len(samples), **_record_stats(records)}

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "stats_path": str(stats_path), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
