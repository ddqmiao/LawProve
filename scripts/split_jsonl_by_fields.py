#!/usr/bin/env python
"""Split JSONL rows into group-preserving shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from law_grounding.data import read_jsonl, write_jsonl
from law_grounding.jsonl_shards import shard_rows_by_fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output-dir", required=True, help="Directory for shard_*.jsonl files")
    parser.add_argument("--fields", required=True, help="Comma-separated nested fields used as the group key")
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--prefix", default="shard")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    shards = shard_rows_by_fields(read_jsonl(args.input), fields, num_shards=args.num_shards)
    output_dir = Path(args.output_dir)
    for index, rows in enumerate(shards):
        write_jsonl(output_dir / f"{args.prefix}_{index}.jsonl", rows)
    print(f"wrote {len(shards)} shards to {output_dir}")


if __name__ == "__main__":
    main()
