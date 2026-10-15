#!/usr/bin/env python
"""Merge JSONL files into one output file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from law_grounding.data import read_jsonl, write_jsonl
from law_grounding.jsonl_shards import merge_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files, merged in the given order")
    parser.add_argument("--output", required=True, help="Merged output JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = merge_rows([read_jsonl(path) for path in args.inputs])
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
