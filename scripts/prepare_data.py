#!/usr/bin/env python
"""Prepare grounding samples and Qwen SFT JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from law_grounding.data import PROMPT_VARIANTS, prepare_splits


DEFAULT_DATA_DIR = "data/grounding"
DEFAULT_LAW_MAP = "data/criminal_law_map.json"
DEFAULT_OUTPUT_DIR = "outputs/prepared"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--law-map", default=DEFAULT_LAW_MAP)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-law-candidates", type=int, default=30)
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default="default")
    parser.add_argument(
        "--retrieval-candidates-dir",
        help="Optional directory containing per-split retrieval files such as train.top100.jsonl",
    )
    parser.add_argument("--retrieval-candidates-suffix", default=".top100.jsonl")
    parser.add_argument("--retrieval-top-k", type=int, default=50)
    parser.add_argument(
        "--no-filter-missing-law-candidates",
        action="store_true",
        help="Keep samples even when retrieved candidates miss a gold law; useful for end-to-end eval samples",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = prepare_splits(
        data_dir=args.data_dir,
        law_map_path=args.law_map,
        output_dir=args.output_dir,
        max_law_candidates=args.max_law_candidates,
        prompt_variant=args.prompt_variant,
        retrieval_candidates_dir=args.retrieval_candidates_dir,
        retrieval_candidates_suffix=args.retrieval_candidates_suffix,
        retrieval_top_k=args.retrieval_top_k,
        filter_missing_law_candidates=not args.no_filter_missing_law_candidates,
    )
    stats_path = Path(args.output_dir) / "stats.json"
    print(json.dumps({"output_dir": args.output_dir, "stats_path": str(stats_path), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
