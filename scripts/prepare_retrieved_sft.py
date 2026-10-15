#!/usr/bin/env python
"""Prepare SFT data by applying retrieved law candidates to prepared samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from law_grounding.data import PROMPT_VARIANTS, prepare_splits_from_prepared_samples


DEFAULT_LAW_MAP = "data/criminal_law_map.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", required=True, help="Directory containing prepared samples/{split}.jsonl")
    parser.add_argument("--law-map", default=DEFAULT_LAW_MAP)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default="default")
    parser.add_argument("--retrieval-candidates-dir", required=True)
    parser.add_argument("--retrieval-candidates-suffix", default=".top100.jsonl")
    parser.add_argument("--retrieval-top-k", type=int, default=50)
    parser.add_argument(
        "--filter-missing-law-candidate-splits",
        help="Comma-separated split names to filter even when --no-filter-missing-law-candidates is set",
    )
    parser.add_argument(
        "--no-filter-missing-law-candidates",
        action="store_true",
        help="Keep samples even when retrieved candidates miss a gold law",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filter_splits = None
    if args.filter_missing_law_candidate_splits:
        filter_splits = {item.strip() for item in args.filter_missing_law_candidate_splits.split(",") if item.strip()}
    stats = prepare_splits_from_prepared_samples(
        samples_dir=args.samples_dir,
        law_map_path=args.law_map,
        output_dir=args.output_dir,
        retrieval_candidates_dir=args.retrieval_candidates_dir,
        retrieval_candidates_suffix=args.retrieval_candidates_suffix,
        retrieval_top_k=args.retrieval_top_k,
        prompt_variant=args.prompt_variant,
        filter_missing_law_candidates=not args.no_filter_missing_law_candidates,
        filter_missing_law_candidate_splits=filter_splits,
    )
    stats_path = Path(args.output_dir) / "stats.json"
    print(json.dumps({"output_dir": args.output_dir, "stats_path": str(stats_path), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
