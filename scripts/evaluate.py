#!/usr/bin/env python
"""Evaluate prediction JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from law_grounding.data import read_jsonl
from law_grounding.metrics import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    parser.add_argument("--output", help="Optional path to write metrics JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.predictions)
    metrics = evaluate_predictions(rows)
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
