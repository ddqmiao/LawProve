"""Split CMDL JSONL fact and court_view fields into numbered sentence units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


JsonRecord = dict[str, Any]

DEFAULT_FACT_PREFIX = "F"
DEFAULT_COURT_VIEW_PREFIX = "CV"
DEFAULT_FACT_FIELD = "fact"
DEFAULT_COURT_VIEW_FIELD = "court_view"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "CMDL_small" / "train_small.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "train_small_split.jsonl"

SENTENCE_TERMINATORS = {"。", "；", ";", "！", "!", "？", "?"}
LINE_BREAKS = {"\n", "\r"}


def split_sentences(text: str) -> list[str]:
    """Split Chinese legal text into sentence-like units while preserving delimiters."""

    if not isinstance(text, str):
        return []

    sentences: list[str] = []
    buffer: list[str] = []
    for char in text:
        if char in SENTENCE_TERMINATORS:
            buffer.append(char)
            _flush(buffer, sentences)
        elif char in LINE_BREAKS:
            _flush(buffer, sentences)
        else:
            buffer.append(char)
    _flush(buffer, sentences)
    return sentences


def label_sentences(text: str, prefix: str) -> list[JsonRecord]:
    """Return sentence units as `{id, text}` records with a contiguous prefix."""

    return [{"id": f"{prefix}{index}", "text": sentence} for index, sentence in enumerate(split_sentences(text))]


def transform_record(
    record: JsonRecord,
    *,
    fact_field: str = DEFAULT_FACT_FIELD,
    court_view_field: str = DEFAULT_COURT_VIEW_FIELD,
    fact_prefix: str = DEFAULT_FACT_PREFIX,
    court_view_prefix: str = DEFAULT_COURT_VIEW_PREFIX,
) -> JsonRecord:
    """Return a record copy with only fact and court_view fields sentence-labeled."""

    transformed = dict(record)
    transformed[fact_field] = label_sentences(str(record.get(fact_field, "")), fact_prefix)
    transformed[court_view_field] = label_sentences(str(record.get(court_view_field, "")), court_view_prefix)
    return transformed


def process_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    fact_field: str = DEFAULT_FACT_FIELD,
    court_view_field: str = DEFAULT_COURT_VIEW_FIELD,
    fact_prefix: str = DEFAULT_FACT_PREFIX,
    court_view_prefix: str = DEFAULT_COURT_VIEW_PREFIX,
) -> JsonRecord:
    """Read JSONL, transform each record, and write JSONL to output_path."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    read_count = 0
    written_count = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object on line {line_number}, got {type(record).__name__}")

            read_count += 1
            transformed = transform_record(
                record,
                fact_field=fact_field,
                court_view_field=court_view_field,
                fact_prefix=fact_prefix,
                court_view_prefix=court_view_prefix,
            )
            target.write(json.dumps(transformed, ensure_ascii=False, separators=(",", ":")) + "\n")
            written_count += 1

    return {"read": read_count, "written": written_count, "output": str(output_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split CMDL JSONL fact and court_view fields into numbered sentence units."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Input JSONL path. Default: {DEFAULT_INPUT}")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--fact-field", default=DEFAULT_FACT_FIELD, help="Fact field name to split.")
    parser.add_argument("--court-view-field", default=DEFAULT_COURT_VIEW_FIELD, help="Court view field name to split.")
    parser.add_argument("--fact-prefix", default=DEFAULT_FACT_PREFIX, help="Label prefix for fact units.")
    parser.add_argument("--court-view-prefix", default=DEFAULT_COURT_VIEW_PREFIX, help="Label prefix for court view units.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = process_jsonl(
        args.input,
        args.output,
        fact_field=args.fact_field,
        court_view_field=args.court_view_field,
        fact_prefix=args.fact_prefix,
        court_view_prefix=args.court_view_prefix,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _flush(buffer: list[str], sentences: list[str]) -> None:
    sentence = "".join(buffer).strip()
    if sentence and not all(char in SENTENCE_TERMINATORS for char in sentence):
        sentences.append(sentence)
    buffer.clear()


if __name__ == "__main__":
    main()
