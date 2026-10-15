"""Expand relevant_articles with judgment articles and high-confidence text law refs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from build_criminal_law_mapping import cn_number_to_int


JsonRecord = dict[str, Any]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "outputs" / "cdml_big_high_quality_20k"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "cdml_big_high_quality_20k_all_law_refs"
DEFAULT_LAW_MAP = SCRIPT_DIR / "outputs" / "criminal_law_20201226" / "criminal_law_map.json"
DEFAULT_FILES = (
    "train_big.len8000.def6.high_quality.jsonl",
    "valid_big.len8000.def6.high_quality.jsonl",
    "test_big.len8000.def6.high_quality.jsonl",
)

CN_NUMBER_CHARS = "零〇一二两三四五六七八九十百千"
CRIMINAL_LAW_MARKERS = ("中华人民共和国刑法", "我国刑法", "刑法", "本法")
SENTENCE_RE = re.compile(r"[^。；;！？!?\n]+[。；;！？!?]?")
LAW_REF_RE = re.compile(
    rf"第(?P<article>[{CN_NUMBER_CHARS}]+)条"
    rf"(?:之(?P<suffix>[{CN_NUMBER_CHARS}]+))?"
    rf"(?:第?(?P<paragraph>[{CN_NUMBER_CHARS}]+)款)?"
    rf"(?:第?（?(?P<item>[{CN_NUMBER_CHARS}]+)）?项)?"
)


def ordered_unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        ref = str(value).strip()
        if ref and ref not in seen:
            result.append(ref)
            seen.add(ref)
    return result


def extract_criminal_law_refs_from_text(text: str, law_map: dict[str, JsonRecord]) -> list[str]:
    refs: list[str] = []
    for sentence_match in SENTENCE_RE.finditer(str(text or "")):
        sentence = sentence_match.group(0)
        if not any(marker in sentence for marker in CRIMINAL_LAW_MARKERS):
            continue
        for match in LAW_REF_RE.finditer(sentence):
            ref = _law_ref_from_match(match, law_map)
            if ref:
                refs.append(ref)
    return ordered_unique(refs)


def expand_record_relevant_articles(record: JsonRecord, law_map: dict[str, JsonRecord]) -> tuple[JsonRecord, JsonRecord]:
    original_relevant = ordered_unique(record.get("relevant_articles") or [])
    judgment_refs = ordered_unique(_iter_judgment_articles(record))
    text_refs = extract_criminal_law_refs_from_text(
        f"{record.get('fact', '')}\n{record.get('court_view', '')}",
        law_map,
    )
    expanded_refs = ordered_unique([*original_relevant, *judgment_refs, *text_refs])
    expanded_record = dict(record)
    expanded_record["relevant_articles"] = expanded_refs
    original_set = set(original_relevant)
    judgment_set = set(judgment_refs)
    detail = {
        "old_count": len(original_relevant),
        "new_count": len(expanded_refs),
        "added_from_judgment": [ref for ref in judgment_refs if ref not in original_set],
        "added_from_text": [ref for ref in text_refs if ref not in original_set and ref not in judgment_set],
    }
    return expanded_record, detail


def process_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    law_map_path: str | Path,
    *,
    files: tuple[str, ...] = DEFAULT_FILES,
) -> JsonRecord:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    law_map = _load_law_map(law_map_path)

    split_summaries: list[JsonRecord] = []
    total_counter: Counter[str] = Counter()
    added_from_judgment: Counter[str] = Counter()
    added_from_text: Counter[str] = Counter()
    for filename in files:
        split_counter: Counter[str] = Counter()
        output_path = output_dir / filename
        with (input_dir / filename).open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                expanded, detail = expand_record_relevant_articles(record, law_map)
                split_counter["records"] += 1
                split_counter["old_refs"] += detail["old_count"]
                split_counter["new_refs"] += detail["new_count"]
                if detail["new_count"] != detail["old_count"]:
                    split_counter["changed_records"] += 1
                added_from_judgment.update(detail["added_from_judgment"])
                added_from_text.update(detail["added_from_text"])
                target.write(json.dumps(expanded, ensure_ascii=False, separators=(",", ":")) + "\n")
        total_counter.update(split_counter)
        split_summaries.append(
            {
                "file": filename,
                "output": str(output_path),
                "records": split_counter["records"],
                "changed_records": split_counter["changed_records"],
                "old_refs": split_counter["old_refs"],
                "new_refs": split_counter["new_refs"],
                "avg_old_refs": split_counter["old_refs"] / split_counter["records"] if split_counter["records"] else 0,
                "avg_new_refs": split_counter["new_refs"] / split_counter["records"] if split_counter["records"] else 0,
            }
        )

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "law_map": str(law_map_path),
        "files": list(files),
        "records": total_counter["records"],
        "changed_records": total_counter["changed_records"],
        "old_refs": total_counter["old_refs"],
        "new_refs": total_counter["new_refs"],
        "avg_old_refs": total_counter["old_refs"] / total_counter["records"] if total_counter["records"] else 0,
        "avg_new_refs": total_counter["new_refs"] / total_counter["records"] if total_counter["records"] else 0,
        "added_from_judgment_unique": len(added_from_judgment),
        "added_from_text_unique": len(added_from_text),
        "top_added_from_judgment": added_from_judgment.most_common(50),
        "top_added_from_text": added_from_text.most_common(50),
        "splits": split_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expand relevant_articles with all structured and high-confidence text law refs.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--law-map", type=Path, default=DEFAULT_LAW_MAP)
    parser.add_argument("--files", nargs="+", default=list(DEFAULT_FILES))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = process_dataset(args.input_dir, args.output_dir, args.law_map, files=tuple(args.files))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _iter_judgment_articles(record: JsonRecord) -> Iterable[str]:
    for outcome in record.get("outcomes") or []:
        if not isinstance(outcome, dict):
            continue
        for judgment in outcome.get("judgment") or []:
            if isinstance(judgment, dict):
                yield from (str(article) for article in judgment.get("article") or [])


def _law_ref_from_match(match: re.Match[str], law_map: dict[str, JsonRecord]) -> str | None:
    article_no = cn_number_to_int(match.group("article"))
    suffix_no = cn_number_to_int(match.group("suffix")) if match.group("suffix") else None
    paragraph_no = cn_number_to_int(match.group("paragraph")) if match.group("paragraph") else None
    has_item = bool(match.group("item"))
    article_id = f"{article_no}.{suffix_no}" if suffix_no else str(article_no)
    if paragraph_no is not None:
        ref = f"{article_id}-{paragraph_no}"
    elif has_item:
        ref = f"{article_id}-1"
    else:
        ref = article_id

    if ref in law_map:
        return ref
    if "-" in ref and ref.rsplit("-", 1)[0] in law_map:
        return ref.rsplit("-", 1)[0]
    if article_id in law_map:
        return article_id
    return None


def _load_law_map(path: str | Path) -> dict[str, JsonRecord]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
