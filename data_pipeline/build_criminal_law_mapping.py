"""Build Criminal Law article lookup maps from the 2020-12-26 DOCX."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


JsonRecord = dict[str, Any]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DOCX = PROJECT_DIR / "data" / "法律法规" / "中华人民共和国刑法_20201226.docx"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "criminal_law_20201226"
DEFAULT_DATASET_DIR = SCRIPT_DIR / "outputs" / "cdml_big_high_quality_20k"
DEFAULT_DATASET_FILES = (
    "train_big.len8000.def6.high_quality.jsonl",
    "valid_big.len8000.def6.high_quality.jsonl",
    "test_big.len8000.def6.high_quality.jsonl",
)

CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_NUMBER_CHARS = "零〇一二两三四五六七八九十百千"
ARTICLE_RE = re.compile(
    rf"^第(?P<number>[{CN_NUMBER_CHARS}]+)条(?P<suffix>之[{CN_NUMBER_CHARS}]+)?\s*(?P<body>.*)$"
)
HEADING_RE = re.compile(rf"^第(?P<number>[{CN_NUMBER_CHARS}]+)(?P<level>[编章节])\s*(?P<title>.*)$")
ITEM_RE = re.compile(rf"^(?P<label>（[{CN_NUMBER_CHARS}]+）)")


def cn_number_to_int(value: str) -> int:
    total = 0
    number = 0
    for char in value:
        if char in CN_DIGITS:
            number = CN_DIGITS[char]
        elif char == "十":
            total += (number or 1) * 10
            number = 0
        elif char == "百":
            total += (number or 1) * 100
            number = 0
        elif char == "千":
            total += (number or 1) * 1000
            number = 0
        else:
            raise ValueError(f"Unsupported Chinese number character: {char}")
    return total + number


def parse_law_paragraphs(paragraphs: Iterable[str], *, source_name: str = "") -> dict[str, JsonRecord]:
    rows = [(index, _normalize_text(text)) for index, text in enumerate(paragraphs, start=1)]
    rows = [(index, text) for index, text in rows if text]
    if not rows:
        return {}

    start_position = _body_start_position(rows)
    articles: dict[str, JsonRecord] = {}
    context = {"part": "", "chapter": "", "section": ""}
    current_article: JsonRecord | None = None

    for source_index, text in rows[start_position:]:
        if _is_appendix_heading(text):
            context = {"part": "附则", "chapter": "", "section": ""}
            current_article = None
            continue

        article_match = ARTICLE_RE.match(text)
        if article_match:
            current_article = _new_article(article_match, text, source_index, context, source_name)
            articles[current_article["article_id"]] = current_article
            continue

        heading_match = HEADING_RE.match(text)
        if heading_match:
            _update_context(context, heading_match, text)
            current_article = None
            continue

        if current_article is None:
            continue

        item_match = ITEM_RE.match(text)
        if item_match and current_article["paragraphs"]:
            _append_item(current_article, item_match, text, source_index)
        else:
            _add_article_paragraph(current_article, text, source_index)

    for article in articles.values():
        article["paragraph_count"] = len(article["paragraphs"])
        article["paragraph_ids"] = [paragraph["paragraph_id"] for paragraph in article["paragraphs"]]
        article["full_text"] = "\n".join(paragraph["text"] for paragraph in article["paragraphs"])
    return articles


def build_lookup(articles: dict[str, JsonRecord]) -> dict[str, JsonRecord]:
    lookup: dict[str, JsonRecord] = {}
    for article_id, article in articles.items():
        base_entry = _article_base_entry(article)
        lookup[article_id] = {
            **base_entry,
            "ref": article_id,
            "scope": "article",
            "text": article["full_text"],
            "paragraph_count": article["paragraph_count"],
            "paragraph_ids": article["paragraph_ids"],
        }
        for paragraph in article["paragraphs"]:
            lookup[paragraph["paragraph_id"]] = {
                **base_entry,
                "ref": paragraph["paragraph_id"],
                "scope": "paragraph",
                "paragraph_id": paragraph["paragraph_id"],
                "paragraph_no": paragraph["paragraph_no"],
                "text": paragraph["text"],
                "article_text": article["full_text"],
                "item_count": len(paragraph.get("items", [])),
            }
    return lookup


def build_flat_map(articles: dict[str, JsonRecord]) -> dict[str, JsonRecord]:
    flat_map: dict[str, JsonRecord] = {}
    for article_id, article in articles.items():
        flat_map[article_id] = {
            "type": "article",
            "article_id": article_id,
            "article_label": article["article_label"],
            "text": article["full_text"],
            "source": article["source"],
        }
        for paragraph in article["paragraphs"]:
            flat_map[paragraph["paragraph_id"]] = {
                "type": "paragraph",
                "article_id": article_id,
                "article_label": article["article_label"],
                "paragraph_no": paragraph["paragraph_no"],
                "text": paragraph["text"],
                "source": article["source"],
            }
    return flat_map


def resolve_reference(ref: str, articles: dict[str, JsonRecord], lookup: dict[str, JsonRecord]) -> JsonRecord:
    ref = str(ref).strip()
    if ref in lookup:
        return {"query": ref, "matched": True, **lookup[ref]}

    article_id = ref.rsplit("-", 1)[0] if "-" in ref else ref
    article = articles.get(article_id)
    if article:
        return {
            "query": ref,
            "matched": False,
            "reason": "paragraph_not_found" if "-" in ref else "reference_not_found",
            "fallback_article_id": article_id,
            "fallback_text": article["full_text"],
            **_article_base_entry(article),
        }

    return {"query": ref, "matched": False, "reason": "article_not_found"}


def load_docx_paragraphs(path: str | Path) -> list[str]:
    from docx import Document

    return [paragraph.text for paragraph in Document(path).paragraphs]


def collect_article_refs(jsonl_paths: Iterable[str | Path]) -> Counter[str]:
    refs: Counter[str] = Counter()
    for path in jsonl_paths:
        with Path(path).open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                refs.update(str(article) for article in record.get("relevant_articles") or [])
                for outcome in record.get("outcomes") or []:
                    for judgment in outcome.get("judgment") or []:
                        refs.update(str(article) for article in judgment.get("article") or [])
    return refs


def build_coverage_report(articles: dict[str, JsonRecord], lookup: dict[str, JsonRecord], refs: Counter[str]) -> JsonRecord:
    exact_total = 0
    fallback_total = 0
    missing_total = 0
    exact_unique = 0
    fallback_unique = 0
    missing_unique = 0
    unresolved: list[JsonRecord] = []

    for ref, count in sorted(refs.items(), key=lambda item: (-item[1], item[0])):
        resolved = resolve_reference(ref, articles, lookup)
        if resolved["matched"]:
            exact_total += count
            exact_unique += 1
        elif resolved["reason"] == "paragraph_not_found":
            fallback_total += count
            fallback_unique += 1
            unresolved.append(
                {
                    "ref": ref,
                    "count": count,
                    "reason": resolved["reason"],
                    "fallback_article_id": resolved["fallback_article_id"],
                }
            )
        else:
            missing_total += count
            missing_unique += 1
            unresolved.append({"ref": ref, "count": count, "reason": resolved["reason"]})

    total_refs = sum(refs.values())
    unique_refs = len(refs)
    return {
        "total_refs": total_refs,
        "unique_refs": unique_refs,
        "exact_matched_total": exact_total,
        "exact_matched_unique": exact_unique,
        "article_fallback_total": fallback_total,
        "article_fallback_unique": fallback_unique,
        "missing_article_total": missing_total,
        "missing_article_unique": missing_unique,
        "exact_match_rate": exact_total / total_refs if total_refs else 0,
        "article_level_coverage_rate": (exact_total + fallback_total) / total_refs if total_refs else 0,
        "unresolved_refs": unresolved,
    }


def write_outputs(
    articles: dict[str, JsonRecord],
    lookup: dict[str, JsonRecord],
    output_dir: str | Path,
    *,
    coverage: JsonRecord | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flat_map = build_flat_map(articles)
    _write_json(output_dir / "criminal_law_map.json", flat_map)
    _write_json(output_dir / "build_summary.json", build_summary(articles, flat_map, coverage))
    if coverage is not None:
        _write_json(output_dir / "coverage_report.json", coverage)
        (output_dir / "coverage_report.md").write_text(render_coverage_markdown(coverage), encoding="utf-8")


def build_summary(
    articles: dict[str, JsonRecord],
    flat_map: dict[str, JsonRecord],
    coverage: JsonRecord | None,
) -> JsonRecord:
    return {
        "article_count": len(articles),
        "map_entry_count": len(flat_map),
        "article_entry_count": sum(1 for entry in flat_map.values() if entry.get("type") == "article"),
        "paragraph_entry_count": sum(1 for entry in flat_map.values() if entry.get("type") == "paragraph"),
        "coverage": coverage,
        "outputs": {
            "law_map": "criminal_law_map.json",
            "summary": "build_summary.json",
        },
    }


def render_coverage_markdown(coverage: JsonRecord) -> str:
    lines = [
        "# Criminal Law Mapping Coverage",
        "",
        f"- total_refs: {coverage['total_refs']}",
        f"- unique_refs: {coverage['unique_refs']}",
        f"- exact_matched_total: {coverage['exact_matched_total']} ({coverage['exact_match_rate']:.4%})",
        f"- article_fallback_total: {coverage['article_fallback_total']}",
        f"- missing_article_total: {coverage['missing_article_total']}",
        f"- article_level_coverage_rate: {coverage['article_level_coverage_rate']:.4%}",
        "",
        "## Top Unresolved Refs",
        "",
    ]
    unresolved = coverage.get("unresolved_refs", [])[:80]
    if not unresolved:
        lines.append("- none")
    else:
        for item in unresolved:
            suffix = f", fallback_article_id={item['fallback_article_id']}" if "fallback_article_id" in item else ""
            lines.append(f"- {item['ref']}: {item['count']} ({item['reason']}{suffix})")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Criminal Law article lookup maps from DOCX.")
    parser.add_argument("--docx", type=Path, default=DEFAULT_DOCX, help=f"Input DOCX. Default: {DEFAULT_DOCX}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument(
        "--dataset-files",
        nargs="*",
        type=Path,
        default=[DEFAULT_DATASET_DIR / filename for filename in DEFAULT_DATASET_FILES],
        help="Optional JSONL files used only for coverage reporting.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    articles = parse_law_paragraphs(load_docx_paragraphs(args.docx), source_name=args.docx.name)
    lookup = build_lookup(articles)
    flat_map = build_flat_map(articles)
    dataset_files = [path for path in args.dataset_files if path.exists()]
    coverage = build_coverage_report(articles, lookup, collect_article_refs(dataset_files)) if dataset_files else None
    write_outputs(articles, lookup, args.output_dir, coverage=coverage)
    print(
        json.dumps(
            {
                "article_count": len(articles),
                "lookup_count": len(lookup),
                "map_entry_count": len(flat_map),
                "output_dir": str(args.output_dir),
                "coverage": coverage,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def _body_start_position(rows: list[tuple[int, str]]) -> int:
    first_article_position = next((position for position, (_, text) in enumerate(rows) if ARTICLE_RE.match(text)), 0)
    for position in range(first_article_position - 1, -1, -1):
        heading_match = HEADING_RE.match(rows[position][1])
        if heading_match and heading_match.group("level") == "编":
            return position
    return first_article_position


def _is_appendix_heading(text: str) -> bool:
    return text.replace(" ", "") == "附则"


def _new_article(
    article_match: re.Match[str],
    text: str,
    source_index: int,
    context: dict[str, str],
    source_name: str,
) -> JsonRecord:
    article_no = cn_number_to_int(article_match.group("number"))
    suffix_text = article_match.group("suffix")
    suffix_no = cn_number_to_int(suffix_text[1:]) if suffix_text else None
    article_id = f"{article_no}.{suffix_no}" if suffix_no else str(article_no)
    article = {
        "article_id": article_id,
        "article_no": article_no,
        "article_suffix_no": suffix_no,
        "article_label": f"第{article_match.group('number')}条{suffix_text or ''}",
        "source": source_name,
        "part": context["part"],
        "chapter": context["chapter"],
        "section": context["section"],
        "start_source_paragraph_index": source_index,
        "paragraphs": [],
    }
    _add_article_paragraph(article, text, source_index)
    return article


def _add_article_paragraph(article: JsonRecord, text: str, source_index: int) -> None:
    paragraph_no = len(article["paragraphs"]) + 1
    paragraph_id = f"{article['article_id']}-{paragraph_no}"
    article["paragraphs"].append(
        {
            "paragraph_id": paragraph_id,
            "paragraph_no": paragraph_no,
            "text": text,
            "items": [],
            "start_source_paragraph_index": source_index,
            "end_source_paragraph_index": source_index,
        }
    )


def _append_item(article: JsonRecord, item_match: re.Match[str], text: str, source_index: int) -> None:
    paragraph = article["paragraphs"][-1]
    paragraph["items"].append(
        {
            "item_label": item_match.group("label"),
            "text": text,
            "source_paragraph_index": source_index,
        }
    )
    paragraph["text"] = f"{paragraph['text']}\n{text}"
    paragraph["end_source_paragraph_index"] = source_index


def _update_context(context: dict[str, str], heading_match: re.Match[str], text: str) -> None:
    level = heading_match.group("level")
    if level == "编":
        context["part"] = text
        context["chapter"] = ""
        context["section"] = ""
    elif level == "章":
        context["chapter"] = text
        context["section"] = ""
    elif level == "节":
        context["section"] = text


def _article_base_entry(article: JsonRecord) -> JsonRecord:
    return {
        "article_id": article["article_id"],
        "article_no": article["article_no"],
        "article_suffix_no": article["article_suffix_no"],
        "article_label": article["article_label"],
        "source": article["source"],
        "part": article["part"],
        "chapter": article["chapter"],
        "section": article["section"],
    }


def _write_json(path: Path, payload: JsonRecord) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
