"""Remove records whose court_view likely contains fact/evidence sections."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


JsonRecord = dict[str, Any]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "outputs" / "cdml_big_len8000_def6_minor_trim_clean"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "cdml_big_len8000_def6_minor_trim_clean_no_cv_contamination"
DEFAULT_FILES = (
    "train_big.len8000.def6.jsonl",
    "valid_big.len8000.def6.jsonl",
    "test_big.len8000.def6.jsonl",
)

EVIDENCE_TERMS = (
    "证人证言",
    "被害人陈述",
    "被害单位",
    "被告人供述",
    "供述和辩解",
    "辨认笔录",
    "鉴定意见",
    "鉴定结论",
    "书证",
    "物证",
    "视听资料",
    "电子数据",
    "现场勘验",
    "现场勘查",
    "检查笔录",
    "搜查笔录",
    "扣押清单",
    "抓获经过",
    "到案经过",
    "破案经过",
    "受案登记表",
    "立案决定书",
    "户籍证明",
    "人口信息",
    "价格认定",
    "价格鉴定",
    "谅解书",
    "收条",
    "病历",
    "诊断证明",
    "证据证实",
    "足以认定",
    "予以认定",
)

FACT_SECTION_PATTERNS = (
    re.compile(r"经审理查明"),
    re.compile(r"上述事实.*?(证据|证实|认定)"),
    re.compile(r"以上事实.*?(证据|证实|认定)"),
    re.compile(r"经庭审.*?(举证|质证|认证)"),
)

NON_COURT_START_PATTERNS = (
    re.compile(r"^公诉机关(认为|指控)"),
    re.compile(r"^被告人.*?(辩称|辩解|无异议)"),
    re.compile(r"^辩护人.*?(认为|提出)"),
    re.compile(r"^经审理查明"),
)


def contamination_reasons(record: JsonRecord) -> list[str]:
    fact = record.get("fact")
    court_view = record.get("court_view")
    if not isinstance(fact, str) or not isinstance(court_view, str):
        return []

    fact_len = len(fact)
    court_view_len = len(court_view)
    if court_view_len <= fact_len:
        return []

    evidence_hits = count_evidence_hits(court_view)
    distinct_evidence_terms = count_distinct_evidence_terms(court_view)
    fact_section = has_fact_section_pattern(court_view)
    non_court_start = has_non_court_start(court_view)
    if distinct_evidence_terms >= 3 or evidence_hits >= 5 or fact_section or non_court_start:
        return ["court_view_gt_fact_with_contamination_signal"]
    return []


def is_high_confidence_contamination(record: JsonRecord) -> bool:
    return bool(contamination_reasons(record))


def count_evidence_hits(text: str) -> int:
    return sum(text.count(term) for term in EVIDENCE_TERMS)


def count_distinct_evidence_terms(text: str) -> int:
    return sum(1 for term in EVIDENCE_TERMS if term in text)


def has_fact_section_pattern(text: str) -> bool:
    return any(pattern.search(text) for pattern in FACT_SECTION_PATTERNS)


def has_non_court_start(text: str) -> bool:
    stripped = text.strip()
    return any(pattern.search(stripped) for pattern in NON_COURT_START_PATTERNS)


def process_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    rejected_path: str | Path,
    *,
    split_name: str,
) -> JsonRecord:
    input_path = Path(input_path)
    output_path = Path(output_path)
    rejected_path = Path(rejected_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)

    read_count = 0
    kept_count = 0
    rejected_count = 0
    reject_reasons: Counter[str] = Counter()
    evidence_terms_in_rejected: Counter[str] = Counter()

    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as kept_target, rejected_path.open("w", encoding="utf-8") as rejected_target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            read_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                rejected_count += 1
                reject_reasons["json_decode_error"] += 1
                rejected_target.write(
                    json.dumps(
                        {
                            "source_split": split_name,
                            "source_line_no": line_number,
                            "reject_reasons": ["json_decode_error"],
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                continue

            if not isinstance(record, dict):
                rejected_count += 1
                reject_reasons["non_object"] += 1
                rejected_target.write(
                    json.dumps(
                        {
                            "source_split": split_name,
                            "source_line_no": line_number,
                            "reject_reasons": ["non_object"],
                            "type": type(record).__name__,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                continue

            reasons = contamination_reasons(record)
            if reasons:
                rejected_count += 1
                reject_reasons.update(reasons)
                court_view = str(record.get("court_view") or "")
                for term in EVIDENCE_TERMS:
                    if term in court_view:
                        evidence_terms_in_rejected[term] += 1
                rejected_target.write(
                    json.dumps(
                        {
                            "source_split": split_name,
                            "source_line_no": line_number,
                            "reject_reasons": reasons,
                            "fact_len": len(str(record.get("fact") or "")),
                            "court_view_len": len(court_view),
                            "evidence_hits": count_evidence_hits(court_view),
                            "distinct_evidence_terms": count_distinct_evidence_terms(court_view),
                            "has_fact_section_pattern": has_fact_section_pattern(court_view),
                            "has_non_court_start": has_non_court_start(court_view),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                continue

            kept_target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            kept_count += 1

    return {
        "input": str(input_path),
        "output": str(output_path),
        "rejected_output": str(rejected_path),
        "split_name": split_name,
        "read": read_count,
        "kept": kept_count,
        "rejected": rejected_count,
        "reject_reasons": dict(reject_reasons),
        "evidence_terms_in_rejected": dict(evidence_terms_in_rejected),
    }


def process_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    files: tuple[str, ...] = DEFAULT_FILES,
) -> JsonRecord:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for filename in files:
        input_path = input_dir / filename
        output_path = output_dir / filename
        rejected_path = output_dir / filename.replace(".jsonl", ".rejected.jsonl")
        summaries.append(process_jsonl(input_path, output_path, rejected_path, split_name=filename.removesuffix(".jsonl")))

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "splits": summaries,
        "aggregate": aggregate_summaries(summaries),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "filter_report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def aggregate_summaries(summaries: list[JsonRecord]) -> JsonRecord:
    aggregate: JsonRecord = {
        "read": 0,
        "kept": 0,
        "rejected": 0,
        "reject_reasons": Counter(),
        "evidence_terms_in_rejected": Counter(),
    }
    for summary in summaries:
        aggregate["read"] += int(summary["read"])
        aggregate["kept"] += int(summary["kept"])
        aggregate["rejected"] += int(summary["rejected"])
        aggregate["reject_reasons"].update(summary["reject_reasons"])
        aggregate["evidence_terms_in_rejected"].update(summary["evidence_terms_in_rejected"])
    aggregate["reject_reasons"] = dict(aggregate["reject_reasons"])
    aggregate["evidence_terms_in_rejected"] = dict(aggregate["evidence_terms_in_rejected"])
    return aggregate


def render_report(summary: JsonRecord) -> str:
    aggregate = summary["aggregate"]
    lines = [
        "# Court View Contamination Clean Report",
        "",
        f"- input_dir: `{summary['input_dir']}`",
        f"- output_dir: `{summary['output_dir']}`",
        "",
        "## Aggregate",
        "",
        _metric_line("read", aggregate["read"], aggregate["read"]),
        _metric_line("kept", aggregate["kept"], aggregate["read"]),
        _metric_line("rejected", aggregate["rejected"], aggregate["read"]),
        "",
        "## Reject Reasons",
        "",
        *_counter_lines(aggregate["reject_reasons"], aggregate["read"]),
        "",
        "## Evidence Terms In Rejected",
        "",
        *_counter_lines(aggregate["evidence_terms_in_rejected"], max(aggregate["rejected"], 1)),
        "",
        "## Splits",
        "",
    ]
    for split in summary["splits"]:
        lines.extend(
            [
                f"### {split['split_name']}",
                "",
                _metric_line("read", split["read"], split["read"]),
                _metric_line("kept", split["kept"], split["read"]),
                _metric_line("rejected", split["rejected"], split["read"]),
                "",
            ]
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean high-confidence court_view fact/evidence contamination.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help=f"Input filtered directory. Default: {DEFAULT_INPUT_DIR}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--files", nargs="+", default=list(DEFAULT_FILES), help="JSONL filenames to process.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = process_dataset(args.input_dir, args.output_dir, files=tuple(args.files))
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))


def _metric_line(name: str, count: int, denominator: int) -> str:
    percent = count / denominator if denominator else 0
    return f"- {name}: {count} ({percent:.2%})"


def _counter_lines(counter: dict[str, int], denominator: int) -> list[str]:
    if not counter:
        return ["- none"]
    return [
        _metric_line(reason, count, denominator)
        for reason, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


if __name__ == "__main__":
    main()
