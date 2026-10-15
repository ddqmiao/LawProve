"""Filter CMDL/CDML JSONL files by simple size thresholds and report flaws."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JsonRecord = dict[str, Any]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "CDML_big"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "cdml_big_len12000_def6"
DEFAULT_MAX_TOTAL_LEN = 12000
DEFAULT_MAX_DEFENDANTS = 6
DEFAULT_FILES = ("train_big.jsonl", "valid_big.jsonl", "test_big.jsonl")
TERMINAL_PUNCTUATION = frozenset("。；;！？!?.．.”」』）】")
SUBSTANTIVE_COURT_TERMS = (
    "构成",
    "指控",
    "成立",
    "从轻",
    "从重",
    "主犯",
    "从犯",
    "自首",
    "坦白",
    "累犯",
    "缓刑",
    "认罪",
    "认罚",
    "处罚",
    "采纳",
)
LOW_RISK_TAIL_EXACT = frozenset({"据此，", "据此,", "综上，", "综上,", "综上所述，", "综上所述,", "为此，", "为此,"})
LOW_RISK_TAIL_PREFIXES = (
    "据此，",
    "据此,",
    "综上，",
    "综上,",
    "综上所述，",
    "综上所述,",
    "根据被告人",
    "根据二被告人",
    "根据各被告人",
    "根据本案",
    "为维护",
    "为了维护",
    "为保护",
    "为了保护",
    "本院为保护",
    "本院为维护",
)
HIGH_RISK_TAIL_TERMS = ("判处", "并处罚金", "决定执行", "判决如下", "依照《", "中华人民共和国刑法", "对被告人", "第")


@dataclass
class FlawStats:
    flawed_records: int
    flaw_reasons: Counter[str]

    @classmethod
    def empty(cls) -> "FlawStats":
        return cls(flawed_records=0, flaw_reasons=Counter())

    def add(self, reasons: list[str]) -> None:
        if not reasons:
            return
        self.flawed_records += 1
        self.flaw_reasons.update(reasons)

    def to_dict(self) -> JsonRecord:
        return {
            "flawed_records": self.flawed_records,
            "flaw_reasons": dict(self.flaw_reasons),
        }


def compute_metrics(record: JsonRecord) -> JsonRecord:
    fact = record.get("fact")
    court_view = record.get("court_view")
    defendants = record.get("defendants")
    fact_len = len(fact) if isinstance(fact, str) else 0
    court_view_len = len(court_view) if isinstance(court_view, str) else 0
    defendant_count = len(defendants) if isinstance(defendants, list) else 0
    return {
        "fact_len": fact_len,
        "court_view_len": court_view_len,
        "total_len": fact_len + court_view_len,
        "defendant_count": defendant_count,
    }


def keep_by_thresholds(metrics: JsonRecord, *, max_total_len: int, max_defendants: int) -> list[str]:
    reasons = []
    if int(metrics["total_len"]) > max_total_len:
        reasons.append(f"total_len_gt_{max_total_len}")
    if int(metrics["defendant_count"]) > max_defendants:
        reasons.append(f"defendant_count_gt_{max_defendants}")
    return reasons


def analyze_flaws(record: JsonRecord) -> list[str]:
    """Return high-confidence flaw reasons without applying length/complexity thresholds."""

    reasons: list[str] = []
    fact = record.get("fact")
    court_view = record.get("court_view")
    defendants = record.get("defendants")
    outcomes = record.get("outcomes")
    relevant_articles = record.get("relevant_articles")

    if not isinstance(fact, str) or not fact.strip():
        reasons.append("bad_fact")
    elif fact.strip()[-1] not in TERMINAL_PUNCTUATION:
        reasons.append("fact_no_terminal_punct")

    if not isinstance(court_view, str) or not court_view.strip():
        reasons.append("bad_court_view")
    elif court_view.strip()[-1] not in TERMINAL_PUNCTUATION:
        reasons.append("court_view_no_terminal_punct")

    if not isinstance(defendants, list) or not defendants:
        reasons.append("bad_defendants")

    if not isinstance(outcomes, list) or not outcomes:
        reasons.append("bad_outcomes")

    if isinstance(defendants, list) and isinstance(outcomes, list):
        if len(defendants) != len(outcomes):
            reasons.append("defendant_outcome_len_mismatch")
        else:
            for defendant, outcome in zip(defendants, outcomes):
                if not isinstance(outcome, dict) or str(defendant).strip() != str(outcome.get("name", "")).strip():
                    reasons.append("defendant_outcome_name_mismatch")
                    break

    if not isinstance(relevant_articles, list) or not relevant_articles:
        reasons.append("bad_relevant_articles")

    return reasons


def classify_court_view_truncation(court_view: Any) -> JsonRecord:
    """Classify a non-terminal court_view tail and return a cleaned text when safe."""

    if not isinstance(court_view, str) or not court_view.strip():
        return {
            "severity": "severe_reject",
            "reason": "bad_court_view",
            "cleaned_court_view": "",
            "tail_fragment": "",
        }

    text = court_view.strip()
    if text[-1] in TERMINAL_PUNCTUATION:
        return {
            "severity": "none",
            "reason": "",
            "cleaned_court_view": text,
            "tail_fragment": "",
        }

    cleaned, tail = _split_at_last_terminal_punctuation(text)
    if not tail:
        return {
            "severity": "none",
            "reason": "",
            "cleaned_court_view": cleaned,
            "tail_fragment": "",
        }

    reason = _minor_trim_reject_reason(text, cleaned, tail)
    if reason:
        return {
            "severity": "severe_reject",
            "reason": reason,
            "cleaned_court_view": cleaned,
            "tail_fragment": tail,
        }

    return {
        "severity": "minor_trimmed",
        "reason": "",
        "cleaned_court_view": cleaned,
        "tail_fragment": tail,
    }


def process_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    rejected_path: str | Path,
    *,
    max_total_len: int,
    max_defendants: int,
    split_name: str,
    reject_flaws: bool = False,
    clean_minor_flaws: bool = False,
    keep_metadata: bool = True,
) -> JsonRecord:
    input_path = Path(input_path)
    output_path = Path(output_path)
    rejected_path = Path(rejected_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)

    read_count = 0
    kept_count = 0
    rejected_count = 0
    overall_flaws = FlawStats.empty()
    kept_flaws = FlawStats.empty()
    reject_reasons_counter: Counter[str] = Counter()
    truncation_severity_counter: Counter[str] = Counter()
    json_errors = 0
    trimmed_records = 0

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
                json_errors += 1
                rejected_count += 1
                reject_reasons_counter["json_decode_error"] += 1
                _write_rejected(
                    rejected_target,
                    split_name=split_name,
                    line_number=line_number,
                    reject_reasons=["json_decode_error"],
                    metrics={},
                    flaw_reasons=[],
                    extra={"error": str(exc)},
                )
                continue

            if not isinstance(record, dict):
                rejected_count += 1
                reject_reasons_counter["non_object"] += 1
                _write_rejected(
                    rejected_target,
                    split_name=split_name,
                    line_number=line_number,
                    reject_reasons=["non_object"],
                    metrics={},
                    flaw_reasons=[],
                    extra={"type": type(record).__name__},
                )
                continue

            metrics = compute_metrics(record)
            flaw_reasons = analyze_flaws(record)
            overall_flaws.add(flaw_reasons)
            reject_reasons = keep_by_thresholds(
                metrics,
                max_total_len=max_total_len,
                max_defendants=max_defendants,
            )
            if reject_flaws:
                reject_reasons.extend(flaw_reasons)

            if reject_reasons:
                rejected_count += 1
                reject_reasons_counter.update(reject_reasons)
                _write_rejected(
                    rejected_target,
                    split_name=split_name,
                    line_number=line_number,
                    reject_reasons=reject_reasons,
                    metrics=metrics,
                    flaw_reasons=flaw_reasons,
                )
                continue

            clean_info = _prepare_clean_record(record, flaw_reasons, clean_minor_flaws=clean_minor_flaws)
            if clean_info["reject_reasons"]:
                rejected_count += 1
                reject_reasons_counter.update(clean_info["reject_reasons"])
                truncation_severity_counter.update([clean_info["truncation_severity"]])
                _write_rejected(
                    rejected_target,
                    split_name=split_name,
                    line_number=line_number,
                    reject_reasons=clean_info["reject_reasons"],
                    metrics=metrics,
                    flaw_reasons=flaw_reasons,
                    extra={
                        "tail_fragment": clean_info["tail_fragment"],
                        "truncation_severity": clean_info["truncation_severity"],
                        "truncation_reason": clean_info["truncation_reason"],
                    },
                )
                continue

            record = clean_info["record"]
            final_metrics = compute_metrics(record)
            kept_record = dict(record)
            if keep_metadata:
                kept_record.update(
                    {
                        "source_split": split_name,
                        "source_line_no": line_number,
                        "fact_len": final_metrics["fact_len"],
                        "court_view_len": final_metrics["court_view_len"],
                        "total_len": final_metrics["total_len"],
                        "defendant_count": final_metrics["defendant_count"],
                        "raw_fact_len": metrics["fact_len"],
                        "raw_court_view_len": metrics["court_view_len"],
                        "raw_total_len": metrics["total_len"],
                        "flaw_reasons": flaw_reasons,
                        "court_view_trimmed": clean_info["court_view_trimmed"],
                        "tail_fragment": clean_info["tail_fragment"],
                        "truncation_severity": clean_info["truncation_severity"],
                    }
                )
                if clean_info["court_view_original"] is not None:
                    kept_record["court_view_original"] = clean_info["court_view_original"]
            kept_target.write(json.dumps(kept_record, ensure_ascii=False, separators=(",", ":")) + "\n")
            kept_count += 1
            if clean_info["court_view_trimmed"]:
                trimmed_records += 1
            truncation_severity_counter.update([clean_info["truncation_severity"]])
            kept_flaws.add(flaw_reasons)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "rejected_output": str(rejected_path),
        "split_name": split_name,
        "read": read_count,
        "kept": kept_count,
        "rejected": rejected_count,
        "json_errors": json_errors,
        "reject_reasons": dict(reject_reasons_counter),
        "flawed_records": overall_flaws.flawed_records,
        "flaw_reasons": dict(overall_flaws.flaw_reasons),
        "kept_flaw_stats": kept_flaws.to_dict(),
        "trimmed_records": trimmed_records,
        "truncation_severity": dict(truncation_severity_counter),
    }


def process_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    files: tuple[str, ...] = DEFAULT_FILES,
    max_total_len: int = DEFAULT_MAX_TOTAL_LEN,
    max_defendants: int = DEFAULT_MAX_DEFENDANTS,
    reject_flaws: bool = False,
    clean_minor_flaws: bool = False,
    keep_metadata: bool = True,
) -> JsonRecord:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for filename in files:
        split_name = filename.removesuffix(".jsonl")
        output_path = output_dir / f"{split_name}.len{max_total_len}.def{max_defendants}.jsonl"
        rejected_path = output_dir / f"{split_name}.len{max_total_len}.def{max_defendants}.rejected.jsonl"
        summaries.append(
            process_jsonl(
                input_dir / filename,
                output_path,
                rejected_path,
                max_total_len=max_total_len,
                max_defendants=max_defendants,
                split_name=split_name,
                reject_flaws=reject_flaws,
                clean_minor_flaws=clean_minor_flaws,
                keep_metadata=keep_metadata,
            )
        )

    aggregate = aggregate_summaries(summaries)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "max_total_len": max_total_len,
        "max_defendants": max_defendants,
        "reject_flaws": reject_flaws,
        "clean_minor_flaws": clean_minor_flaws,
        "keep_metadata": keep_metadata,
        "splits": summaries,
        "aggregate": aggregate,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "filter_report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def aggregate_summaries(summaries: list[JsonRecord]) -> JsonRecord:
    aggregate: JsonRecord = {
        "read": 0,
        "kept": 0,
        "rejected": 0,
        "json_errors": 0,
        "flawed_records": 0,
        "reject_reasons": Counter(),
        "flaw_reasons": Counter(),
        "trimmed_records": 0,
        "truncation_severity": Counter(),
        "kept_flaw_stats": {
            "flawed_records": 0,
            "flaw_reasons": Counter(),
        },
    }
    for summary in summaries:
        for key in ("read", "kept", "rejected", "json_errors", "flawed_records", "trimmed_records"):
            aggregate[key] += int(summary[key])
        aggregate["reject_reasons"].update(summary["reject_reasons"])
        aggregate["flaw_reasons"].update(summary["flaw_reasons"])
        aggregate["truncation_severity"].update(summary["truncation_severity"])
        aggregate["kept_flaw_stats"]["flawed_records"] += int(summary["kept_flaw_stats"]["flawed_records"])
        aggregate["kept_flaw_stats"]["flaw_reasons"].update(summary["kept_flaw_stats"]["flaw_reasons"])

    aggregate["reject_reasons"] = dict(aggregate["reject_reasons"])
    aggregate["flaw_reasons"] = dict(aggregate["flaw_reasons"])
    aggregate["truncation_severity"] = dict(aggregate["truncation_severity"])
    aggregate["kept_flaw_stats"]["flaw_reasons"] = dict(aggregate["kept_flaw_stats"]["flaw_reasons"])
    return aggregate


def render_report(summary: JsonRecord) -> str:
    aggregate = summary["aggregate"]
    lines = [
        "# CDML Big Filter Report",
        "",
        f"- input_dir: `{summary['input_dir']}`",
        f"- output_dir: `{summary['output_dir']}`",
        f"- max_total_len: `{summary['max_total_len']}`",
        f"- max_defendants: `{summary['max_defendants']}`",
        f"- reject_flaws: `{summary['reject_flaws']}`",
        f"- clean_minor_flaws: `{summary['clean_minor_flaws']}`",
        f"- keep_metadata: `{summary['keep_metadata']}`",
        "",
        "## Aggregate",
        "",
        _metric_line("read", aggregate["read"], aggregate["read"]),
        _metric_line("kept", aggregate["kept"], aggregate["read"]),
        _metric_line("rejected", aggregate["rejected"], aggregate["read"]),
        _metric_line("trimmed_records", aggregate["trimmed_records"], aggregate["read"]),
        _metric_line("flawed_records_in_input", aggregate["flawed_records"], aggregate["read"]),
        _metric_line(
            "flawed_records_in_kept",
            aggregate["kept_flaw_stats"]["flawed_records"],
            aggregate["kept"],
        ),
        "",
        "## Reject Reasons",
        "",
        *_counter_lines(aggregate["reject_reasons"], aggregate["read"]),
        "",
        "## Flaw Reasons In Input",
        "",
        *_counter_lines(aggregate["flaw_reasons"], aggregate["read"]),
        "",
        "## Flaw Reasons In Kept",
        "",
        *_counter_lines(aggregate["kept_flaw_stats"]["flaw_reasons"], aggregate["kept"]),
        "",
        "## Truncation Severity",
        "",
        *_counter_lines(aggregate["truncation_severity"], aggregate["read"]),
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
                _metric_line("trimmed_records", split["trimmed_records"], split["read"]),
                _metric_line("flawed_records_in_input", split["flawed_records"], split["read"]),
                _metric_line("flawed_records_in_kept", split["kept_flaw_stats"]["flawed_records"], split["kept"]),
                "",
            ]
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter CDML/CMDL JSONL files by total length and defendant count.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help=f"Input dataset directory. Default: {DEFAULT_INPUT_DIR}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--max-total-len", type=int, default=DEFAULT_MAX_TOTAL_LEN)
    parser.add_argument("--max-defendants", type=int, default=DEFAULT_MAX_DEFENDANTS)
    parser.add_argument("--files", nargs="+", default=list(DEFAULT_FILES), help="JSONL filenames to process.")
    parser.add_argument(
        "--reject-flaws",
        action="store_true",
        help="Also reject high-confidence flawed records. By default flaws are only reported.",
    )
    parser.add_argument(
        "--clean-minor-flaws",
        action="store_true",
        help="Keep complete records and minor court_view tail trims only; reject all other flaws.",
    )
    parser.add_argument(
        "--clean-output-schema",
        action="store_true",
        help="Write kept records with only original fields after cleaning, without trace/metadata fields.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = process_dataset(
        args.input_dir,
        args.output_dir,
        files=tuple(args.files),
        max_total_len=args.max_total_len,
        max_defendants=args.max_defendants,
        reject_flaws=args.reject_flaws,
        clean_minor_flaws=args.clean_minor_flaws,
        keep_metadata=not args.clean_output_schema,
    )
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))


def _prepare_clean_record(record: JsonRecord, flaw_reasons: list[str], *, clean_minor_flaws: bool) -> JsonRecord:
    if not clean_minor_flaws:
        return {
            "record": record,
            "reject_reasons": [],
            "court_view_trimmed": False,
            "court_view_original": None,
            "tail_fragment": "",
            "truncation_severity": "none" if "court_view_no_terminal_punct" not in flaw_reasons else "unprocessed",
            "truncation_reason": "",
        }

    if not flaw_reasons:
        return {
            "record": record,
            "reject_reasons": [],
            "court_view_trimmed": False,
            "court_view_original": None,
            "tail_fragment": "",
            "truncation_severity": "none",
            "truncation_reason": "",
        }

    non_court_view_flaws = [reason for reason in flaw_reasons if reason != "court_view_no_terminal_punct"]
    if non_court_view_flaws:
        return {
            "record": record,
            "reject_reasons": non_court_view_flaws,
            "court_view_trimmed": False,
            "court_view_original": None,
            "tail_fragment": "",
            "truncation_severity": "severe_reject",
            "truncation_reason": "non_court_view_flaw",
        }

    truncation = classify_court_view_truncation(record.get("court_view"))
    if truncation["severity"] != "minor_trimmed":
        return {
            "record": record,
            "reject_reasons": ["court_view_truncation_not_minor"],
            "court_view_trimmed": False,
            "court_view_original": None,
            "tail_fragment": str(truncation["tail_fragment"]),
            "truncation_severity": "severe_reject",
            "truncation_reason": str(truncation["reason"]),
        }

    cleaned_record = dict(record)
    original_court_view = str(record.get("court_view") or "")
    cleaned_record["court_view"] = str(truncation["cleaned_court_view"])
    return {
        "record": cleaned_record,
        "reject_reasons": [],
        "court_view_trimmed": True,
        "court_view_original": original_court_view,
        "tail_fragment": str(truncation["tail_fragment"]),
        "truncation_severity": "minor_trimmed",
        "truncation_reason": "",
    }


def _split_at_last_terminal_punctuation(text: str) -> tuple[str, str]:
    last_index = -1
    for index, char in enumerate(text):
        if char in TERMINAL_PUNCTUATION:
            last_index = index
    if last_index < 0:
        return "", text
    return text[: last_index + 1], text[last_index + 1 :]


def _minor_trim_reject_reason(original: str, cleaned: str, tail: str) -> str:
    tail = tail.strip()
    if len(tail) > 80:
        return "tail_len_gt_80"
    if len(tail) / max(len(original), 1) > 0.15:
        return "tail_ratio_gt_15pct"
    if _sentence_count(cleaned) < 2:
        return "trimmed_sent_lt_2"
    if len(cleaned) < 80:
        return "trimmed_len_lt_80"
    if "本院认为" not in cleaned:
        return "trimmed_no_benyuanrenwei"
    if not any(term in cleaned for term in SUBSTANTIVE_COURT_TERMS):
        return "trimmed_no_substantive_term"
    if not _is_low_risk_tail(tail):
        return "tail_pattern_not_low_risk"
    return ""


def _sentence_count(text: str) -> int:
    return sum(1 for sentence in re.split(r"(?<=[。；;！？!?])", text) if sentence.strip())


def _is_low_risk_tail(tail: str) -> bool:
    tail = tail.strip()
    if tail in LOW_RISK_TAIL_EXACT:
        return True
    if not tail.startswith(LOW_RISK_TAIL_PREFIXES):
        return False
    return not any(term in tail for term in HIGH_RISK_TAIL_TERMS)


def _write_rejected(
    target: Any,
    *,
    split_name: str,
    line_number: int,
    reject_reasons: list[str],
    metrics: JsonRecord,
    flaw_reasons: list[str],
    extra: JsonRecord | None = None,
) -> None:
    row = {
        "source_split": split_name,
        "source_line_no": line_number,
        "reject_reasons": reject_reasons,
        "flaw_reasons": flaw_reasons,
        **metrics,
    }
    if extra:
        row.update(extra)
    target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


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
