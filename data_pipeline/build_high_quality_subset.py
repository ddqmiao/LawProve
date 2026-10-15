"""Build a 20k high-quality, diverse CDML subset from cleaned JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from clean_court_view_contamination import count_evidence_hits


JsonRecord = dict[str, Any]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "outputs" / "cdml_big_len8000_def6_minor_trim_clean_no_cv_contamination"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "cdml_big_high_quality_20k"
DEFAULT_FILES = (
    "train_big.len8000.def6.jsonl",
    "valid_big.len8000.def6.jsonl",
    "test_big.len8000.def6.jsonl",
)
DEFAULT_TARGET_COUNT = 20000
DEFAULT_SEED = 20260624
DEFAULT_DEFENDANT_WEIGHTS = {2: 0.30, 3: 0.25, 4: 0.20, 5: 0.15, 6: 0.10}
SPLIT_WEIGHTS = {"train_big": 0.80, "valid_big": 0.10, "test_big": 0.10}
CASE_ID_PREFIXES = {"train_big": "train", "valid_big": "valid", "test_big": "test"}
COURT_TERMS = ("本院认为", "构成", "罪名成立", "指控成立", "从轻", "从重", "主犯", "从犯", "自首", "坦白", "累犯", "缓刑", "认罪认罚", "量刑建议", "共同犯罪")
BAD_COURT_VIEW_STARTS = ("公诉机关认为", "公诉机关指控", "被告人", "辩护人", "经审理查明")
TOTAL_LEN_LIMIT_BY_DEFENDANTS = {2: 2800, 3: 3400, 4: 4000, 5: 4600, 6: 5200}
COURT_VIEW_LEN_LIMIT_BY_DEFENDANTS = {2: 1000, 3: 1200, 4: 1400, 5: 1600, 6: 1800}


@dataclass(frozen=True)
class Candidate:
    accepted: bool
    reasons: tuple[str, ...]
    record: JsonRecord | None = None
    split_name: str = ""
    line_number: int = 0
    charge: str = ""
    defendant_count: int = 0
    fact_len: int = 0
    court_view_len: int = 0
    total_len: int = 0
    article_count: int = 0


def is_high_quality_candidate(record: JsonRecord, *, split_name: str = "", line_number: int = 0) -> Candidate:
    reasons: list[str] = []
    fact = record.get("fact")
    court_view = record.get("court_view")
    defendants = record.get("defendants")
    fact_text = fact if isinstance(fact, str) else ""
    court_view_text = court_view if isinstance(court_view, str) else ""
    defendant_count = len(defendants) if isinstance(defendants, list) else 0
    accusations = _accusations(record)
    charge = accusations[0] if len(set(accusations)) == 1 and accusations else ""
    total_len = len(fact_text) + len(court_view_text)
    article_total = article_count(record)

    if not (2 <= defendant_count <= 6):
        reasons.append("defendant_count_not_2_to_6")
    if not _outcomes_align(record):
        reasons.append("defendant_outcome_not_aligned")
    if not _single_judgment(record):
        reasons.append("not_single_judgment")
    if not charge:
        reasons.append("not_single_accusation")
    if not record.get("relevant_articles"):
        reasons.append("empty_relevant_articles")
    if not _all_have_penalty(record):
        reasons.append("missing_penalty")
    if len(fact_text) < 250:
        reasons.append("fact_len_lt_250")
    if len(court_view_text) < 150:
        reasons.append("court_view_len_lt_150")
    if defendant_count in TOTAL_LEN_LIMIT_BY_DEFENDANTS and total_len > TOTAL_LEN_LIMIT_BY_DEFENDANTS[defendant_count]:
        reasons.append(f"total_len_gt_{TOTAL_LEN_LIMIT_BY_DEFENDANTS[defendant_count]}")
    if defendant_count in COURT_VIEW_LEN_LIMIT_BY_DEFENDANTS and len(court_view_text) > COURT_VIEW_LEN_LIMIT_BY_DEFENDANTS[defendant_count]:
        reasons.append(f"court_view_len_gt_{COURT_VIEW_LEN_LIMIT_BY_DEFENDANTS[defendant_count]}")
    if "本院认为" not in court_view_text:
        reasons.append("court_view_missing_benyuanrenwei")
    if not any(term in court_view_text for term in COURT_TERMS):
        reasons.append("court_view_missing_substantive_term")
    if court_view_text.startswith(BAD_COURT_VIEW_STARTS):
        reasons.append("court_view_bad_start")
    if _sentence_count(court_view_text) < 2:
        reasons.append("court_view_sentence_count_lt_2")
    if count_evidence_hits(court_view_text) > 2:
        reasons.append("court_view_evidence_hits_gt_2")
    if article_total > 8:
        reasons.append("article_count_gt_8")
    if not _judgment_articles_ok(record, max_per_judgment=4):
        reasons.append("judgment_article_count_gt_4")

    return Candidate(
        accepted=not reasons,
        reasons=tuple(reasons),
        record=record if not reasons else None,
        split_name=split_name,
        line_number=line_number,
        charge=charge,
        defendant_count=defendant_count,
        fact_len=len(fact_text),
        court_view_len=len(court_view_text),
        total_len=total_len,
        article_count=article_total,
    )


def choose_split_targets(split_names: list[str], target_count: int) -> dict[str, int]:
    targets: dict[str, int] = {}
    remaining = target_count
    normalized = [_base_split_name(name) for name in split_names]
    for split_name, base_name in zip(split_names[:-1], normalized[:-1]):
        target = round(target_count * SPLIT_WEIGHTS.get(base_name, 1 / len(split_names)))
        targets[split_name] = target
        remaining -= target
    targets[split_names[-1]] = remaining
    return targets


def select_diverse_candidates(
    candidates: list[Candidate],
    *,
    target_count: int,
    defendant_weights: dict[int, float] = DEFAULT_DEFENDANT_WEIGHTS,
    max_charge_share: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> list[Candidate]:
    accepted = [candidate for candidate in candidates if candidate.accepted]
    if target_count >= len(accepted):
        return sorted(accepted, key=lambda candidate: (candidate.split_name, candidate.line_number))

    target_by_defendant = _bucket_targets(defendant_weights, target_count)
    charge_cap = max(1, math.ceil(target_count * max_charge_share))
    selected: list[Candidate] = []
    selected_ids: set[tuple[str, int]] = set()
    charge_counts: Counter[str] = Counter()

    for defendant_count in sorted(target_by_defendant):
        pool = [candidate for candidate in accepted if candidate.defendant_count == defendant_count]
        selected.extend(
            _round_robin_by_charge(
                pool,
                target_by_defendant[defendant_count],
                charge_cap=charge_cap,
                charge_counts=charge_counts,
                selected_ids=selected_ids,
                seed=seed,
            )
        )

    if len(selected) < target_count:
        selected.extend(
            _fill_remaining(
                accepted,
                target_count - len(selected),
                charge_cap=charge_cap,
                charge_counts=charge_counts,
                selected_ids=selected_ids,
                seed=seed,
                enforce_charge_cap=True,
            )
        )

    if len(selected) < target_count:
        selected.extend(
            _fill_remaining(
                accepted,
                target_count - len(selected),
                charge_cap=charge_cap,
                charge_counts=charge_counts,
                selected_ids=selected_ids,
                seed=seed,
                enforce_charge_cap=False,
            )
        )

    return sorted(selected[:target_count], key=lambda candidate: (candidate.split_name, candidate.line_number))


def case_id_for(split_name: str, output_index: int) -> str:
    prefix = CASE_ID_PREFIXES.get(_base_split_name(split_name), _base_split_name(split_name))
    return f"{prefix}_{output_index:06d}"


def record_with_case_id(record: JsonRecord, case_id: str) -> JsonRecord:
    clean_record = dict(record)
    clean_record.pop("case_id", None)
    return {"case_id": case_id, **clean_record}


def process_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    files: tuple[str, ...] = DEFAULT_FILES,
    target_count: int = DEFAULT_TARGET_COUNT,
    seed: int = DEFAULT_SEED,
    max_charge_share: float = 0.15,
) -> JsonRecord:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates_by_split: dict[str, list[Candidate]] = {}
    rejection_reasons_by_split: dict[str, Counter[str]] = {}
    read_by_split: Counter[str] = Counter()
    for filename in files:
        split_name = filename.removesuffix(".jsonl")
        candidates: list[Candidate] = []
        rejection_reasons: Counter[str] = Counter()
        with (input_dir / filename).open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                read_by_split[split_name] += 1
                record = json.loads(line)
                candidate = is_high_quality_candidate(record, split_name=split_name, line_number=line_number)
                if candidate.accepted:
                    candidates.append(candidate)
                else:
                    rejection_reasons.update(candidate.reasons)
        candidates_by_split[split_name] = candidates
        rejection_reasons_by_split[split_name] = rejection_reasons

    split_targets = choose_split_targets(list(candidates_by_split), target_count)
    summaries = []
    selected_all: list[Candidate] = []
    for filename in files:
        split_name = filename.removesuffix(".jsonl")
        selected = select_diverse_candidates(
            candidates_by_split[split_name],
            target_count=min(split_targets[split_name], len(candidates_by_split[split_name])),
            max_charge_share=max_charge_share,
            seed=seed,
        )
        selected_all.extend(selected)
        output_path = output_dir / filename.replace(".jsonl", ".high_quality.jsonl")
        with output_path.open("w", encoding="utf-8") as target:
            for output_index, candidate in enumerate(selected, start=1):
                record = record_with_case_id(candidate.record or {}, case_id_for(split_name, output_index))
                target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        summaries.append(
            _split_summary(
                split_name=split_name,
                read_count=read_by_split[split_name],
                candidates=candidates_by_split[split_name],
                selected=selected,
                output_path=output_path,
                rejection_reasons=rejection_reasons_by_split[split_name],
            )
        )

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "target_count": target_count,
        "seed": seed,
        "max_charge_share": max_charge_share,
        "case_id": {
            "field": "case_id",
            "kind": "artificial_sequential_id",
            "formats": {
                "train_big": "train_000001",
                "valid_big": "valid_000001",
                "test_big": "test_000001",
            },
        },
        "split_targets": split_targets,
        "splits": summaries,
        "aggregate": _aggregate_summary(read_by_split, candidates_by_split, selected_all, rejection_reasons_by_split),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "sample_report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def render_report(summary: JsonRecord) -> str:
    aggregate = summary["aggregate"]
    lines = [
        "# CDML High Quality 20k Report",
        "",
        f"- input_dir: `{summary['input_dir']}`",
        f"- output_dir: `{summary['output_dir']}`",
        f"- target_count: `{summary['target_count']}`",
        f"- seed: `{summary['seed']}`",
        f"- max_charge_share: `{summary['max_charge_share']}`",
        f"- case_id: artificial sequential `{summary.get('case_id', {}).get('field', 'case_id')}`",
        "",
        "## Aggregate",
        "",
        _metric_line("read", aggregate["read"], aggregate["read"]),
        _metric_line("candidate_count", aggregate["candidate_count"], aggregate["read"]),
        _metric_line("selected_count", aggregate["selected_count"], aggregate["read"]),
        "",
        "## Selected Defendant Count",
        "",
        *_counter_lines(aggregate["selected_defendant_counts"], aggregate["selected_count"]),
        "",
        "## Selected Charges",
        "",
        *_counter_lines(aggregate["selected_charges"], aggregate["selected_count"], limit=60),
        "",
        "## Top Candidate Rejection Reasons",
        "",
        *_counter_lines(aggregate["candidate_rejection_reasons"], aggregate["read"], limit=30),
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
                _metric_line("candidate_count", split["candidate_count"], split["read"]),
                _metric_line("selected_count", split["selected_count"], split["read"]),
                "",
            ]
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a high-quality diverse 20k CDML subset.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help=f"Input directory. Default: {DEFAULT_INPUT_DIR}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--files", nargs="+", default=list(DEFAULT_FILES), help="Input JSONL filenames.")
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-charge-share", type=float, default=0.15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = process_dataset(
        args.input_dir,
        args.output_dir,
        files=tuple(args.files),
        target_count=args.target_count,
        seed=args.seed,
        max_charge_share=args.max_charge_share,
    )
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))


def _accusations(record: JsonRecord) -> list[str]:
    values: list[str] = []
    for outcome in record.get("outcomes") or []:
        if not isinstance(outcome, dict):
            continue
        for judgment in outcome.get("judgment") or []:
            if isinstance(judgment, dict):
                accusation = str(judgment.get("standard_accusation") or judgment.get("accusation") or "").strip()
                if accusation:
                    values.append(accusation)
    return values


def _outcomes_align(record: JsonRecord) -> bool:
    defendants = record.get("defendants")
    outcomes = record.get("outcomes")
    return isinstance(defendants, list) and isinstance(outcomes, list) and len(defendants) == len(outcomes) and all(
        isinstance(outcome, dict) and str(defendant).strip() == str(outcome.get("name", "")).strip()
        for defendant, outcome in zip(defendants, outcomes)
    )


def _single_judgment(record: JsonRecord) -> bool:
    outcomes = record.get("outcomes")
    return isinstance(outcomes, list) and bool(outcomes) and all(
        isinstance(outcome, dict)
        and isinstance(outcome.get("judgment"), list)
        and len(outcome["judgment"]) == 1
        and isinstance(outcome["judgment"][0], dict)
        for outcome in outcomes
    )


def _all_have_penalty(record: JsonRecord) -> bool:
    for outcome in record.get("outcomes") or []:
        if not isinstance(outcome, dict):
            return False
        judgments = outcome.get("judgment") or []
        if not judgments or not isinstance(judgments[0], dict) or not isinstance(judgments[0].get("penalty"), dict):
            return False
    return True


def article_count(record: JsonRecord) -> int:
    articles = {str(article).strip() for article in record.get("relevant_articles") or [] if str(article).strip()}
    for outcome in record.get("outcomes") or []:
        if not isinstance(outcome, dict):
            continue
        for judgment in outcome.get("judgment") or []:
            if not isinstance(judgment, dict):
                continue
            articles.update(str(article).strip() for article in judgment.get("article") or [] if str(article).strip())
    return len(articles)


def _judgment_articles_ok(record: JsonRecord, *, max_per_judgment: int) -> bool:
    for outcome in record.get("outcomes") or []:
        if not isinstance(outcome, dict):
            continue
        for judgment in outcome.get("judgment") or []:
            if isinstance(judgment, dict) and len(judgment.get("article") or []) > max_per_judgment:
                return False
    return True


def _sentence_count(text: str) -> int:
    return sum(1 for sentence in re.split(r"(?<=[。；;！？!?])", text) if sentence.strip())


def _bucket_targets(weights: dict[int, float], target_count: int) -> dict[int, int]:
    targets: dict[int, int] = {}
    remaining = target_count
    buckets = sorted(weights)
    for bucket in buckets[:-1]:
        count = math.floor(target_count * weights[bucket])
        targets[bucket] = count
        remaining -= count
    targets[buckets[-1]] = remaining
    return targets


def _round_robin_by_charge(
    candidates: list[Candidate],
    target_count: int,
    *,
    charge_cap: int,
    charge_counts: Counter[str],
    selected_ids: set[tuple[str, int]],
    seed: int,
) -> list[Candidate]:
    by_charge: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_charge[candidate.charge].append(candidate)
    for charge in by_charge:
        by_charge[charge].sort(key=lambda candidate: _stable_hash(seed, candidate))

    selected: list[Candidate] = []
    charges = sorted(by_charge)
    while len(selected) < target_count and charges:
        progressed = False
        for charge in list(charges):
            if len(selected) >= target_count:
                break
            if charge_counts[charge] >= charge_cap:
                charges.remove(charge)
                continue
            while by_charge[charge]:
                candidate = by_charge[charge].pop(0)
                candidate_id = (candidate.split_name, candidate.line_number)
                if candidate_id in selected_ids:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate_id)
                charge_counts[charge] += 1
                progressed = True
                break
            if not by_charge[charge] and charge in charges:
                charges.remove(charge)
        if not progressed:
            break
    return selected


def _fill_remaining(
    candidates: list[Candidate],
    target_count: int,
    *,
    charge_cap: int,
    charge_counts: Counter[str],
    selected_ids: set[tuple[str, int]],
    seed: int,
    enforce_charge_cap: bool,
) -> list[Candidate]:
    selected: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: _stable_hash(seed, item)):
        if len(selected) >= target_count:
            break
        candidate_id = (candidate.split_name, candidate.line_number)
        if candidate_id in selected_ids:
            continue
        if enforce_charge_cap and charge_counts[candidate.charge] >= charge_cap:
            continue
        selected.append(candidate)
        selected_ids.add(candidate_id)
        charge_counts[candidate.charge] += 1
    return selected


def _stable_hash(seed: int, candidate: Candidate) -> str:
    value = f"{seed}:{candidate.split_name}:{candidate.line_number}:{candidate.charge}:{candidate.defendant_count}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _base_split_name(split_name: str) -> str:
    if split_name.startswith("train_big"):
        return "train_big"
    if split_name.startswith("valid_big"):
        return "valid_big"
    if split_name.startswith("test_big"):
        return "test_big"
    return split_name


def _split_summary(
    *,
    split_name: str,
    read_count: int,
    candidates: list[Candidate],
    selected: list[Candidate],
    output_path: Path,
    rejection_reasons: Counter[str],
) -> JsonRecord:
    return {
        "split_name": split_name,
        "read": read_count,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "output": str(output_path),
        "candidate_defendant_counts": dict(Counter(candidate.defendant_count for candidate in candidates)),
        "selected_defendant_counts": dict(Counter(candidate.defendant_count for candidate in selected)),
        "candidate_charges": dict(Counter(candidate.charge for candidate in candidates)),
        "selected_charges": dict(Counter(candidate.charge for candidate in selected)),
        "candidate_rejection_reasons": dict(rejection_reasons),
    }


def _aggregate_summary(
    read_by_split: Counter[str],
    candidates_by_split: dict[str, list[Candidate]],
    selected: list[Candidate],
    rejection_reasons_by_split: dict[str, Counter[str]],
) -> JsonRecord:
    all_candidates = [candidate for candidates in candidates_by_split.values() for candidate in candidates]
    rejection_reasons: Counter[str] = Counter()
    for reasons in rejection_reasons_by_split.values():
        rejection_reasons.update(reasons)
    return {
        "read": sum(read_by_split.values()),
        "candidate_count": len(all_candidates),
        "selected_count": len(selected),
        "candidate_defendant_counts": dict(Counter(candidate.defendant_count for candidate in all_candidates)),
        "selected_defendant_counts": dict(Counter(candidate.defendant_count for candidate in selected)),
        "candidate_charges": dict(Counter(candidate.charge for candidate in all_candidates)),
        "selected_charges": dict(Counter(candidate.charge for candidate in selected)),
        "candidate_rejection_reasons": dict(rejection_reasons),
    }


def _metric_line(name: str, count: int, denominator: int) -> str:
    return f"- {name}: {count} ({(count / denominator if denominator else 0):.2%})"


def _counter_lines(counter: dict[str, int], denominator: int, *, limit: int | None = None) -> list[str]:
    if not counter:
        return ["- none"]
    rows = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        rows = rows[:limit]
    return [_metric_line(str(key), int(value), denominator) for key, value in rows]


if __name__ == "__main__":
    main()
