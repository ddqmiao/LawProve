"""Data preparation for defendant-level court-view grounding."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .law import format_law_text, is_paragraph_law_id, lookup_law_info, normalize_law_id


SYSTEM_PROMPT = (
    "你是法律裁判观点 grounding 模型。请根据被告人、裁判观点、候选事实和候选法条，"
    "选择能够支撑该裁判观点的事实ID和法条ID。只能从候选列表中选择ID，不得生成候选外ID。"
    "只输出JSON，不要解释。"
)
STRICT_FACT_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + " 事实选择时采用最小必要事实集合，只选择直接支撑当前裁判观点结论的事实。"
    "不要选择背景事实、量刑情节、其他被告人的事实或重复表达的事实。"
)
PROMPT_VARIANTS = ("default", "strict_fact")

COMMON_LAW_IDS = ["25", "27", "52", "53", "61", "64", "65", "67", "72", "73"]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_law_map(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def collect_case_link_laws(case: dict[str, Any]) -> list[str]:
    law_ids: list[str] = []
    for defendant in case.get("defendants") or []:
        for link in defendant.get("grounding_links") or []:
            law_ids.extend(str(item) for item in (link.get("law_ids") or []))
    return law_ids


def collect_outcome_articles(case: dict[str, Any]) -> list[str]:
    law_ids: list[str] = []
    for outcome in case.get("outcomes") or []:
        for judgment in outcome.get("judgment") or []:
            article = judgment.get("article") or []
            if isinstance(article, str):
                law_ids.append(article)
            else:
                law_ids.extend(str(item) for item in article)
    return law_ids


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def build_law_candidate_ids(
    case: dict[str, Any],
    law_map: dict[str, dict[str, Any]],
    *,
    max_law_candidates: int = 30,
) -> list[str]:
    """Build law candidates from case-level relevant articles only."""
    del law_map
    candidate_ids = _dedupe_preserve_order(
        [str(item) for item in (case.get("relevant_articles") or []) if str(item)]
    )
    return candidate_ids[:max_law_candidates]


def align_law_ids_to_candidates(law_ids: list[str], candidate_ids: list[str]) -> list[str]:
    """Map gold law IDs to candidate IDs when they differ only by normalization."""
    candidate_set = set(candidate_ids)
    normalized_to_candidate: dict[str, str] = {}
    for candidate_id in candidate_ids:
        normalized_to_candidate.setdefault(normalize_law_id(candidate_id), candidate_id)

    aligned: list[str] = []
    for law_id in law_ids:
        law_id = str(law_id)
        if law_id in candidate_set:
            aligned.append(law_id)
        else:
            aligned.append(normalized_to_candidate.get(normalize_law_id(law_id), law_id))
    return aligned


def sample_has_candidate_coverage(sample: dict[str, Any]) -> bool:
    candidate_ids = {str(item["id"]) for item in sample.get("law_candidates", [])}
    return all(str(law_id) in candidate_ids for law_id in sample.get("gold_laws", []))


def _key_part(value: Any) -> str:
    return "" if value is None else str(value)


def sample_retrieval_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    """Return the stable key shared by prepared samples and retrieval rows."""
    return (
        _key_part(row.get("case_id")),
        _key_part(row.get("defendant_index")),
        _key_part(row.get("defendant")),
        _key_part(row.get("court_unit_id")),
    )


def _retrieved_law_items(row: dict[str, Any]) -> list[dict[str, str]]:
    retrieved_laws = row.get("retrieved_laws") or []
    if retrieved_laws:
        return [
            {"id": str(item.get("id")), "text": str(item.get("text") or "")}
            for item in retrieved_laws
            if str(item.get("id") or "")
        ]
    return [
        {"id": str(law_id), "text": ""}
        for law_id in row.get("retrieved_law_ids") or []
        if str(law_id)
    ]


def _format_candidate_text(law_id: str, retrieved_text: str, law_map: dict[str, dict[str, Any]] | None) -> str:
    if retrieved_text:
        return retrieved_text
    if law_map is not None:
        return format_law_text(law_id, law_map)
    return f"{law_id}: 法条ID：{law_id}"


def _build_retrieval_candidate_index(
    retrieval_rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str], list[dict[str, str]]]:
    index: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in retrieval_rows:
        key = sample_retrieval_key(row)
        if key in index:
            raise ValueError(f"duplicate retrieval candidate row for sample key: {key}")
        index[key] = _retrieved_law_items(row)
    return index


def _refresh_law_candidate_warnings(
    sample: dict[str, Any],
    law_map: dict[str, dict[str, Any]] | None,
) -> list[dict[str, str]]:
    warnings = [
        dict(warning)
        for warning in sample.get("warnings", [])
        if warning.get("type") not in {"missing_gold_law_candidate", "missing_law_text"}
    ]
    candidate_ids = [str(item["id"]) for item in sample.get("law_candidates", [])]
    candidate_set = set(candidate_ids)
    original_gold_laws = [str(item) for item in sample.get("original_gold_laws") or sample.get("gold_laws") or []]
    gold_laws = align_law_ids_to_candidates(original_gold_laws, candidate_ids)
    for original_law_id, law_id in zip(original_gold_laws, gold_laws, strict=True):
        if law_id not in candidate_set:
            warnings.append({"type": "missing_gold_law_candidate", "value": original_law_id})
        if law_map is not None and not lookup_law_info(law_id, law_map):
            warnings.append({"type": "missing_law_text", "value": law_id})
    sample["gold_laws"] = gold_laws
    return warnings


def apply_retrieval_candidates_to_samples(
    samples: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    law_map: dict[str, dict[str, Any]] | None = None,
    *,
    top_k: int = 50,
    require_all: bool = True,
) -> list[dict[str, Any]]:
    """Replace each sample's law candidates with retrieved candidates by sample key."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    index = _build_retrieval_candidate_index(retrieval_rows)
    updated_samples: list[dict[str, Any]] = []
    for sample in samples:
        key = sample_retrieval_key(sample)
        if key not in index:
            if require_all:
                raise ValueError(f"missing retrieval candidates for sample key: {key}")
            updated_samples.append(dict(sample))
            continue

        seen: set[str] = set()
        candidate_items: list[dict[str, str]] = []
        for item in index[key]:
            law_id = str(item["id"])
            if law_id in seen:
                continue
            seen.add(law_id)
            candidate_items.append(
                {
                    "id": law_id,
                    "text": _format_candidate_text(law_id, item.get("text", ""), law_map),
                }
            )
            if len(candidate_items) >= top_k:
                break

        updated = dict(sample)
        updated["law_candidates"] = candidate_items
        updated["law_candidate_source"] = "retrieval"
        updated["retrieval_candidate_top_k"] = top_k
        updated["warnings"] = _refresh_law_candidate_warnings(updated, law_map)
        updated_samples.append(updated)
    return updated_samples


def _court_view_map(case: dict[str, Any]) -> dict[str, str]:
    return {str(item.get("id")): str(item.get("text") or "") for item in (case.get("court_view") or [])}


def _fact_candidates(case: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"id": str(item.get("id")), "text": str(item.get("text") or "")}
        for item in (case.get("fact") or [])
    ]


def build_sample(
    case: dict[str, Any],
    defendant: dict[str, Any],
    link: dict[str, Any],
    law_map: dict[str, dict[str, Any]],
    *,
    max_law_candidates: int = 30,
) -> dict[str, Any]:
    warnings: list[dict[str, str]] = []
    facts = _fact_candidates(case)
    fact_ids = {item["id"] for item in facts}
    court_views = _court_view_map(case)

    court_unit_id = str(link.get("court_unit_id"))
    if court_unit_id not in court_views:
        warnings.append({"type": "missing_court_view", "value": court_unit_id})

    gold_facts = [str(item) for item in (link.get("fact_ids") or [])]
    original_gold_laws = [str(item) for item in (link.get("law_ids") or [])]
    for fact_id in gold_facts:
        if fact_id not in fact_ids:
            warnings.append({"type": "missing_gold_fact", "value": fact_id})

    law_candidate_ids = build_law_candidate_ids(
        case,
        law_map,
        max_law_candidates=max_law_candidates,
    )
    gold_laws = align_law_ids_to_candidates(original_gold_laws, law_candidate_ids)
    law_candidate_set = set(law_candidate_ids)
    for original_law_id, law_id in zip(original_gold_laws, gold_laws, strict=True):
        if law_id not in law_candidate_set:
            warnings.append({"type": "missing_gold_law_candidate", "value": original_law_id})
    for law_id in gold_laws:
        if not lookup_law_info(law_id, law_map):
            warnings.append({"type": "missing_law_text", "value": law_id})

    return {
        "case_id": str(case.get("case_id")),
        "defendant": str(defendant.get("name") or ""),
        "defendant_index": defendant.get("defendant_index"),
        "court_unit_id": court_unit_id,
        "court_view": court_views.get(court_unit_id, ""),
        "fact_candidates": facts,
        "law_candidates": [
            {"id": law_id, "text": format_law_text(law_id, law_map)} for law_id in law_candidate_ids
        ],
        "gold_facts": gold_facts,
        "gold_laws": gold_laws,
        "original_gold_laws": original_gold_laws,
        "warnings": warnings,
    }


def expand_case(
    case: dict[str, Any],
    law_map: dict[str, dict[str, Any]],
    *,
    max_law_candidates: int = 30,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for defendant in case.get("defendants") or []:
        for link in defendant.get("grounding_links") or []:
            sample = build_sample(
                case,
                defendant,
                link,
                law_map,
                max_law_candidates=max_law_candidates,
            )
            samples.append(sample)
            for warning in sample["warnings"]:
                warnings.append(
                    {
                        "case_id": sample["case_id"],
                        "defendant": sample["defendant"],
                        "court_unit_id": sample["court_unit_id"],
                        **warning,
                    }
                )
    return samples, warnings


def _sample_warnings(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for sample in samples:
        for warning in sample.get("warnings", []):
            warnings.append(
                {
                    "case_id": sample["case_id"],
                    "defendant": sample["defendant"],
                    "court_unit_id": sample["court_unit_id"],
                    **warning,
                }
            )
    return warnings


def _validate_prompt_variant(prompt_variant: str) -> str:
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"unsupported prompt_variant: {prompt_variant}")
    return prompt_variant


def build_user_prompt(sample: dict[str, Any], *, prompt_variant: str = "default") -> str:
    prompt_variant = _validate_prompt_variant(prompt_variant)
    fact_lines = [f'{item["id"]}: {item["text"]}' for item in sample["fact_candidates"]]
    law_lines = [item["text"] for item in sample["law_candidates"]]
    strict_fact_instruction = ""
    if prompt_variant == "strict_fact":
        strict_fact_instruction = (
            "\n\n事实选择要求：只输出支撑该裁判观点所必需的最少事实ID；"
            "宁可少选，不要多选；不要因为事实提到同一被告人就选择它。"
        )
    return (
        f'被告人：{sample["defendant"]}\n\n'
        f'裁判观点：{sample["court_view"]}\n\n'
        "候选事实：\n"
        + "\n".join(fact_lines)
        + "\n\n候选法条：\n"
        + "\n".join(law_lines)
        + strict_fact_instruction
        + "\n\n请输出JSON，字段为support_facts和support_laws。"
    )


def build_inference_messages(sample: dict[str, Any], *, prompt_variant: str = "default") -> list[dict[str, str]]:
    prompt_variant = _validate_prompt_variant(prompt_variant)
    system_prompt = STRICT_FACT_SYSTEM_PROMPT if prompt_variant == "strict_fact" else SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(sample, prompt_variant=prompt_variant)},
    ]


def build_sft_record(sample: dict[str, Any], *, prompt_variant: str = "default") -> dict[str, Any]:
    answer = {
        "support_facts": sample["gold_facts"],
        "support_laws": sample["gold_laws"],
    }
    return {
        "messages": [
            *build_inference_messages(sample, prompt_variant=prompt_variant),
            {
                "role": "assistant",
                "content": json.dumps(answer, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "meta": {
            "case_id": sample["case_id"],
            "defendant": sample["defendant"],
            "defendant_index": sample["defendant_index"],
            "court_unit_id": sample["court_unit_id"],
        },
    }


def split_name_from_path(path: Path) -> str:
    return path.name[:-6] if path.name.endswith(".jsonl") else path.stem


def prepare_splits(
    data_dir: str | Path,
    law_map_path: str | Path,
    output_dir: str | Path,
    *,
    max_law_candidates: int = 30,
    prompt_variant: str = "default",
    retrieval_candidates_dir: str | Path | None = None,
    retrieval_candidates_suffix: str = ".top100.jsonl",
    retrieval_top_k: int = 50,
    filter_missing_law_candidates: bool = True,
) -> dict[str, Any]:
    prompt_variant = _validate_prompt_variant(prompt_variant)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    retrieval_candidates_dir = Path(retrieval_candidates_dir) if retrieval_candidates_dir else None
    law_map = load_law_map(law_map_path)
    all_stats: dict[str, Any] = {
        "splits": {},
        "warnings": [],
    }
    all_samples: list[dict[str, Any]] = []

    for path in sorted(data_dir.glob("*.jsonl")):
        split = split_name_from_path(path)
        cases = read_jsonl(path)
        samples: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        defendant_count = 0
        link_count = 0
        fact_count = 0
        court_view_count = 0
        for case in cases:
            defendant_count += len(case.get("defendants") or [])
            fact_count += len(case.get("fact") or [])
            court_view_count += len(case.get("court_view") or [])
            expanded, case_warnings = expand_case(case, law_map, max_law_candidates=max_law_candidates)
            samples.extend(expanded)
            link_count += len(expanded)

        if retrieval_candidates_dir is not None:
            retrieval_path = retrieval_candidates_dir / f"{split}{retrieval_candidates_suffix}"
            if not retrieval_path.exists():
                fallback_path = retrieval_candidates_dir / f"{split}.jsonl"
                retrieval_path = fallback_path if fallback_path.exists() else retrieval_path
            retrieval_rows = read_jsonl(retrieval_path)
            samples = apply_retrieval_candidates_to_samples(
                samples,
                retrieval_rows,
                law_map,
                top_k=retrieval_top_k,
            )

        warnings = _sample_warnings(samples)
        missing_candidate_count = sum(1 for sample in samples if not sample_has_candidate_coverage(sample))
        if filter_missing_law_candidates:
            samples = [sample for sample in samples if sample_has_candidate_coverage(sample)]

        sft_rows = [build_sft_record(sample, prompt_variant=prompt_variant) for sample in samples]
        write_jsonl(output_dir / "samples" / f"{split}.jsonl", samples)
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", sft_rows)
        write_jsonl(output_dir / "warnings" / f"{split}.jsonl", warnings)
        all_samples.extend(samples)
        all_stats["warnings"].extend(warnings)
        all_stats["splits"][split] = {
            "cases": len(cases),
            "defendants": defendant_count,
            "facts": fact_count,
            "court_views": court_view_count,
            "raw_samples": link_count,
            "samples": len(samples),
            "filtered_missing_law_candidates": missing_candidate_count if filter_missing_law_candidates else 0,
            "samples_missing_law_candidates": missing_candidate_count,
        }

    all_stats.update(compute_sample_stats(all_samples, law_map))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stats.json").write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return all_stats


def prepare_splits_from_prepared_samples(
    samples_dir: str | Path,
    law_map_path: str | Path,
    output_dir: str | Path,
    *,
    retrieval_candidates_dir: str | Path,
    retrieval_candidates_suffix: str = ".top100.jsonl",
    retrieval_top_k: int = 50,
    prompt_variant: str = "default",
    filter_missing_law_candidates: bool = True,
    filter_missing_law_candidate_splits: set[str] | None = None,
) -> dict[str, Any]:
    """Rebuild prepared samples and SFT rows after replacing law candidates from retrieval files."""
    prompt_variant = _validate_prompt_variant(prompt_variant)
    samples_dir = Path(samples_dir)
    output_dir = Path(output_dir)
    retrieval_candidates_dir = Path(retrieval_candidates_dir)
    law_map = load_law_map(law_map_path)
    all_samples: list[dict[str, Any]] = []
    all_stats: dict[str, Any] = {"splits": {}, "warnings": []}

    for path in sorted(samples_dir.glob("*.jsonl")):
        split = split_name_from_path(path)
        samples = read_jsonl(path)
        retrieval_path = retrieval_candidates_dir / f"{split}{retrieval_candidates_suffix}"
        if not retrieval_path.exists():
            fallback_path = retrieval_candidates_dir / f"{split}.jsonl"
            retrieval_path = fallback_path if fallback_path.exists() else retrieval_path
        retrieval_rows = read_jsonl(retrieval_path)
        samples = apply_retrieval_candidates_to_samples(
            samples,
            retrieval_rows,
            law_map,
            top_k=retrieval_top_k,
        )
        warnings = _sample_warnings(samples)
        missing_candidate_count = sum(1 for sample in samples if not sample_has_candidate_coverage(sample))
        raw_sample_count = len(samples)
        should_filter_missing = filter_missing_law_candidates or (
            filter_missing_law_candidate_splits is not None and split in filter_missing_law_candidate_splits
        )
        if should_filter_missing:
            samples = [sample for sample in samples if sample_has_candidate_coverage(sample)]

        sft_rows = [build_sft_record(sample, prompt_variant=prompt_variant) for sample in samples]
        write_jsonl(output_dir / "samples" / f"{split}.jsonl", samples)
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", sft_rows)
        write_jsonl(output_dir / "warnings" / f"{split}.jsonl", warnings)
        all_samples.extend(samples)
        all_stats["warnings"].extend(warnings)
        all_stats["splits"][split] = {
            "raw_samples": raw_sample_count,
            "samples": len(samples),
            "filtered_missing_law_candidates": missing_candidate_count if should_filter_missing else 0,
            "samples_missing_law_candidates": missing_candidate_count,
        }

    all_stats.update(compute_sample_stats(all_samples, law_map))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stats.json").write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return all_stats


def _avg(values: list[int]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def compute_sample_stats(samples: list[dict[str, Any]], law_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gold_law_counter: Counter[str] = Counter()
    unmapped_law_counter: Counter[str] = Counter()
    paragraph_law_occurrences = 0
    article_law_occurrences = 0
    for sample in samples:
        for law_id in sample["gold_laws"]:
            gold_law_counter[law_id] += 1
            if not lookup_law_info(law_id, law_map):
                unmapped_law_counter[law_id] += 1
            if is_paragraph_law_id(law_id, law_map):
                paragraph_law_occurrences += 1
            else:
                article_law_occurrences += 1

    gold_fact_counts = [len(sample["gold_facts"]) for sample in samples]
    gold_law_counts = [len(sample["gold_laws"]) for sample in samples]
    fact_candidate_counts = [len(sample["fact_candidates"]) for sample in samples]
    law_candidate_counts = [len(sample["law_candidates"]) for sample in samples]
    mapped_gold = sum(count for law_id, count in gold_law_counter.items() if lookup_law_info(law_id, law_map))
    gold_total = sum(gold_law_counter.values())

    return {
        "total_samples": len(samples),
        "avg_fact_candidates": _avg(fact_candidate_counts),
        "avg_law_candidates": _avg(law_candidate_counts),
        "avg_gold_facts": _avg(gold_fact_counts),
        "avg_gold_laws": _avg(gold_law_counts),
        "empty_gold_facts_rate": _avg([1 if count == 0 else 0 for count in gold_fact_counts]),
        "empty_gold_laws_rate": _avg([1 if count == 0 else 0 for count in gold_law_counts]),
        "gold_law_text_coverage": round(mapped_gold / gold_total, 6) if gold_total else 1.0,
        "unmapped_gold_laws": unmapped_law_counter.most_common(),
        "top_gold_laws": gold_law_counter.most_common(30),
        "paragraph_law_occurrences": paragraph_law_occurrences,
        "article_law_occurrences": article_law_occurrences,
    }
