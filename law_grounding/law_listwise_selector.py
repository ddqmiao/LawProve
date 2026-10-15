"""Listwise law selector SFT data helpers."""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

from .law import article_key, normalize_law_id
from .law_rerank import _fact_context, _sample_meta
from .postprocess import extract_json_object
from .structured_law import format_structured_law_group, group_candidate_ids_by_article


LAW_LISTWISE_SYSTEM_PROMPT = (
    "你是法律法条候选组筛选模型。请比较同一案件中的多个候选法条，"
    "判断哪些法条应进入后续精排。必须精确区分整条与具体款项，"
    "例如第67条和第67条第3款不是同一个ID。只输出JSON，不要解释。"
)


def _same_article(candidate_id: str, gold_laws: list[str], law_map: dict[str, dict[str, Any]]) -> bool:
    candidate_article = article_key(candidate_id, law_map)
    return candidate_article in {article_key(str(gold), law_map) for gold in gold_laws}


def _exact_match(candidate_id: str, gold_laws: list[str]) -> bool:
    normalized = normalize_law_id(candidate_id)
    return normalized in {normalize_law_id(str(gold)) for gold in gold_laws}


def chunk_candidates(sample: dict[str, Any], *, top_k: int, group_size: int) -> list[list[dict[str, Any]]]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    candidates = list(sample.get("law_candidates", []) or [])[:top_k]
    return [candidates[start : start + group_size] for start in range(0, len(candidates), group_size)]


def label_candidate_ids(
    candidate_ids: list[str],
    gold_laws: list[str],
    law_map: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    labels = {"keep": [], "maybe": [], "reject": []}
    for law_id in candidate_ids:
        if _exact_match(law_id, gold_laws):
            labels["keep"].append(law_id)
        elif gold_laws and _same_article(law_id, gold_laws, law_map):
            labels["maybe"].append(law_id)
        else:
            labels["reject"].append(law_id)
    return labels


def build_listwise_selector_user_prompt(
    sample: dict[str, Any],
    candidate_ids: list[str],
    law_map: dict[str, dict[str, Any]],
    *,
    max_fact_chars: int = 3000,
) -> str:
    facts = _fact_context(sample, max_fact_chars=max_fact_chars)
    law_sections = [
        f"候选法条组{i + 1}：\n{format_structured_law_group(group)}"
        for i, group in enumerate(group_candidate_ids_by_article(candidate_ids, law_map))
    ]
    return (
        f'被告人：{sample["defendant"]}\n\n'
        f'裁判观点：{sample["court_view"]}\n\n'
        "候选事实：\n"
        + facts
        + "\n\n候选法条：\n"
        + "\n\n".join(law_sections)
        + "\n\n请将可选ID分为keep、maybe、reject三类。"
        + "keep表示直接支撑当前裁判观点；maybe表示同条下需要进入后续精排比较的竞争项；"
        + "reject表示当前组内应剔除。只能从可选ID中输出，不得输出候选外ID。"
        + '只输出JSON：{"keep":[],"maybe":[],"reject":[]}。'
    )


def build_listwise_selector_record(
    sample: dict[str, Any],
    candidate_ids: list[str],
    law_map: dict[str, dict[str, Any]],
    *,
    group_index: int,
    max_fact_chars: int = 3000,
) -> dict[str, Any]:
    gold_laws = [str(item) for item in sample.get("gold_laws", [])]
    labels = label_candidate_ids(candidate_ids, gold_laws, law_map)
    return {
        "messages": [
            {"role": "system", "content": LAW_LISTWISE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_listwise_selector_user_prompt(
                    sample,
                    candidate_ids,
                    law_map,
                    max_fact_chars=max_fact_chars,
                ),
            },
            {"role": "assistant", "content": json.dumps(labels, ensure_ascii=False, separators=(",", ":"))},
        ],
        "meta": {
            **_sample_meta(sample),
            "group_index": group_index,
            "candidate_law_ids": candidate_ids,
        },
    }


def build_listwise_selector_records(
    samples: list[dict[str, Any]],
    law_map: dict[str, dict[str, Any]],
    *,
    top_k: int = 50,
    group_size: int = 10,
    max_fact_chars: int = 3000,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in samples:
        for group_index, group in enumerate(chunk_candidates(sample, top_k=top_k, group_size=group_size)):
            candidate_ids = [str(candidate["id"]) for candidate in group]
            if not candidate_ids:
                continue
            records.append(
                build_listwise_selector_record(
                    sample,
                    candidate_ids,
                    law_map,
                    group_index=group_index,
                    max_fact_chars=max_fact_chars,
                )
            )
    return records


def _list_from_json_value(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _filter_to_candidates(items: list[str], candidate_ids: list[str], seen: set[str]) -> list[str]:
    candidate_set = set(candidate_ids)
    filtered: list[str] = []
    for item in items:
        if item not in candidate_set or item in seen:
            continue
        seen.add(item)
        filtered.append(item)
    return filtered


def parse_listwise_selector_output(raw_output: str, candidate_ids: list[str]) -> dict[str, Any]:
    """Parse keep/maybe/reject JSON and constrain IDs to the current group."""
    obj = extract_json_object(raw_output)
    if obj is None:
        return {"keep": [], "maybe": [], "reject": [], "valid_json": False}
    seen: set[str] = set()
    keep = _filter_to_candidates(_list_from_json_value(obj.get("keep")), candidate_ids, seen)
    maybe = _filter_to_candidates(_list_from_json_value(obj.get("maybe")), candidate_ids, seen)
    reject = _filter_to_candidates(_list_from_json_value(obj.get("reject")), candidate_ids, seen)
    return {"keep": keep, "maybe": maybe, "reject": reject, "valid_json": True}


def _aggregate_key(meta: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(meta.get("case_id")),
        str(meta.get("defendant_index")),
        str(meta.get("defendant")),
        str(meta.get("court_unit_id")),
    )


def _append_unique(output: list[str], seen: set[str], items: list[str], max_candidates: int | None) -> None:
    for item in items:
        if item in seen:
            continue
        if max_candidates is not None and len(output) >= max_candidates:
            return
        seen.add(item)
        output.append(item)


def aggregate_listwise_predictions(
    predictions: list[dict[str, Any]],
    *,
    include_maybe: bool = True,
    max_candidates: int | None = None,
    always_keep_top_n: int = 0,
) -> list[dict[str, Any]]:
    """Aggregate listwise group predictions into retrieval candidate rows."""
    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be >= 1 when set")
    if always_keep_top_n < 0:
        raise ValueError("always_keep_top_n must be >= 0")
    if max_candidates is not None and always_keep_top_n > max_candidates:
        raise ValueError("always_keep_top_n cannot exceed max_candidates")
    grouped: OrderedDict[tuple[str, str, str, str], list[dict[str, Any]]] = OrderedDict()
    for row in predictions:
        grouped.setdefault(_aggregate_key(row.get("meta", {})), []).append(row)

    output_rows: list[dict[str, Any]] = []
    for _, rows in grouped.items():
        rows.sort(key=lambda row: int(row.get("meta", {}).get("group_index", 0)))
        first_meta = rows[0].get("meta", {})
        selected: list[str] = []
        seen: set[str] = set()
        valid_json_count = 0
        original_candidates = [
            str(item)
            for row in rows
            for item in row.get("meta", {}).get("candidate_law_ids", [])
        ]
        _append_unique(selected, seen, original_candidates[:always_keep_top_n], max_candidates)
        for row in rows:
            if row.get("valid_json"):
                valid_json_count += 1
            _append_unique(selected, seen, [str(item) for item in row.get("pred_keep", [])], max_candidates)
            if include_maybe:
                _append_unique(selected, seen, [str(item) for item in row.get("pred_maybe", [])], max_candidates)
        output_rows.append(
            {
                "case_id": first_meta.get("case_id"),
                "defendant": first_meta.get("defendant"),
                "defendant_index": first_meta.get("defendant_index"),
                "court_unit_id": first_meta.get("court_unit_id"),
                "gold_laws": [str(item) for item in first_meta.get("gold_laws", [])],
                "retrieved_law_ids": selected,
                "group_prediction_count": len(rows),
                "valid_json_rate": valid_json_count / len(rows) if rows else 0.0,
                "valid_json": valid_json_count == len(rows) if rows else False,
            }
        )
    return output_rows


def candidate_pool_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute candidate pool recall ceilings against gold_laws."""
    total_gold = 0
    hit_gold = 0
    full_covered = 0
    any_covered = 0
    total_candidates = 0
    evaluable_count = 0
    for row in rows:
        gold = {str(item) for item in row.get("gold_laws", [])}
        candidates = {str(item) for item in row.get("retrieved_law_ids", [])}
        total_candidates += len(candidates)
        if not gold:
            continue
        evaluable_count += 1
        total_gold += len(gold)
        hit_gold += len(gold & candidates)
        full_covered += gold <= candidates
        any_covered += bool(gold & candidates)
    count = len(rows)
    return {
        "num_samples": count,
        "num_evaluable_samples": evaluable_count,
        "num_empty_gold_samples": count - evaluable_count,
        "avg_candidates": total_candidates / count if count else 0.0,
        "law_micro_recall_ceiling": hit_gold / total_gold if total_gold else None,
        "full_gold_covered_rate": full_covered / evaluable_count if evaluable_count else None,
        "any_gold_covered_rate": any_covered / evaluable_count if evaluable_count else None,
    }
