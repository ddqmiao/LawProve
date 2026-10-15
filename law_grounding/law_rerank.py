"""Candidate-level law reranking data and prediction helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Iterable

from .law import article_key, normalize_law_id
from .postprocess import extract_json_object


LAW_RERANK_SYSTEM_PROMPT = (
    "你是法律法条候选筛选模型。请判断单个候选法条是否直接支撑当前裁判观点。"
    "必须精确区分整条与具体款项，例如第25条和第25条第1款不是同一个ID；"
    "必须精确区分同一条下不同款项，例如第67条第1款和第67条第3款不同。"
    "只输出JSON，不要解释。"
)


def _sample_meta(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": sample["case_id"],
        "defendant": sample["defendant"],
        "defendant_index": sample["defendant_index"],
        "court_unit_id": sample["court_unit_id"],
        "gold_laws": [str(item) for item in sample.get("gold_laws", [])],
        "gold_facts": [str(item) for item in sample.get("gold_facts", [])],
    }


def _law_article_key(law_id: str, law_map: dict[str, dict[str, Any]] | None) -> str:
    return article_key(normalize_law_id(law_id), law_map)


def _candidate_is_positive(candidate_id: str, gold_laws: Iterable[str]) -> bool:
    normalized = normalize_law_id(candidate_id)
    return normalized in {normalize_law_id(item) for item in gold_laws}


def _candidate_negative_type(
    candidate_id: str,
    gold_laws: list[str],
    law_map: dict[str, dict[str, Any]] | None,
) -> str:
    if not gold_laws:
        return "empty_gold"
    candidate_article = _law_article_key(candidate_id, law_map)
    gold_articles = {_law_article_key(item, law_map) for item in gold_laws}
    return "same_article" if candidate_article in gold_articles else "top_ranked"


def _fact_context(sample: dict[str, Any], *, max_fact_chars: int) -> str:
    lines: list[str] = []
    total = 0
    for fact in sample.get("fact_candidates", []) or []:
        line = f'{fact["id"]}: {fact["text"]}'
        if max_fact_chars > 0 and total + len(line) > max_fact_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def build_law_rerank_user_prompt(
    sample: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_fact_chars: int = 3000,
) -> str:
    """Build a prompt for judging one candidate law."""
    facts = _fact_context(sample, max_fact_chars=max_fact_chars)
    return (
        f'被告人：{sample["defendant"]}\n\n'
        f'裁判观点：{sample["court_view"]}\n\n'
        "候选事实：\n"
        + facts
        + "\n\n候选法条ID："
        + str(candidate["id"])
        + "\n候选法条内容：\n"
        + str(candidate.get("text") or "")
        + "\n\n请判断该候选法条是否应被选为support_laws。"
        + '只输出JSON：{"relevant":true}或{"relevant":false}。'
    )


def build_law_rerank_sft_record(
    sample: dict[str, Any],
    candidate: dict[str, Any],
    *,
    candidate_rank: int,
    relevant: bool,
    negative_type: str = "",
    max_fact_chars: int = 3000,
) -> dict[str, Any]:
    """Build one SFT message record for a sample-candidate pair."""
    return {
        "messages": [
            {"role": "system", "content": LAW_RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": build_law_rerank_user_prompt(sample, candidate, max_fact_chars=max_fact_chars)},
            {
                "role": "assistant",
                "content": json.dumps({"relevant": bool(relevant)}, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "meta": {
            **_sample_meta(sample),
            "candidate_law_id": str(candidate["id"]),
            "candidate_rank": candidate_rank,
            "label": bool(relevant),
            "negative_type": "" if relevant else negative_type,
        },
    }


def _select_training_candidate_indices(
    sample: dict[str, Any],
    law_map: dict[str, dict[str, Any]] | None,
    *,
    top_k: int,
    negatives_per_positive: int,
    hard_negatives_per_positive: int,
    negatives_per_empty_sample: int,
) -> list[int]:
    candidates = list(sample.get("law_candidates", []) or [])[:top_k]
    gold_laws = [str(item) for item in sample.get("gold_laws", [])]
    positive_indices: list[int] = []
    hard_negative_indices: list[int] = []
    top_negative_indices: list[int] = []

    for index, candidate in enumerate(candidates):
        candidate_id = str(candidate["id"])
        if _candidate_is_positive(candidate_id, gold_laws):
            positive_indices.append(index)
        elif _candidate_negative_type(candidate_id, gold_laws, law_map) == "same_article":
            hard_negative_indices.append(index)
        else:
            top_negative_indices.append(index)

    if not positive_indices:
        return top_negative_indices[:negatives_per_empty_sample]

    hard_limit = max(0, hard_negatives_per_positive) * len(positive_indices)
    top_limit = max(0, negatives_per_positive) * len(positive_indices)
    selected = [
        *positive_indices,
        *hard_negative_indices[:hard_limit],
        *top_negative_indices[:top_limit],
    ]
    return sorted(dict.fromkeys(selected))


def build_law_rerank_records(
    samples: list[dict[str, Any]],
    law_map: dict[str, dict[str, Any]] | None = None,
    *,
    top_k: int = 50,
    include_all_candidates: bool = False,
    negatives_per_positive: int = 4,
    hard_negatives_per_positive: int = 4,
    negatives_per_empty_sample: int = 3,
    max_fact_chars: int = 3000,
) -> list[dict[str, Any]]:
    """Build candidate-level law reranking SFT records.

    Training mode samples positives, same-article hard negatives, and top-ranked negatives.
    Evaluation mode (`include_all_candidates=True`) emits every candidate up to `top_k`.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    records: list[dict[str, Any]] = []
    for sample in samples:
        candidates = list(sample.get("law_candidates", []) or [])[:top_k]
        if include_all_candidates:
            selected_indices = list(range(len(candidates)))
        else:
            selected_indices = _select_training_candidate_indices(
                sample,
                law_map,
                top_k=top_k,
                negatives_per_positive=negatives_per_positive,
                hard_negatives_per_positive=hard_negatives_per_positive,
                negatives_per_empty_sample=negatives_per_empty_sample,
            )

        gold_laws = [str(item) for item in sample.get("gold_laws", [])]
        for index in selected_indices:
            candidate = candidates[index]
            candidate_id = str(candidate["id"])
            relevant = _candidate_is_positive(candidate_id, gold_laws)
            records.append(
                build_law_rerank_sft_record(
                    sample,
                    candidate,
                    candidate_rank=index + 1,
                    relevant=relevant,
                    negative_type=_candidate_negative_type(candidate_id, gold_laws, law_map),
                    max_fact_chars=max_fact_chars,
                )
            )
    return records


def parse_law_rerank_relevance(raw_output: str) -> dict[str, Any]:
    """Parse a reranker output into a boolean relevance decision."""
    obj = extract_json_object(raw_output)
    if obj is None:
        return {"relevant": False, "valid_json": False}
    value = obj.get("relevant")
    if isinstance(value, bool):
        relevant = value
    elif isinstance(value, str):
        relevant = value.strip().lower() in {"true", "yes", "y", "1", "是", "相关"}
    elif isinstance(value, (int, float)):
        relevant = bool(value)
    else:
        relevant = False
    return {"relevant": relevant, "valid_json": True}


def _aggregate_key(meta: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(meta.get("case_id")),
        str(meta.get("defendant_index")),
        str(meta.get("defendant")),
        str(meta.get("court_unit_id")),
    )


def aggregate_law_rerank_predictions(
    candidate_predictions: list[dict[str, Any]],
    *,
    rank_limit: int | None = 10,
    max_selected: int | None = None,
) -> list[dict[str, Any]]:
    """Aggregate candidate-level reranker predictions into grounding-style rows."""
    if rank_limit is not None and rank_limit < 1:
        raise ValueError("rank_limit must be >= 1 when set")
    if max_selected is not None and max_selected < 1:
        raise ValueError("max_selected must be >= 1 when set")
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_predictions:
        grouped[_aggregate_key(row.get("meta", {}))].append(row)

    output_rows: list[dict[str, Any]] = []
    for _, rows in grouped.items():
        rows.sort(key=lambda row: int(row.get("meta", {}).get("candidate_rank", 0)))
        first_meta = rows[0]["meta"]
        selected: list[str] = []
        seen: set[str] = set()
        valid_json_count = 0
        for row in rows:
            meta = row.get("meta", {})
            if row.get("valid_json"):
                valid_json_count += 1
            rank = int(meta.get("candidate_rank", 0))
            if rank_limit is not None and rank > rank_limit:
                continue
            if row.get("pred_relevant") is not True:
                continue
            if max_selected is not None and len(selected) >= max_selected:
                continue
            law_id = str(meta.get("candidate_law_id"))
            if law_id and law_id not in seen:
                seen.add(law_id)
                selected.append(law_id)

        output_rows.append(
            {
                "case_id": first_meta.get("case_id"),
                "defendant": first_meta.get("defendant"),
                "defendant_index": first_meta.get("defendant_index"),
                "court_unit_id": first_meta.get("court_unit_id"),
                "gold": {
                    "support_facts": list(first_meta.get("gold_facts") or []),
                    "support_laws": list(first_meta.get("gold_laws") or []),
                },
                "pred": {"support_facts": [], "support_laws": selected},
                "candidate_prediction_count": len(rows),
                "valid_json_rate": valid_json_count / len(rows) if rows else 0.0,
                "valid_json": valid_json_count == len(rows) if rows else False,
            }
        )
    return output_rows
