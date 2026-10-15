"""Shared listwise-ranking records for the joint grounding Causal LM.

The ranking task deliberately reuses the same chat model and LM head as the
grounding generator.  Each record presents a *set* of provisions and trains on
their identifier-token logits at the first assistant position.  Article
families are auxiliary supervision only; they never alter the candidate pool.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from .law import article_key, normalize_law_id
from .law_listwise_selector import chunk_candidates
from .law_rerank import _fact_context, _sample_meta
from .structured_law import group_candidate_ids_by_article


CANDIDATE_MARKERS = tuple("ABCDEFGHIJ")
NULL_MARKER = "N"
JOINT_RANK_SYSTEM_PROMPT = (
    "你是法律依据组内排序模型。请比较当前组的全部候选刑法依据，判断哪些候选直接支撑当前被告人的裁判观点。"
    "必须精确区分整条与具体款项，也必须区分同一条下的不同款项。"
    "下一步只需对候选标记进行相关性排序；不得利用候选组之外的法条。"
)


def _stable_int(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _normalized_set(values: list[str]) -> set[str]:
    return {normalize_law_id(str(value)) for value in values}


def _candidate_order(sample: dict[str, Any], group_index: int, size: int, seed: int) -> list[int]:
    """Return a deterministic rotation/reversal to reduce marker-token bias."""
    if size < 1:
        return []
    value = _stable_int(
        seed,
        sample.get("case_id"),
        sample.get("defendant_index"),
        sample.get("court_unit_id"),
        group_index,
    )
    order = list(range(size))
    shift = value % size
    order = order[shift:] + order[:shift]
    if (value // max(size, 1)) % 2:
        order.reverse()
    return order


def _format_marked_candidates(
    candidate_ids: list[str],
    markers: list[str],
    law_map: dict[str, dict[str, Any]],
) -> str:
    marker_by_id = dict(zip(candidate_ids, markers, strict=True))
    sections: list[str] = []
    for group in group_candidate_ids_by_article(candidate_ids, law_map):
        lines = [
            f"所属条文：{group.get('article_label') or group.get('article_id')}"
            f"（article_id={group.get('article_id')}）",
            f"完整条文：{group.get('article_text') or '未提供完整条文。'}",
            "候选项：",
        ]
        for item in group.get("selectable_items", []):
            law_id = str(item["id"])
            paragraph_no = item.get("paragraph_no")
            suffix = f"，第{paragraph_no}款" if paragraph_no is not None else ""
            lines.append(
                f"- {marker_by_id[law_id]} = {law_id} "
                f"[{item.get('type') or ''}{suffix}]：{item.get('text') or ''}"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def build_joint_rank_user_prompt(
    sample: dict[str, Any],
    candidate_ids: list[str],
    markers: list[str],
    law_map: dict[str, dict[str, Any]],
    *,
    max_fact_chars: int = 3000,
) -> str:
    if len(candidate_ids) != len(markers):
        raise ValueError("candidate_ids and markers must have the same length")
    facts = _fact_context(sample, max_fact_chars=max_fact_chars)
    candidates = _format_marked_candidates(candidate_ids, markers, law_map)
    return (
        f'被告人：{sample["defendant"]}\n\n'
        f'裁判观点：{sample["court_view"]}\n\n'
        "候选事实：\n"
        + facts
        + "\n\n候选法条：\n"
        + candidates
        + f"\n\n候选标记为{','.join(markers)}；{NULL_MARKER}表示本组没有任何直接相关法条。"
        + "请比较整个候选组。"
    )


def build_joint_rank_record(
    sample: dict[str, Any],
    candidates: list[dict[str, Any]],
    law_map: dict[str, dict[str, Any]],
    *,
    group_index: int,
    group_start_rank: int | None = None,
    max_fact_chars: int = 3000,
    seed: int = 42,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("ranking group cannot be empty")
    if len(candidates) > len(CANDIDATE_MARKERS):
        raise ValueError(f"ranking group cannot exceed {len(CANDIDATE_MARKERS)} candidates")

    order = _candidate_order(sample, group_index, len(candidates), seed)
    start_rank = group_start_rank if group_start_rank is not None else group_index * len(candidates) + 1
    source_ranks = [int(item.get("rank") or (start_rank + index)) for index, item in enumerate(candidates)]
    ordered = [candidates[index] for index in order]
    candidate_ids = [str(item["id"]) for item in ordered]
    markers = list(CANDIDATE_MARKERS[: len(candidate_ids)])
    gold_laws = [str(item) for item in sample.get("gold_laws", [])]
    normalized_gold = _normalized_set(gold_laws)
    gold_families = {article_key(item, law_map) for item in gold_laws}
    candidate_families = [article_key(item, law_map) for item in candidate_ids]
    family_to_index: dict[str, int] = {}
    family_indices: list[int] = []
    for family in candidate_families:
        family_to_index.setdefault(family, len(family_to_index))
        family_indices.append(family_to_index[family])

    exact_positive_indices = [
        index for index, law_id in enumerate(candidate_ids) if normalize_law_id(law_id) in normalized_gold
    ]
    gold_family_indices = sorted(
        {family_to_index[family] for family in gold_families if family in family_to_index}
    )
    target_is_null = not exact_positive_indices
    family_target_is_null = not gold_family_indices
    return {
        "task_type": "law_listwise_rank",
        "messages": [
            {"role": "system", "content": JOINT_RANK_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_joint_rank_user_prompt(
                    sample,
                    candidate_ids,
                    markers,
                    law_map,
                    max_fact_chars=max_fact_chars,
                ),
            },
        ],
        "ranking": {
            "candidate_markers": markers,
            "null_marker": NULL_MARKER,
            "exact_positive_indices": exact_positive_indices,
            "target_is_null": target_is_null,
            "candidate_family_indices": family_indices,
            "gold_family_indices": gold_family_indices,
            "family_target_is_null": family_target_is_null,
        },
        "meta": {
            **_sample_meta(sample),
            "group_index": group_index,
            "candidate_law_ids": candidate_ids,
            "candidate_article_ids": candidate_families,
            "original_candidate_ranks": [source_ranks[index] for index in order],
        },
    }


def build_all_joint_rank_records(
    samples: list[dict[str, Any]],
    law_map: dict[str, dict[str, Any]],
    *,
    top_k: int = 50,
    group_size: int = 10,
    max_fact_chars: int = 3000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if group_size > len(CANDIDATE_MARKERS):
        raise ValueError(f"group_size cannot exceed {len(CANDIDATE_MARKERS)}")
    records: list[dict[str, Any]] = []
    for sample in samples:
        groups = chunk_candidates(sample, top_k=top_k, group_size=group_size)
        for group_index, group in enumerate(groups):
            records.append(
                build_joint_rank_record(
                    sample,
                    group,
                    law_map,
                    group_index=group_index,
                    group_start_rank=group_index * group_size + 1,
                    max_fact_chars=max_fact_chars,
                    seed=seed,
                )
            )
    return records


def select_joint_rank_training_records(
    records: list[dict[str, Any]],
    *,
    max_positive_groups_per_unit: int = 1,
    max_negative_groups_per_unit: int = 1,
    negative_unit_fraction: float = 1.0,
    empty_groups_per_unit: int = 1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose at most one exact-positive and one hard-negative group per unit.

    Negative priority is: a group containing a gold Article sibling, then a
    lower retrieval group index.  Empty-law units contribute one explicit-NULL
    group.  Selection is deterministic and does not inspect model predictions.
    """
    if min(max_positive_groups_per_unit, max_negative_groups_per_unit, empty_groups_per_unit) < 0:
        raise ValueError("group limits must be non-negative")
    if not 0.0 <= negative_unit_fraction <= 1.0:
        raise ValueError("negative_unit_fraction must be between 0 and 1")
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        meta = record.get("meta", {})
        key = (
            str(meta.get("case_id")),
            str(meta.get("defendant_index")),
            str(meta.get("defendant")),
            str(meta.get("court_unit_id")),
        )
        grouped.setdefault(key, []).append(record)

    selected: list[dict[str, Any]] = []
    stats = Counter()
    for key, rows in grouped.items():
        rows.sort(key=lambda row: int(row["meta"]["group_index"]))
        gold_laws = rows[0].get("meta", {}).get("gold_laws") or []
        if not gold_laws:
            chosen = rows[:empty_groups_per_unit]
            selected.extend(chosen)
            stats["empty_units"] += 1
            stats["empty_records"] += len(chosen)
            continue

        positives = [row for row in rows if row["ranking"]["exact_positive_indices"]]
        negatives = [row for row in rows if not row["ranking"]["exact_positive_indices"]]
        positives.sort(
            key=lambda row: _stable_int(seed, *key, "positive", row["meta"]["group_index"])
        )
        negatives.sort(
            key=lambda row: (
                not bool(row["ranking"]["gold_family_indices"]),
                int(row["meta"]["group_index"]),
            )
        )
        chosen_positive = positives[:max_positive_groups_per_unit]
        negative_bucket = _stable_int(seed, *key, "negative_unit") % 1_000_000
        include_negative = negative_bucket < round(negative_unit_fraction * 1_000_000)
        chosen_negative = negatives[:max_negative_groups_per_unit] if include_negative else []
        selected.extend(chosen_positive)
        selected.extend(chosen_negative)
        stats["nonempty_units"] += 1
        stats["positive_records"] += len(chosen_positive)
        stats["negative_records"] += len(chosen_negative)
        stats["eligible_negative_units"] += bool(negatives)
        stats["sampled_negative_units"] += bool(chosen_negative)
        if not positives:
            stats["uncovered_nonempty_units"] += 1

    selected.sort(
        key=lambda row: _stable_int(
            seed,
            row["meta"]["case_id"],
            row["meta"]["defendant_index"],
            row["meta"]["court_unit_id"],
            row["meta"]["group_index"],
            "shuffle",
        )
    )
    result_stats = {
        "input_records": len(records),
        "input_units": len(grouped),
        "output_records": len(selected),
        **dict(stats),
    }
    return selected, result_stats


def joint_rank_record_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_counts = [len(record["meta"]["candidate_law_ids"]) for record in records]
    return {
        "records": len(records),
        "units": len(
            {
                (
                    record["meta"]["case_id"],
                    str(record["meta"]["defendant_index"]),
                    record["meta"]["court_unit_id"],
                )
                for record in records
            }
        ),
        "exact_positive_records": sum(bool(record["ranking"]["exact_positive_indices"]) for record in records),
        "null_target_records": sum(bool(record["ranking"]["target_is_null"]) for record in records),
        "article_positive_records": sum(bool(record["ranking"]["gold_family_indices"]) for record in records),
        "avg_candidates": sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0,
        "max_candidates": max(candidate_counts, default=0),
    }


def _unit_key(meta: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(meta.get("case_id")),
        str(meta.get("defendant_index")),
        str(meta.get("defendant")),
        str(meta.get("court_unit_id")),
    )


def aggregate_joint_rank_scores(
    predictions: list[dict[str, Any]],
    *,
    max_candidates: int = 16,
    always_keep_top_n: int = 5,
) -> list[dict[str, Any]]:
    """Build one bounded candidate pool per unit from groupwise marker logits.

    Original retrieval Top-N candidates are retained as a safety floor.  The
    remaining slots are filled by candidate-minus-NULL logit, which makes
    scores from separately presented groups approximately comparable.
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    if always_keep_top_n < 0:
        raise ValueError("always_keep_top_n must be >= 0")
    if always_keep_top_n > max_candidates:
        raise ValueError("always_keep_top_n cannot exceed max_candidates")

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in predictions:
        grouped.setdefault(_unit_key(row.get("meta", {})), []).append(row)

    output: list[dict[str, Any]] = []
    for rows in grouped.values():
        candidates: dict[str, dict[str, Any]] = {}
        for row in rows:
            meta = row.get("meta", {})
            ids = [str(item) for item in meta.get("candidate_law_ids", [])]
            ranks = [int(item) for item in meta.get("original_candidate_ranks", [])]
            scores = [float(item) for item in row.get("candidate_scores", [])]
            if not (len(ids) == len(ranks) == len(scores)):
                raise ValueError("candidate IDs, original ranks, and scores must have equal lengths")
            for law_id, rank, score in zip(ids, ranks, scores, strict=True):
                previous = candidates.get(law_id)
                current = {"law_id": law_id, "rank": rank, "score": score}
                if previous is None or score > previous["score"]:
                    candidates[law_id] = current

        by_original_rank = sorted(candidates.values(), key=lambda item: (item["rank"], item["law_id"]))
        safety_floor = [item for item in by_original_rank if item["rank"] <= always_keep_top_n]
        safety_ids = {item["law_id"] for item in safety_floor}
        learned = sorted(
            (item for item in candidates.values() if item["law_id"] not in safety_ids),
            key=lambda item: (-item["score"], item["rank"], item["law_id"]),
        )
        selected = (safety_floor + learned)[:max_candidates]
        first_meta = rows[0].get("meta", {})
        output.append(
            {
                "case_id": first_meta.get("case_id"),
                "defendant": first_meta.get("defendant"),
                "defendant_index": first_meta.get("defendant_index"),
                "court_unit_id": first_meta.get("court_unit_id"),
                "gold_laws": [str(item) for item in first_meta.get("gold_laws", [])],
                "retrieved_law_ids": [item["law_id"] for item in selected],
                "retrieved_law_scores": [item["score"] for item in selected],
                "original_candidate_ranks": [item["rank"] for item in selected],
                "group_prediction_count": len(rows),
            }
        )
    return output
