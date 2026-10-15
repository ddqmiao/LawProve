"""Sampling helpers for high-recall listwise law selector training."""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from typing import Any


def _unit_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    meta = record.get("meta", {})
    return (
        str(meta.get("case_id")),
        str(meta.get("defendant_index")),
        str(meta.get("defendant")),
        str(meta.get("court_unit_id")),
    )


def _target(record: dict[str, Any]) -> dict[str, list[str]]:
    return json.loads(record["messages"][-1]["content"])


def _is_positive(record: dict[str, Any]) -> bool:
    target = _target(record)
    return bool(target.get("keep") or target.get("maybe"))


def rebalance_listwise_records(
    records: list[dict[str, Any]],
    *,
    max_negative_groups_per_unit: int = 1,
    empty_gold_unit_fraction: float = 0.1,
    late_positive_copies: int = 2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep all positive groups while limiting easy all-reject groups."""
    if max_negative_groups_per_unit < 0:
        raise ValueError("max_negative_groups_per_unit must be >= 0")
    if not 0.0 <= empty_gold_unit_fraction <= 1.0:
        raise ValueError("empty_gold_unit_fraction must be between 0 and 1")
    if late_positive_copies < 1:
        raise ValueError("late_positive_copies must be >= 1")

    grouped: OrderedDict[tuple[str, str, str, str], list[dict[str, Any]]] = OrderedDict()
    for record in records:
        grouped.setdefault(_unit_key(record), []).append(record)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row.get("meta", {}).get("group_index", 0)))

    rng = random.Random(seed)
    empty_keys = [
        key
        for key, rows in grouped.items()
        if not any(row.get("meta", {}).get("gold_laws") for row in rows)
    ]
    selected_empty_count = round(len(empty_keys) * empty_gold_unit_fraction)
    selected_empty_keys = set(rng.sample(empty_keys, selected_empty_count))

    output: list[dict[str, Any]] = []
    positive_records = 0
    late_positive_extra_copies = 0
    negative_records = 0
    for key, rows in grouped.items():
        is_empty = key in empty_keys
        if is_empty:
            if key not in selected_empty_keys:
                continue
            chosen = rows[:max_negative_groups_per_unit]
            output.extend(chosen)
            negative_records += len(chosen)
            continue

        positives = [row for row in rows if _is_positive(row)]
        negatives = [row for row in rows if not _is_positive(row)]
        for row in positives:
            copies = late_positive_copies if int(row.get("meta", {}).get("group_index", 0)) > 0 else 1
            output.extend([row] * copies)
            positive_records += 1
            late_positive_extra_copies += copies - 1
        chosen = negatives[:max_negative_groups_per_unit]
        output.extend(chosen)
        negative_records += len(chosen)

    rng.shuffle(output)
    stats = {
        "input_records": len(records),
        "input_units": len(grouped),
        "empty_gold_units": len(empty_keys),
        "selected_empty_gold_units": len(selected_empty_keys),
        "positive_records": positive_records,
        "late_positive_extra_copies": late_positive_extra_copies,
        "negative_records": negative_records,
        "output_records": len(output),
    }
    return output, stats
