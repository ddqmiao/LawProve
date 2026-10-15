"""Helpers for sharding and merging JSONL rows."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterable


def value_at_path(row: dict[str, Any], path: str) -> Any:
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


def shard_rows_by_fields(
    rows: list[dict[str, Any]],
    fields: Iterable[str],
    *,
    num_shards: int,
) -> list[list[dict[str, Any]]]:
    """Shard rows while keeping all rows with the same field tuple together."""
    field_list = [field for field in fields if field]
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not field_list:
        raise ValueError("fields must not be empty")

    grouped: OrderedDict[tuple[Any, ...], list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        key = tuple(value_at_path(row, field) for field in field_list)
        grouped.setdefault(key, []).append(row)

    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]
    shard_sizes = [0] * num_shards
    for group in grouped.values():
        shard_index = min(range(num_shards), key=lambda index: shard_sizes[index])
        shards[shard_index].extend(group)
        shard_sizes[shard_index] += len(group)
    return shards


def merge_rows(row_groups: Iterable[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in row_groups:
        rows.extend(group)
    return rows
