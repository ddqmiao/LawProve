"""Structured law candidate formatting for listwise selection."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from .law import article_key, law_sort_key, lookup_law_info


def _law_type(law_id: str, info: dict[str, Any] | None) -> str:
    if info and info.get("type"):
        return str(info["type"])
    return "paragraph" if "-" in law_id else "article"


def structured_selectable_item(law_id: str, law_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build one selectable law item, preserving the exact output ID."""
    info = lookup_law_info(law_id, law_map) or {}
    return {
        "id": str(law_id),
        "type": _law_type(str(law_id), info),
        "article_id": str(info.get("article_id") or article_key(str(law_id), law_map)),
        "article_label": str(info.get("article_label") or ""),
        "paragraph_no": info.get("paragraph_no"),
        "text": str(info.get("text") or "").strip(),
    }


def build_structured_law_group(
    article_id: str,
    selectable_ids: list[str],
    law_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build an article-context group with only the provided selectable IDs."""
    article_info = lookup_law_info(article_id, law_map) or {}
    unique_ids = sorted(dict.fromkeys(str(item) for item in selectable_ids), key=law_sort_key)
    article_label = str(article_info.get("article_label") or "")
    if not article_label and unique_ids:
        first_info = lookup_law_info(unique_ids[0], law_map) or {}
        article_label = str(first_info.get("article_label") or "")
    return {
        "article_id": str(article_id),
        "article_label": article_label,
        "article_text": str(article_info.get("text") or "").strip(),
        "selectable_items": [structured_selectable_item(law_id, law_map) for law_id in unique_ids],
    }


def group_candidate_ids_by_article(
    candidate_ids: list[str],
    law_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group candidate IDs by article while preserving first article occurrence."""
    grouped: OrderedDict[str, list[str]] = OrderedDict()
    for law_id in candidate_ids:
        key = article_key(str(law_id), law_map)
        grouped.setdefault(key, []).append(str(law_id))
    return [build_structured_law_group(article_id, ids, law_map) for article_id, ids in grouped.items()]


def format_structured_law_group(group: dict[str, Any]) -> str:
    """Render one structured law group for an LLM prompt."""
    header = f"所属条文：{group.get('article_label') or group.get('article_id')}（article_id={group.get('article_id')}）"
    article_text = f"完整条文：{group.get('article_text') or '未提供完整条文。'}"
    lines = [header, article_text, "可选ID："]
    for item in group.get("selectable_items", []):
        type_label = str(item.get("type") or "")
        paragraph_no = item.get("paragraph_no")
        suffix = f" 第{paragraph_no}款" if paragraph_no is not None else ""
        lines.append(f"- {item['id']} [{type_label}{suffix}]：{item.get('text') or ''}")
    return "\n".join(lines)
