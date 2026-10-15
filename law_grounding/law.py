"""Law ID normalization and law text helpers."""

from __future__ import annotations

import re
from typing import Any


CN_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def chinese_number_to_int(text: str) -> int:
    """Convert small Chinese numerals used in law amendment IDs to int."""
    if not text:
        raise ValueError("empty Chinese number")
    if text == "十":
        return 10
    if "十" in text:
        left, _, right = text.partition("十")
        tens = CN_DIGITS.get(left, 1) if left else 1
        ones = CN_DIGITS.get(right, 0) if right else 0
        return tens * 10 + ones
    if text in CN_DIGITS:
        return CN_DIGITS[text]
    raise ValueError(f"unsupported Chinese number: {text}")


def normalize_law_id(law_id: str | int) -> str:
    """Normalize amendment-style IDs, preserving ordinary paragraph IDs.

    Examples:
        224之一 -> 224.1
        253之一-1 -> 253.1-1
    """
    value = str(law_id).strip()

    def repl(match: re.Match[str]) -> str:
        return "." + str(chinese_number_to_int(match.group(1)))

    return re.sub(r"之([一二三四五六七八九十]+)", repl, value)


def lookup_law_info(law_id: str, law_map: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Return law-map info by direct ID or normalized ID."""
    if law_id in law_map:
        return law_map[law_id]
    normalized = normalize_law_id(law_id)
    return law_map.get(normalized)


def article_key(law_id: str, law_map: dict[str, dict[str, Any]] | None = None) -> str:
    """Return the article-level key used for article-level comparison."""
    if law_map:
        info = lookup_law_info(law_id, law_map)
        if info and info.get("article_id"):
            return str(info["article_id"])
    return normalize_law_id(law_id).split("-", 1)[0]


def is_paragraph_law_id(law_id: str, law_map: dict[str, dict[str, Any]] | None = None) -> bool:
    """Whether a law ID denotes a paragraph-level provision."""
    if law_map:
        info = lookup_law_info(law_id, law_map)
        if info and info.get("type"):
            return info.get("type") == "paragraph"
    return bool(re.search(r"-\d+$", normalize_law_id(law_id)))


def law_sort_key(law_id: str) -> tuple[Any, ...]:
    """Sort law IDs in human-stable numeric order where possible."""
    normalized = normalize_law_id(law_id)
    parts: list[Any] = []
    for chunk in re.split(r"([.-])", normalized):
        if not chunk or chunk in {".", "-"}:
            continue
        parts.append(int(chunk) if chunk.isdigit() else chunk)
    return tuple(parts)


def sibling_paragraph_ids(law_id: str, law_map: dict[str, dict[str, Any]]) -> list[str]:
    """Return all paragraph IDs under the same article as law_id."""
    key = article_key(law_id, law_map)
    siblings = [
        item_id
        for item_id, info in law_map.items()
        if info.get("type") == "paragraph" and str(info.get("article_id")) == key
    ]
    return sorted(siblings, key=law_sort_key)


def format_law_text(law_id: str, law_map: dict[str, dict[str, Any]]) -> str:
    """Format a candidate law as '<original id>: <label>:<text>'."""
    info = lookup_law_info(law_id, law_map)
    if not info:
        return f"{law_id}: 法条ID：{law_id}"

    label = str(info.get("article_label") or "")
    text = str(info.get("text") or "").strip()
    if info.get("type") == "paragraph":
        paragraph_no = info.get("paragraph_no")
        return f"{law_id}: {label}第{paragraph_no}款：{text}"
    return f"{law_id}: {label}：{text}"
