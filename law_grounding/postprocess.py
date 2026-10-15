"""Model output parsing and candidate-ID validation."""

from __future__ import annotations

import json
from typing import Any


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from model text."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def sanitize_prediction(raw_output: str, sample: dict[str, Any]) -> dict[str, Any]:
    """Parse model output, filter IDs outside candidates, and record invalid IDs."""
    obj = extract_json_object(raw_output)
    if obj is None:
        return {
            "pred": {"support_facts": [], "support_laws": []},
            "raw_output": raw_output,
            "valid_json": False,
            "invalid_ids": [],
        }

    fact_candidates = {str(item["id"]) for item in sample.get("fact_candidates", [])}
    law_candidates = {str(item["id"]) for item in sample.get("law_candidates", [])}
    raw_facts = _dedupe(_string_list(obj.get("support_facts")))
    raw_laws = _dedupe(_string_list(obj.get("support_laws")))

    pred_facts: list[str] = []
    pred_laws: list[str] = []
    invalid_ids: list[str] = []
    for fact_id in raw_facts:
        if fact_id in fact_candidates:
            pred_facts.append(fact_id)
        else:
            invalid_ids.append(fact_id)
    for law_id in raw_laws:
        if law_id in law_candidates:
            pred_laws.append(law_id)
        else:
            invalid_ids.append(law_id)

    return {
        "pred": {"support_facts": pred_facts, "support_laws": pred_laws},
        "raw_output": raw_output,
        "valid_json": True,
        "invalid_ids": invalid_ids,
    }
