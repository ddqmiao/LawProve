"""Evaluation metrics for court-view grounding predictions."""

from __future__ import annotations

from typing import Any

from .law import article_key, is_paragraph_law_id, normalize_law_id


def _as_set(row: dict[str, Any], side: str, key: str) -> set[str]:
    return set(str(item) for item in row.get(side, {}).get(key, []))


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _micro_prf(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    tp = fp = fn = 0
    for row in rows:
        gold = _as_set(row, "gold", key)
        pred = _as_set(row, "pred", key)
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    return _prf(tp, fp, fn)


def _article_law_set(law_ids: set[str]) -> set[str]:
    return {article_key(law_id) for law_id in law_ids}


def _article_level_law_f1(rows: list[dict[str, Any]]) -> float:
    article_rows = []
    for row in rows:
        article_rows.append(
            {
                "gold": {"support_laws": sorted(_article_law_set(_as_set(row, "gold", "support_laws")))},
                "pred": {"support_laws": sorted(_article_law_set(_as_set(row, "pred", "support_laws")))},
            }
        )
    return _micro_prf(article_rows, "support_laws")["f1"]


def _has_paragraph_confusion(row: dict[str, Any]) -> bool:
    gold = _as_set(row, "gold", "support_laws")
    pred = _as_set(row, "pred", "support_laws")
    if gold == pred:
        return False
    for gold_id in gold:
        for pred_id in pred:
            if normalize_law_id(gold_id) != normalize_law_id(pred_id) and article_key(gold_id) == article_key(pred_id):
                if is_paragraph_law_id(gold_id) or is_paragraph_law_id(pred_id):
                    return True
    return False


def evaluate_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    fact = _micro_prf(rows, "support_facts")
    law = _micro_prf(rows, "support_laws")
    joint_exact = 0
    valid_json = 0
    invalid_id_rows = 0
    paragraph_confusions = 0
    fact_exact = 0
    law_exact = 0
    law_cardinality_exact = 0
    empty_fact_rows = 0
    empty_fact_correct = 0
    empty_law_rows = 0
    empty_law_correct = 0
    predicted_fact_count = 0
    predicted_law_count = 0
    gold_fact_count = 0
    gold_law_count = 0
    for row in rows:
        if row.get("valid_json"):
            valid_json += 1
        if row.get("invalid_ids"):
            invalid_id_rows += 1
        gold_facts = _as_set(row, "gold", "support_facts")
        pred_facts = _as_set(row, "pred", "support_facts")
        gold_laws = _as_set(row, "gold", "support_laws")
        pred_laws = _as_set(row, "pred", "support_laws")
        fact_exact += gold_facts == pred_facts
        law_exact += gold_laws == pred_laws
        law_cardinality_exact += len(gold_laws) == len(pred_laws)
        predicted_fact_count += len(pred_facts)
        predicted_law_count += len(pred_laws)
        gold_fact_count += len(gold_facts)
        gold_law_count += len(gold_laws)
        if not gold_facts:
            empty_fact_rows += 1
            empty_fact_correct += not pred_facts
        if not gold_laws:
            empty_law_rows += 1
            empty_law_correct += not pred_laws
        if gold_facts == pred_facts and gold_laws == pred_laws:
            joint_exact += 1
        if _has_paragraph_confusion(row):
            paragraph_confusions += 1

    return {
        "num_samples": n,
        "valid_json_rate": valid_json / n if n else 0.0,
        "invalid_id_rate": invalid_id_rows / n if n else 0.0,
        "fact_precision": fact["precision"],
        "fact_recall": fact["recall"],
        "fact_f1": fact["f1"],
        "law_precision": law["precision"],
        "law_recall": law["recall"],
        "law_f1": law["f1"],
        "fact_set_exact_match": fact_exact / n if n else 0.0,
        "law_set_exact_match": law_exact / n if n else 0.0,
        "law_cardinality_exact_match": law_cardinality_exact / n if n else 0.0,
        "joint_exact_match": joint_exact / n if n else 0.0,
        "empty_fact_accuracy": empty_fact_correct / empty_fact_rows if empty_fact_rows else 0.0,
        "empty_law_accuracy": empty_law_correct / empty_law_rows if empty_law_rows else 0.0,
        "avg_gold_facts": gold_fact_count / n if n else 0.0,
        "avg_predicted_facts": predicted_fact_count / n if n else 0.0,
        "avg_gold_laws": gold_law_count / n if n else 0.0,
        "avg_predicted_laws": predicted_law_count / n if n else 0.0,
        "article_level_law_f1": _article_level_law_f1(rows),
        "paragraph_level_error_rate": paragraph_confusions / n if n else 0.0,
    }
