import json

from law_grounding.law_listwise_selector import build_listwise_selector_records
from law_grounding.law_listwise_selector import (
    aggregate_listwise_predictions,
    candidate_pool_metrics,
    parse_listwise_selector_output,
)


LAW_MAP = {
    "25": {"type": "article", "article_id": "25", "article_label": "第二十五条", "text": "共同犯罪整条。"},
    "25-1": {"type": "paragraph", "article_id": "25", "article_label": "第二十五条", "paragraph_no": 1, "text": "共同故意犯罪。"},
    "67": {"type": "article", "article_id": "67", "article_label": "第六十七条", "text": "自首和坦白整条。"},
    "67-1": {"type": "paragraph", "article_id": "67", "article_label": "第六十七条", "paragraph_no": 1, "text": "自首。"},
    "67-3": {"type": "paragraph", "article_id": "67", "article_label": "第六十七条", "paragraph_no": 3, "text": "如实供述。"},
    "72-1": {"type": "paragraph", "article_id": "72", "article_label": "第七十二条", "paragraph_no": 1, "text": "缓刑。"},
}


def sample():
    return {
        "case_id": "case-a",
        "defendant": "张某",
        "defendant_index": 0,
        "court_unit_id": "CV0",
        "court_view": "张某到案后如实供述自己的罪行，可以从轻处罚。",
        "fact_candidates": [
            {"id": "F0", "text": "张某实施犯罪。"},
            {"id": "F1", "text": "张某到案后如实供述。"},
        ],
        "law_candidates": [
            {"id": "67-1", "text": ""},
            {"id": "67-3", "text": ""},
            {"id": "25-1", "text": ""},
            {"id": "72-1", "text": ""},
        ],
        "gold_facts": ["F1"],
        "gold_laws": ["67-3"],
    }


def test_build_listwise_selector_records_labels_keep_maybe_reject_by_exact_and_article_match():
    records = build_listwise_selector_records([sample()], LAW_MAP, group_size=4, top_k=4)

    answer = json.loads(records[0]["messages"][-1]["content"])
    assert answer == {
        "keep": ["67-3"],
        "maybe": ["67-1"],
        "reject": ["25-1", "72-1"],
    }
    assert records[0]["meta"]["candidate_law_ids"] == ["67-1", "67-3", "25-1", "72-1"]
    assert records[0]["meta"]["group_index"] == 0


def test_build_listwise_selector_prompt_contains_structured_article_context_and_exact_id_rule():
    records = build_listwise_selector_records([sample()], LAW_MAP, group_size=4, top_k=4, max_fact_chars=1000)
    user_prompt = records[0]["messages"][1]["content"]

    assert "完整条文：自首和坦白整条。" in user_prompt
    assert "可选ID：" in user_prompt
    assert "67-1 [paragraph 第1款]：自首。" in user_prompt
    assert "67-3 [paragraph 第3款]：如实供述。" in user_prompt
    assert "只能从可选ID中输出" in user_prompt


def test_build_listwise_selector_records_chunks_top_k_candidates():
    records = build_listwise_selector_records([sample()], LAW_MAP, group_size=2, top_k=3)

    assert [record["meta"]["candidate_law_ids"] for record in records] == [
        ["67-1", "67-3"],
        ["25-1"],
    ]
    assert json.loads(records[1]["messages"][-1]["content"]) == {
        "keep": [],
        "maybe": [],
        "reject": ["25-1"],
    }


def test_parse_listwise_selector_output_filters_to_candidate_ids_and_dedupes():
    parsed = parse_listwise_selector_output(
        '{"keep":["67-3","999","67-3"],"maybe":["67-1"],"reject":["25-1"]}',
        ["67-1", "67-3", "25-1"],
    )

    assert parsed == {
        "keep": ["67-3"],
        "maybe": ["67-1"],
        "reject": ["25-1"],
        "valid_json": True,
    }


def test_parse_listwise_selector_output_marks_invalid_json():
    assert parse_listwise_selector_output("not json", ["67-3"]) == {
        "keep": [],
        "maybe": [],
        "reject": [],
        "valid_json": False,
    }


def test_aggregate_listwise_predictions_keeps_group_order_and_maybe_candidates():
    predictions = [
        {
            "meta": {
                "case_id": "case-a",
                "defendant": "张某",
                "defendant_index": 0,
                "court_unit_id": "CV0",
                "gold_laws": ["67-3"],
                "candidate_law_ids": ["67-1", "67-3"],
                "group_index": 0,
            },
            "pred_keep": ["67-3"],
            "pred_maybe": ["67-1"],
            "valid_json": True,
        },
        {
            "meta": {
                "case_id": "case-a",
                "defendant": "张某",
                "defendant_index": 0,
                "court_unit_id": "CV0",
                "gold_laws": ["67-3"],
                "candidate_law_ids": ["25-1", "72-1"],
                "group_index": 1,
            },
            "pred_keep": [],
            "pred_maybe": ["25-1"],
            "valid_json": True,
        },
    ]

    rows = aggregate_listwise_predictions(predictions, include_maybe=True, max_candidates=3)

    assert rows == [
        {
            "case_id": "case-a",
            "defendant": "张某",
            "defendant_index": 0,
            "court_unit_id": "CV0",
            "gold_laws": ["67-3"],
            "retrieved_law_ids": ["67-3", "67-1", "25-1"],
            "group_prediction_count": 2,
            "valid_json_rate": 1.0,
            "valid_json": True,
        }
    ]


def test_aggregate_listwise_predictions_always_keeps_original_top_n_first():
    predictions = [
        {
            "meta": {
                "case_id": "case-a",
                "defendant": "张某",
                "defendant_index": 0,
                "court_unit_id": "CV0",
                "gold_laws": ["67-3"],
                "candidate_law_ids": ["67-1", "67-3"],
                "group_index": 0,
            },
            "pred_keep": [],
            "pred_maybe": [],
            "valid_json": True,
        },
        {
            "meta": {
                "case_id": "case-a",
                "defendant": "张某",
                "defendant_index": 0,
                "court_unit_id": "CV0",
                "gold_laws": ["67-3"],
                "candidate_law_ids": ["25-1", "72-1"],
                "group_index": 1,
            },
            "pred_keep": ["72-1"],
            "pred_maybe": [],
            "valid_json": True,
        },
    ]

    rows = aggregate_listwise_predictions(
        predictions,
        include_maybe=True,
        max_candidates=3,
        always_keep_top_n=2,
    )

    assert rows[0]["retrieved_law_ids"] == ["67-1", "67-3", "72-1"]


def test_candidate_pool_metrics_excludes_empty_gold_rows_from_coverage_rates():
    metrics = candidate_pool_metrics(
        [
            {"gold_laws": ["67-3"], "retrieved_law_ids": ["67-3"]},
            {"gold_laws": [], "retrieved_law_ids": []},
        ]
    )

    assert metrics == {
        "num_samples": 2,
        "num_evaluable_samples": 1,
        "num_empty_gold_samples": 1,
        "avg_candidates": 0.5,
        "law_micro_recall_ceiling": 1.0,
        "full_gold_covered_rate": 1.0,
        "any_gold_covered_rate": 1.0,
    }
