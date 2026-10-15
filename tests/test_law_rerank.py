import json

from law_grounding.law_rerank import (
    aggregate_law_rerank_predictions,
    build_law_rerank_records,
    parse_law_rerank_relevance,
)


LAW_MAP = {
    "25": {"type": "article", "article_id": "25", "article_label": "第二十五条", "text": "共同犯罪整条。"},
    "25-1": {"type": "paragraph", "article_id": "25", "article_label": "第二十五条", "paragraph_no": 1, "text": "共同故意犯罪。"},
    "67-1": {"type": "paragraph", "article_id": "67", "article_label": "第六十七条", "paragraph_no": 1, "text": "自首。"},
    "67-3": {"type": "paragraph", "article_id": "67", "article_label": "第六十七条", "paragraph_no": 3, "text": "如实供述。"},
    "72-1": {"type": "paragraph", "article_id": "72", "article_label": "第七十二条", "paragraph_no": 1, "text": "缓刑。"},
}


def sample_with_retrieved_laws():
    return {
        "case_id": "test_000001",
        "defendant": "张某",
        "defendant_index": 0,
        "court_unit_id": "CV0",
        "court_view": "张某到案后如实供述自己的罪行，可以从轻处罚。",
        "fact_candidates": [
            {"id": "F0", "text": "张某实施犯罪。"},
            {"id": "F1", "text": "张某到案后如实供述。"},
        ],
        "law_candidates": [
            {"id": "67-1", "text": "67-1: 第六十七条第1款：自首。"},
            {"id": "67-3", "text": "67-3: 第六十七条第3款：如实供述。"},
            {"id": "72-1", "text": "72-1: 第七十二条第1款：缓刑。"},
            {"id": "25-1", "text": "25-1: 第二十五条第1款：共同故意犯罪。"},
        ],
        "gold_facts": ["F1"],
        "gold_laws": ["67-3"],
    }


def test_build_law_rerank_records_labels_exact_positive_and_same_article_hard_negative():
    records = build_law_rerank_records(
        [sample_with_retrieved_laws()],
        LAW_MAP,
        top_k=4,
        negatives_per_positive=1,
        hard_negatives_per_positive=2,
    )

    labels_by_law = {record["meta"]["candidate_law_id"]: json.loads(record["messages"][-1]["content"])["relevant"] for record in records}
    negative_types = {record["meta"]["candidate_law_id"]: record["meta"]["negative_type"] for record in records}

    assert labels_by_law["67-3"] is True
    assert labels_by_law["67-1"] is False
    assert negative_types["67-1"] == "same_article"
    assert "必须精确区分整条与具体款项" in records[0]["messages"][0]["content"]
    assert "候选法条ID：67-1" in records[0]["messages"][1]["content"]


def test_build_law_rerank_records_can_emit_all_candidates_for_evaluation():
    records = build_law_rerank_records(
        [sample_with_retrieved_laws()],
        LAW_MAP,
        top_k=3,
        include_all_candidates=True,
    )

    assert [record["meta"]["candidate_law_id"] for record in records] == ["67-1", "67-3", "72-1"]
    assert [record["meta"]["candidate_rank"] for record in records] == [1, 2, 3]
    assert [json.loads(record["messages"][-1]["content"])["relevant"] for record in records] == [False, True, False]


def test_parse_law_rerank_relevance_accepts_json_boolean_and_records_invalid_output():
    assert parse_law_rerank_relevance('{"relevant": true}') == {"relevant": True, "valid_json": True}
    assert parse_law_rerank_relevance('解释 {"relevant": "否"}') == {"relevant": False, "valid_json": True}
    assert parse_law_rerank_relevance("not json") == {"relevant": False, "valid_json": False}


def test_aggregate_law_rerank_predictions_keeps_candidate_rank_order():
    candidate_predictions = [
        {
            "meta": {
                "case_id": "test_000001",
                "defendant": "张某",
                "defendant_index": 0,
                "court_unit_id": "CV0",
                "candidate_law_id": "67-1",
                "candidate_rank": 1,
                "gold_laws": ["67-3"],
                "gold_facts": ["F1"],
            },
            "pred_relevant": False,
        },
        {
            "meta": {
                "case_id": "test_000001",
                "defendant": "张某",
                "defendant_index": 0,
                "court_unit_id": "CV0",
                "candidate_law_id": "67-3",
                "candidate_rank": 2,
                "gold_laws": ["67-3"],
                "gold_facts": ["F1"],
            },
            "pred_relevant": True,
        },
    ]

    rows = aggregate_law_rerank_predictions(candidate_predictions)

    assert rows == [
        {
            "case_id": "test_000001",
            "defendant": "张某",
            "defendant_index": 0,
            "court_unit_id": "CV0",
            "gold": {"support_facts": ["F1"], "support_laws": ["67-3"]},
            "pred": {"support_facts": [], "support_laws": ["67-3"]},
            "candidate_prediction_count": 2,
            "valid_json_rate": 0.0,
            "valid_json": False,
        }
    ]


def test_aggregate_law_rerank_predictions_can_limit_rank_and_selected_count():
    candidate_predictions = []
    for rank, law_id in enumerate(["67-1", "67-3", "72-1", "53"], start=1):
        candidate_predictions.append(
            {
                "meta": {
                    "case_id": "test_000001",
                    "defendant": "张某",
                    "defendant_index": 0,
                    "court_unit_id": "CV0",
                    "candidate_law_id": law_id,
                    "candidate_rank": rank,
                    "gold_laws": ["67-3"],
                    "gold_facts": ["F1"],
                },
                "pred_relevant": True,
                "valid_json": True,
            }
        )

    rows = aggregate_law_rerank_predictions(candidate_predictions, rank_limit=3, max_selected=2)

    assert rows[0]["pred"]["support_laws"] == ["67-1", "67-3"]
    assert rows[0]["candidate_prediction_count"] == 4
    assert rows[0]["valid_json_rate"] == 1.0
    assert rows[0]["valid_json"] is True
