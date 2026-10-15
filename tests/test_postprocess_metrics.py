from law_grounding.metrics import evaluate_predictions
from law_grounding.postprocess import sanitize_prediction


def test_sanitize_prediction_extracts_json_and_filters_candidate_external_ids():
    sample = {
        "fact_candidates": [{"id": "F0", "text": "事实0"}, {"id": "F1", "text": "事实1"}],
        "law_candidates": [{"id": "67-3", "text": "坦白"}, {"id": "125-2", "text": "危险物质"}],
    }
    raw = '解释文字 {"support_facts":["F0","F9"],"support_laws":["67-3","999"]} trailing'

    result = sanitize_prediction(raw, sample)

    assert result["valid_json"] is True
    assert result["pred"] == {"support_facts": ["F0"], "support_laws": ["67-3"]}
    assert result["invalid_ids"] == ["F9", "999"]


def test_sanitize_prediction_marks_invalid_json():
    sample = {"fact_candidates": [], "law_candidates": []}

    result = sanitize_prediction("not json", sample)

    assert result["valid_json"] is False
    assert result["pred"] == {"support_facts": [], "support_laws": []}


def test_evaluate_predictions_computes_grounding_and_law_diagnostics():
    rows = [
        {
            "gold": {"support_facts": ["F0", "F1"], "support_laws": ["125-2"]},
            "pred": {"support_facts": ["F0"], "support_laws": ["125-1"]},
            "valid_json": True,
            "invalid_ids": [],
        },
        {
            "gold": {"support_facts": [], "support_laws": ["67-3"]},
            "pred": {"support_facts": [], "support_laws": ["67-3"]},
            "valid_json": False,
            "invalid_ids": ["F9"],
        },
    ]

    metrics = evaluate_predictions(rows)

    assert metrics["num_samples"] == 2
    assert metrics["valid_json_rate"] == 0.5
    assert metrics["invalid_id_rate"] == 0.5
    assert round(metrics["fact_precision"], 4) == 1.0
    assert round(metrics["fact_recall"], 4) == 0.5
    assert round(metrics["fact_f1"], 4) == 0.6667
    assert round(metrics["law_precision"], 4) == 0.5
    assert round(metrics["law_recall"], 4) == 0.5
    assert round(metrics["law_f1"], 4) == 0.5
    assert round(metrics["article_level_law_f1"], 4) == 1.0
    assert metrics["joint_exact_match"] == 0.5
    assert metrics["paragraph_level_error_rate"] == 0.5
