import json
from pathlib import Path

from merge_cv_grounding_outputs import merge_outputs, process_file


def cv_tasks():
    return [
        {
            "task_id": "train_000000__D0__CV1",
            "case_id": "train_000000",
            "defendant_index": 0,
            "defendant_name": "张某",
            "court_unit": {"id": "CV1", "text": "张某系主犯。"},
            "fact_candidates": [{"id": "F1", "text": "张某负责实施盗窃。"}, {"id": "F2", "text": "证据证实。"}],
            "law_candidates": [{"article": "26-1", "text": "主犯。"}],
        },
        {
            "task_id": "train_000000__D0__CV0",
            "case_id": "train_000000",
            "defendant_index": 0,
            "defendant_name": "张某",
            "court_unit": {"id": "CV0", "text": "张某构成盗窃罪。"},
            "fact_candidates": [{"id": "F0", "text": "张某盗窃手机。"}],
            "law_candidates": [{"article": "264", "text": "盗窃罪。"}],
        },
    ]


def model_outputs():
    return [
        {
            "task_id": "train_000000__D0__CV1",
            "parsed_output": {
                "grounding_link": {
                    "court_unit_id": "CV1",
                    "fact_ids": ["F2", "BAD", "F1"],
                    "law_ids": ["26-1", "999"],
                }
            },
            "error": None,
        },
        {
            "task_id": "train_000000__D0__CV0",
            "parsed_output": {
                "grounding_links": [
                    {
                        "court_unit_id": "CV0",
                        "fact_ids": ["F0"],
                        "law_ids": ["264"],
                    }
                ]
            },
            "error": None,
        },
    ]


def test_merge_outputs_filters_invalid_ids_and_orders_links_by_court_unit():
    rows = merge_outputs(cv_tasks(), model_outputs())

    assert rows == [
        {
            "case_id": "train_000000",
            "defendant_index": 0,
            "defendant_name": "张某",
            "grounding_links": [
                {"court_unit_id": "CV0", "fact_ids": ["F0"], "law_ids": ["264"]},
                {"court_unit_id": "CV1", "fact_ids": ["F1", "F2"], "law_ids": ["26-1"]},
            ],
            "warnings": [
                {"task_id": "train_000000__D0__CV1", "type": "bad_fact", "value": "BAD"},
                {"task_id": "train_000000__D0__CV1", "type": "bad_law", "value": "999"},
            ],
        }
    ]


def test_process_file_writes_merged_defendant_rows(tmp_path: Path):
    tasks_path = tmp_path / "cv_tasks.jsonl"
    outputs_path = tmp_path / "cv_outputs.jsonl"
    merged_path = tmp_path / "merged.jsonl"
    tasks_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in cv_tasks()) + "\n", encoding="utf-8")
    outputs_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in model_outputs()) + "\n",
        encoding="utf-8",
    )

    summary = process_file(tasks_path, outputs_path, merged_path)

    rows = [json.loads(line) for line in merged_path.read_text(encoding="utf-8").splitlines()]
    assert summary == {
        "tasks_read": 2,
        "model_outputs_read": 2,
        "defendants_written": 1,
        "output": str(merged_path),
    }
    assert rows[0]["grounding_links"][0]["court_unit_id"] == "CV0"
