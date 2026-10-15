import json
from pathlib import Path

from split_cmdl_sentences import (
    DEFAULT_COURT_VIEW_PREFIX,
    DEFAULT_FACT_PREFIX,
    label_sentences,
    process_jsonl,
    split_sentences,
    transform_record,
)


def test_split_sentences_keeps_terminators_and_drops_empty_parts():
    text = "甲某盗窃手机一部。乙某负责望风；\n案发后归案。"

    assert split_sentences(text) == [
        "甲某盗窃手机一部。",
        "乙某负责望风；",
        "案发后归案。",
    ]


def test_label_sentences_uses_requested_prefix():
    assert label_sentences("甲某盗窃。乙某望风。", DEFAULT_FACT_PREFIX) == [
        {"id": "F0", "text": "甲某盗窃。"},
        {"id": "F1", "text": "乙某望风。"},
    ]


def test_transform_record_replaces_only_fact_and_court_view():
    record = {
        "fact": "甲某盗窃。乙某望风。",
        "court_view": "本院认为，甲某构成盗窃罪。可从轻处罚。",
        "defendants": ["甲某", "乙某"],
        "outcomes": [{"name": "甲某"}],
    }

    transformed = transform_record(record)

    assert transformed["fact"] == [
        {"id": "F0", "text": "甲某盗窃。"},
        {"id": "F1", "text": "乙某望风。"},
    ]
    assert transformed["court_view"] == [
        {"id": "CV0", "text": "本院认为，甲某构成盗窃罪。"},
        {"id": "CV1", "text": "可从轻处罚。"},
    ]
    assert transformed["defendants"] == record["defendants"]
    assert transformed["outcomes"] == record["outcomes"]


def test_process_jsonl_creates_output_dir_and_writes_transformed_records(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "outputs" / "split.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "fact": "甲某盗窃。乙某望风。",
                "court_view": "本院认为，构成盗窃罪。",
                "relevant_articles": ["264"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = process_jsonl(input_path, output_path)

    assert summary == {"read": 1, "written": 1, "output": str(output_path)}
    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert written[0]["fact"][0] == {"id": "F0", "text": "甲某盗窃。"}
    assert written[0]["court_view"][0] == {
        "id": f"{DEFAULT_COURT_VIEW_PREFIX}0",
        "text": "本院认为，构成盗窃罪。",
    }
    assert written[0]["relevant_articles"] == ["264"]
