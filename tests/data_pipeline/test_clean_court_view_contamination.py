import json
from pathlib import Path

from clean_court_view_contamination import (
    contamination_reasons,
    is_high_confidence_contamination,
    process_jsonl,
)


def make_record(*, fact: str, court_view: str) -> dict:
    return {
        "fact": fact,
        "court_view": court_view,
        "defendants": ["甲某", "乙某"],
        "outcomes": [{"name": "甲某"}, {"name": "乙某"}],
        "relevant_articles": ["264"],
    }


def test_high_confidence_contamination_requires_length_inversion_and_strong_signal():
    contaminated = make_record(
        fact="甲某、乙某共同盗窃手机一部。",
        court_view=(
            "公诉机关认为，被告人甲某、乙某构成盗窃罪。经审理查明，二被告人共同盗窃手机一部。"
            "上述事实，有证人证言、辨认笔录、鉴定意见、书证、抓获经过等证据证实，足以认定。"
            "本院认为，被告人甲某、乙某构成盗窃罪。"
        ),
    )
    normal_long_view = make_record(
        fact="甲某、乙某共同盗窃手机一部。",
        court_view=(
            "本院认为，被告人甲某、乙某以非法占有为目的，秘密窃取他人财物，其行为均已构成盗窃罪。"
            "二被告人共同实施犯罪，系共同犯罪。二被告人到案后能够如实供述自己的罪行，依法可以从轻处罚。"
            "公诉机关指控的罪名成立，量刑建议适当，本院予以采纳。"
        ),
    )
    long_fact = make_record(
        fact="甲某、乙某共同盗窃手机一部。" * 80,
        court_view="经审理查明，甲某、乙某共同盗窃手机一部。本院认为，二被告人构成盗窃罪。",
    )

    assert is_high_confidence_contamination(contaminated) is True
    assert "court_view_gt_fact_with_contamination_signal" in contamination_reasons(contaminated)
    assert is_high_confidence_contamination(normal_long_view) is False
    assert is_high_confidence_contamination(long_fact) is False


def test_process_jsonl_removes_high_confidence_contamination(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "kept.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    kept = make_record(
        fact="甲某、乙某共同盗窃手机一部。",
        court_view=(
            "本院认为，被告人甲某、乙某构成盗窃罪。二被告人如实供述，可以从轻处罚。"
            "公诉机关的量刑建议适当，本院予以采纳。"
        ),
    )
    rejected = make_record(
        fact="甲某、乙某共同盗窃手机一部。",
        court_view=(
            "公诉机关认为，被告人甲某、乙某构成盗窃罪。经审理查明，二被告人共同盗窃手机一部。"
            "上述事实，有证人证言、辨认笔录、鉴定意见、书证、抓获经过等证据证实，足以认定。"
            "本院认为，被告人甲某、乙某构成盗窃罪。"
        ),
    )
    input_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in [kept, rejected]) + "\n",
        encoding="utf-8",
    )

    summary = process_jsonl(input_path, output_path, rejected_path, split_name="train_big")

    kept_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    rejected_rows = [json.loads(line) for line in rejected_path.read_text(encoding="utf-8").splitlines()]
    assert summary["read"] == 2
    assert summary["kept"] == 1
    assert summary["rejected"] == 1
    assert kept_rows == [kept]
    assert rejected_rows[0]["source_line_no"] == 2
    assert rejected_rows[0]["reject_reasons"] == ["court_view_gt_fact_with_contamination_signal"]
