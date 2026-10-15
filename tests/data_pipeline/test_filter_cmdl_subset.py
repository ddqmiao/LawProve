import json
from pathlib import Path

from filter_cmdl_subset import (
    FlawStats,
    analyze_flaws,
    classify_court_view_truncation,
    compute_metrics,
    keep_by_thresholds,
    process_jsonl,
)


def make_record(
    *,
    fact: str = "被告人甲某、乙某共同盗窃手机一部。",
    court_view: str = "本院认为，被告人甲某、乙某构成盗窃罪。",
    defendants: list[str] | None = None,
    outcomes: list[dict] | None = None,
    relevant_articles: list[str] | None = None,
) -> dict:
    if defendants is None:
        defendants = ["甲某", "乙某"]
    if outcomes is None:
        outcomes = [{"name": name, "judgment": [{"article": ["264"]}]} for name in defendants]
    if relevant_articles is None:
        relevant_articles = ["264"]
    return {
        "fact": fact,
        "court_view": court_view,
        "defendants": defendants,
        "outcomes": outcomes,
        "relevant_articles": relevant_articles,
    }


def test_keep_by_thresholds_uses_total_length_and_defendant_count():
    kept = make_record(fact="甲" * 5999 + "。", court_view="乙" * 5999 + "。")
    too_long = make_record(fact="甲" * 12000 + "。", court_view="乙。")
    too_many_defendants = make_record(defendants=["甲", "乙", "丙", "丁", "戊", "己", "庚"])

    assert keep_by_thresholds(compute_metrics(kept), max_total_len=12000, max_defendants=6) == []
    assert keep_by_thresholds(compute_metrics(too_long), max_total_len=12000, max_defendants=6) == [
        "total_len_gt_12000"
    ]
    assert keep_by_thresholds(compute_metrics(too_many_defendants), max_total_len=12000, max_defendants=6) == [
        "defendant_count_gt_6"
    ]


def test_analyze_flaws_flags_incomplete_text_and_alignment_errors():
    record = make_record(
        fact="事实没有句末标点",
        court_view="据此，",
        defendants=["甲某", "乙某"],
        outcomes=[{"name": "甲某"}, {"name": "丙某"}],
        relevant_articles=[],
    )

    assert analyze_flaws(record) == [
        "fact_no_terminal_punct",
        "court_view_no_terminal_punct",
        "defendant_outcome_name_mismatch",
        "bad_relevant_articles",
    ]


def test_process_jsonl_writes_kept_records_and_reports_flaw_ratios(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "filtered.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    rows = [
        make_record(),
        make_record(fact="甲" * 12000 + "。", court_view="乙。"),
        make_record(court_view="据此，"),
    ]
    input_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    summary = process_jsonl(
        input_path,
        output_path,
        rejected_path,
        max_total_len=12000,
        max_defendants=6,
        split_name="train_big",
    )

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    rejected = [json.loads(line) for line in rejected_path.read_text(encoding="utf-8").splitlines()]
    assert [row["source_line_no"] for row in written] == [1, 3]
    assert summary["read"] == 3
    assert summary["kept"] == 2
    assert summary["rejected"] == 1
    assert summary["flawed_records"] == 1
    assert summary["flaw_reasons"] == {"court_view_no_terminal_punct": 1}
    assert summary["kept_flaw_stats"] == {
        "flawed_records": 1,
        "flaw_reasons": {"court_view_no_terminal_punct": 1},
    }
    assert rejected[0]["reject_reasons"] == ["total_len_gt_12000"]


def test_classify_court_view_truncation_allows_only_low_risk_tails():
    minor = (
        "本院认为，被告人甲某、乙某以非法占有为目的秘密窃取他人财物，数额较大，其行为均已构成盗窃罪。"
        "二被告人归案后能够如实供述自己的罪行，依法可以从轻处罚，公诉机关的量刑建议适当，本院予以采纳。据此，"
    )
    risky = (
        "本院认为，被告人甲某、乙某构成盗窃罪。二被告人如实供述，可以从轻处罚。"
        "依照《中华人民共和国刑法》第二百六十四条，对被告人甲某"
    )

    assert classify_court_view_truncation(minor)["severity"] == "minor_trimmed"
    assert classify_court_view_truncation(minor)["cleaned_court_view"].endswith("本院予以采纳。")
    assert classify_court_view_truncation(risky)["severity"] == "severe_reject"


def test_process_jsonl_clean_minor_flaws_trims_minor_and_rejects_other_flaws(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "filtered.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    rows = [
        make_record(),
        make_record(
            court_view=(
                "本院认为，被告人甲某、乙某以非法占有为目的秘密窃取他人财物，数额较大，其行为均已构成盗窃罪。"
                "二被告人归案后能够如实供述自己的罪行，依法可以从轻处罚，公诉机关的量刑建议适当，本院予以采纳。据此，"
            )
        ),
        make_record(fact="事实没有句末标点"),
        make_record(
            court_view=(
                "本院认为，被告人甲某、乙某构成盗窃罪。二被告人如实供述，可以从轻处罚。"
                "对被告人甲某还"
            )
        ),
        make_record(fact="甲" * 8000 + "。", court_view="乙。"),
    ]
    input_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    summary = process_jsonl(
        input_path,
        output_path,
        rejected_path,
        max_total_len=8000,
        max_defendants=6,
        split_name="train_big",
        clean_minor_flaws=True,
    )

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    rejected = [json.loads(line) for line in rejected_path.read_text(encoding="utf-8").splitlines()]
    assert [row["source_line_no"] for row in written] == [1, 2]
    assert written[0]["court_view_trimmed"] is False
    assert written[1]["court_view_trimmed"] is True
    assert written[1]["tail_fragment"] == "据此，"
    assert written[1]["court_view"].endswith("本院予以采纳。")
    assert summary["read"] == 5
    assert summary["kept"] == 2
    assert summary["rejected"] == 3
    assert summary["trimmed_records"] == 1
    assert summary["reject_reasons"] == {
        "fact_no_terminal_punct": 1,
        "court_view_truncation_not_minor": 1,
        "total_len_gt_8000": 1,
    }
    assert [row["reject_reasons"][0] for row in rejected] == [
        "fact_no_terminal_punct",
        "court_view_truncation_not_minor",
        "total_len_gt_8000",
    ]


def test_process_jsonl_can_write_clean_schema_without_trace_fields(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "filtered.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    original = make_record(
        court_view=(
            "本院认为，被告人甲某、乙某以非法占有为目的秘密窃取他人财物，数额较大，其行为均已构成盗窃罪。"
            "二被告人归案后能够如实供述自己的罪行，依法可以从轻处罚，公诉机关的量刑建议适当，本院予以采纳。据此，"
        )
    )
    input_path.write_text(json.dumps(original, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = process_jsonl(
        input_path,
        output_path,
        rejected_path,
        max_total_len=8000,
        max_defendants=6,
        split_name="train_big",
        clean_minor_flaws=True,
        keep_metadata=False,
    )

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert summary["trimmed_records"] == 1
    assert len(written) == 1
    assert set(written[0]) == set(original)
    assert written[0]["court_view"].endswith("本院予以采纳。")
    assert "据此，" not in written[0]["court_view"]
