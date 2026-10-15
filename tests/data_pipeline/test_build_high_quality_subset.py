import json
from pathlib import Path

from build_high_quality_subset import (
    case_id_for,
    choose_split_targets,
    is_high_quality_candidate,
    record_with_case_id,
    select_diverse_candidates,
)


def make_record(
    *,
    accusation: str = "盗窃罪",
    defendant_count: int = 3,
    fact: str | None = None,
    court_view: str | None = None,
    relevant_articles: list[str] | None = None,
    judgment_articles: list[str] | None = None,
) -> dict:
    defendants = [f"被告人{i}" for i in range(defendant_count)]
    if fact is None:
        fact = "经审理查明，" + "二被告人共同实施犯罪行为，事实清楚。" * 20
    if court_view is None:
        court_view = (
            "本院认为，被告人的行为均已构成盗窃罪。"
            "各被告人归案后能够如实供述自己的罪行，依法可以从轻处罚。"
            "公诉机关指控的罪名成立，量刑建议适当，本院予以采纳。"
            "在共同犯罪中，各被告人分工配合，应当按照各自犯罪事实、情节及悔罪表现分别量刑。"
            "结合退赃、认罪认罚等情节，对部分被告人可以适用缓刑。"
            "综上，本院根据各被告人在犯罪中的作用和社会危害程度依法作出判决。"
        )
    if relevant_articles is None:
        relevant_articles = ["264", "67-3"]
    if judgment_articles is None:
        judgment_articles = ["264"]
    return {
        "fact": fact,
        "court_view": court_view,
        "defendants": defendants,
        "outcomes": [
            {
                "name": name,
                "judgment": [
                    {
                        "accusation": accusation,
                        "standard_accusation": accusation,
                        "article": judgment_articles,
                        "penalty": {
                            "surveillance": 0,
                            "detention": 0,
                            "imprisonment": 6,
                            "death_penalty": False,
                            "life_imprisonment": False,
                            "fine": 1000,
                            "fine_without_amount": False,
                        },
                    }
                ],
            }
            for name in defendants
        ],
        "relevant_articles": relevant_articles,
    }


def test_is_high_quality_candidate_accepts_multi_defendant_clean_records():
    record = make_record(defendant_count=5)

    candidate = is_high_quality_candidate(record)

    assert candidate.accepted is True
    assert candidate.defendant_count == 5
    assert candidate.charge == "盗窃罪"


def test_is_high_quality_candidate_rejects_structural_and_text_quality_issues():
    multi_judgment = make_record()
    multi_judgment["outcomes"][0]["judgment"].append(multi_judgment["outcomes"][0]["judgment"][0])
    contaminated = make_record(
        court_view=(
            "公诉机关认为，被告人构成盗窃罪。经审理查明，被告人盗窃手机。"
            "上述事实，有证人证言、辨认笔录、鉴定意见等证据证实。本院认为，被告人构成盗窃罪。"
        )
    )
    too_many_articles = make_record(relevant_articles=[str(i) for i in range(10)])

    assert is_high_quality_candidate(multi_judgment).accepted is False
    assert "not_single_judgment" in is_high_quality_candidate(multi_judgment).reasons
    assert is_high_quality_candidate(contaminated).accepted is False
    assert "court_view_bad_start" in is_high_quality_candidate(contaminated).reasons
    assert is_high_quality_candidate(too_many_articles).accepted is False
    assert "article_count_gt_8" in is_high_quality_candidate(too_many_articles).reasons


def test_select_diverse_candidates_respects_defendant_targets_and_charge_cap():
    candidates = []
    for index in range(20):
        record = make_record(accusation="盗窃罪", defendant_count=2)
        candidates.append(is_high_quality_candidate(record, split_name="train", line_number=index))
    for index in range(20, 40):
        record = make_record(accusation="诈骗罪", defendant_count=3)
        candidates.append(is_high_quality_candidate(record, split_name="train", line_number=index))
    for index in range(40, 60):
        record = make_record(accusation="抢劫罪", defendant_count=4)
        candidates.append(is_high_quality_candidate(record, split_name="train", line_number=index))

    selected = select_diverse_candidates(
        candidates,
        target_count=12,
        defendant_weights={2: 0.4, 3: 0.35, 4: 0.25},
        max_charge_share=0.5,
        seed=7,
    )

    assert len(selected) == 12
    charge_counts = {}
    defendant_counts = {}
    for candidate in selected:
        charge_counts[candidate.charge] = charge_counts.get(candidate.charge, 0) + 1
        defendant_counts[candidate.defendant_count] = defendant_counts.get(candidate.defendant_count, 0) + 1
    assert max(charge_counts.values()) <= 6
    assert defendant_counts[2] >= 4
    assert defendant_counts[3] >= 3
    assert defendant_counts[4] >= 2


def test_choose_split_targets_uses_train_valid_test_ratio():
    assert choose_split_targets(["train_big", "valid_big", "test_big"], 20000) == {
        "train_big": 16000,
        "valid_big": 2000,
        "test_big": 2000,
    }


def test_record_with_case_id_adds_artificial_case_number_without_mutating_record():
    record = make_record()

    with_case_id = record_with_case_id(record, case_id_for("train_big.len8000.def6", 1))

    assert list(with_case_id)[:2] == ["case_id", "fact"]
    assert with_case_id["case_id"] == "train_000001"
    assert "case_id" not in record
    assert case_id_for("valid_big.len8000.def6", 12) == "valid_000012"
    assert case_id_for("test_big.len8000.def6", 123) == "test_000123"
