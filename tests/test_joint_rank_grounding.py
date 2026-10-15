import pytest

from law_grounding.joint_rank_grounding import (
    aggregate_joint_rank_scores,
    build_all_joint_rank_records,
    build_joint_rank_record,
    joint_rank_record_stats,
    select_joint_rank_training_records,
)


LAW_MAP = {
    "25": {"type": "article", "article_id": "25", "article_label": "第二十五条", "text": "共同犯罪整条。"},
    "25-1": {
        "type": "paragraph",
        "article_id": "25",
        "article_label": "第二十五条",
        "paragraph_no": 1,
        "text": "共同故意犯罪。",
    },
    "67": {"type": "article", "article_id": "67", "article_label": "第六十七条", "text": "自首和坦白整条。"},
    "67-1": {
        "type": "paragraph",
        "article_id": "67",
        "article_label": "第六十七条",
        "paragraph_no": 1,
        "text": "自首。",
    },
    "67-3": {
        "type": "paragraph",
        "article_id": "67",
        "article_label": "第六十七条",
        "paragraph_no": 3,
        "text": "如实供述。",
    },
    "72-1": {
        "type": "paragraph",
        "article_id": "72",
        "article_label": "第七十二条",
        "paragraph_no": 1,
        "text": "缓刑。",
    },
    "73-3": {
        "type": "paragraph",
        "article_id": "73",
        "article_label": "第七十三条",
        "paragraph_no": 3,
        "text": "缓刑考验期限。",
    },
}


def sample(*, case_id="case-a", gold_laws=None):
    return {
        "case_id": case_id,
        "defendant": "张某",
        "defendant_index": 0,
        "court_unit_id": "CV0",
        "court_view": "张某到案后如实供述自己的罪行，可以从轻处罚。",
        "fact_candidates": [{"id": "F1", "text": "张某到案后如实供述。"}],
        "law_candidates": [
            {"id": "67-3", "text": ""},
            {"id": "25-1", "text": ""},
            {"id": "67-1", "text": ""},
            {"id": "72-1", "text": ""},
            {"id": "73-3", "text": ""},
        ],
        "gold_facts": ["F1"],
        "gold_laws": ["67-3"] if gold_laws is None else gold_laws,
    }


def test_joint_rank_record_tracks_exact_family_null_and_original_ranks_after_shuffle():
    candidates = sample()["law_candidates"][:3]
    record = build_joint_rank_record(
        sample(),
        candidates,
        LAW_MAP,
        group_index=2,
        group_start_rank=21,
        seed=9,
    )

    ids = record["meta"]["candidate_law_ids"]
    rank_by_id = dict(zip(ids, record["meta"]["original_candidate_ranks"], strict=True))
    assert rank_by_id == {"67-3": 21, "25-1": 22, "67-1": 23}
    assert [ids[index] for index in record["ranking"]["exact_positive_indices"]] == ["67-3"]
    assert record["ranking"]["target_is_null"] is False
    assert record["ranking"]["gold_family_indices"]
    assert record["messages"][-1]["role"] == "user"
    assert "完整条文：自首和坦白整条。" in record["messages"][-1]["content"]
    for marker, law_id in zip(record["ranking"]["candidate_markers"], ids, strict=True):
        assert f"{marker} = {law_id}" in record["messages"][-1]["content"]


def test_all_records_use_true_global_topk_ranks_for_short_final_group():
    records = build_all_joint_rank_records([sample()], LAW_MAP, top_k=5, group_size=2, seed=3)

    rank_by_id = {
        law_id: rank
        for record in records
        for law_id, rank in zip(
            record["meta"]["candidate_law_ids"],
            record["meta"]["original_candidate_ranks"],
            strict=True,
        )
    }
    assert rank_by_id == {"67-3": 1, "25-1": 2, "67-1": 3, "72-1": 4, "73-3": 5}
    assert joint_rank_record_stats(records)["records"] == 3


def test_training_selection_uses_positive_and_same_article_hard_negative():
    records = build_all_joint_rank_records([sample()], LAW_MAP, top_k=5, group_size=2, seed=4)
    selected, stats = select_joint_rank_training_records(records, seed=4)

    assert len(selected) == 2
    positive = next(row for row in selected if row["ranking"]["exact_positive_indices"])
    negative = next(row for row in selected if not row["ranking"]["exact_positive_indices"])
    assert "67-3" in positive["meta"]["candidate_law_ids"]
    assert "67-1" in negative["meta"]["candidate_law_ids"]
    assert negative["ranking"]["gold_family_indices"]
    assert negative["ranking"]["target_is_null"] is True
    assert negative["ranking"]["family_target_is_null"] is False
    assert stats["positive_records"] == 1
    assert stats["negative_records"] == 1


def test_empty_gold_unit_selects_one_explicit_null_group():
    records = build_all_joint_rank_records(
        [sample(case_id="empty", gold_laws=[])],
        LAW_MAP,
        top_k=5,
        group_size=2,
    )
    selected, stats = select_joint_rank_training_records(records)

    assert len(selected) == 1
    assert selected[0]["ranking"]["target_is_null"] is True
    assert selected[0]["ranking"]["family_target_is_null"] is True
    assert stats["empty_records"] == 1


def test_negative_unit_fraction_is_deterministic_and_does_not_drop_positives_or_empty_units():
    samples = [sample(case_id=f"case-{index}") for index in range(20)]
    samples.append(sample(case_id="empty", gold_laws=[]))
    records = build_all_joint_rank_records(samples, LAW_MAP, top_k=5, group_size=2)

    first, stats = select_joint_rank_training_records(records, negative_unit_fraction=0.25, seed=17)
    second, _ = select_joint_rank_training_records(records, negative_unit_fraction=0.25, seed=17)

    assert first == second
    assert stats["positive_records"] == 20
    assert stats["empty_records"] == 1
    assert 0 < stats["negative_records"] < 20


def test_negative_unit_fraction_must_be_a_probability():
    with pytest.raises(ValueError, match="between 0 and 1"):
        select_joint_rank_training_records([], negative_unit_fraction=1.1)


def prediction(group_index, ids, ranks, scores):
    return {
        "meta": {
            "case_id": "case-a",
            "defendant": "张某",
            "defendant_index": 0,
            "court_unit_id": "CV0",
            "gold_laws": ["67-3"],
            "group_index": group_index,
            "candidate_law_ids": ids,
            "original_candidate_ranks": ranks,
        },
        "candidate_scores": scores,
    }


def test_aggregate_keeps_retrieval_floor_then_fills_by_candidate_null_margin():
    rows = aggregate_joint_rank_scores(
        [
            prediction(0, ["25-1", "67-3"], [2, 1], [-4.0, -5.0]),
            prediction(1, ["72-1", "67-1"], [4, 3], [3.0, 1.0]),
            prediction(2, ["73-3"], [5], [2.0]),
        ],
        max_candidates=4,
        always_keep_top_n=2,
    )

    assert rows[0]["retrieved_law_ids"] == ["67-3", "25-1", "72-1", "73-3"]
    assert rows[0]["original_candidate_ranks"] == [1, 2, 4, 5]
    assert rows[0]["group_prediction_count"] == 3


def test_aggregate_rejects_misaligned_prediction_vectors():
    with pytest.raises(ValueError, match="equal lengths"):
        aggregate_joint_rank_scores([prediction(0, ["67-3"], [1], [])])
