import pytest

from law_grounding.jsonl_shards import merge_rows, shard_rows_by_fields


def test_shard_rows_by_fields_keeps_candidate_group_together():
    rows = [
        {"meta": {"case_id": "case-a", "defendant_index": 0}, "candidate": "a1"},
        {"meta": {"case_id": "case-b", "defendant_index": 0}, "candidate": "b1"},
        {"meta": {"case_id": "case-a", "defendant_index": 0}, "candidate": "a2"},
        {"meta": {"case_id": "case-c", "defendant_index": 1}, "candidate": "c1"},
        {"meta": {"case_id": "case-b", "defendant_index": 0}, "candidate": "b2"},
    ]

    shards = shard_rows_by_fields(rows, ["meta.case_id", "meta.defendant_index"], num_shards=2)

    assert len(shards) == 2
    placements = {}
    for shard_index, shard in enumerate(shards):
        for row in shard:
            key = (row["meta"]["case_id"], row["meta"]["defendant_index"])
            assert placements.setdefault(key, shard_index) == shard_index
    assert sum(len(shard) for shard in shards) == len(rows)


def test_shard_rows_by_fields_rejects_missing_key():
    with pytest.raises(KeyError):
        shard_rows_by_fields([{"meta": {"case_id": "case-a"}}], ["meta.defendant_index"], num_shards=2)


def test_merge_rows_preserves_file_order():
    shards = [
        [{"id": "a"}, {"id": "b"}],
        [{"id": "c"}],
    ]

    assert merge_rows(shards) == [{"id": "a"}, {"id": "b"}, {"id": "c"}]
