import json

import pytest

from scripts.train_joint_rank_grounding import (
    is_distributed,
    marker_token_id,
    model_load_kwargs,
    read_jsonl,
    should_write_artifacts,
    validate_rows,
)


class Tokenizer:
    def __init__(self, values):
        self.values = values

    def __call__(self, marker, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": self.values[marker]}


def generation_row():
    return {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }


def ranking_row():
    return {
        "task_type": "law_listwise_rank",
        "messages": [{"role": "user", "content": "rank"}],
        "ranking": {
            "candidate_markers": ["A", "B"],
            "null_marker": "N",
            "exact_positive_indices": [1],
            "target_is_null": False,
            "candidate_family_indices": [0, 1],
            "gold_family_indices": [1],
            "family_target_is_null": False,
        },
    }


def test_read_jsonl_honors_limit(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps({"id": value}) for value in range(3)), encoding="utf-8")

    assert read_jsonl(path, limit=2) == [{"id": 0}, {"id": 1}]


def test_distributed_loading_omits_device_map_and_only_rank_zero_writes():
    env = {"WORLD_SIZE": "2", "RANK": "1"}

    assert is_distributed(env) is True
    assert model_load_kwargs("bf16", "auto", env) == {"torch_dtype": "bf16", "trust_remote_code": True}
    assert should_write_artifacts(env) is False


def test_marker_must_be_exactly_one_token():
    assert marker_token_id(Tokenizer({"A": [7]}), "A") == 7
    with pytest.raises(ValueError, match="exactly one token"):
        marker_token_id(Tokenizer({"A": [7, 8]}), "A")


def test_validate_rows_accepts_joint_contract_and_rejects_contradictory_null_target():
    validate_rows([generation_row()], [ranking_row()])
    bad = ranking_row()
    bad["ranking"]["target_is_null"] = True

    with pytest.raises(ValueError, match="xor NULL"):
        validate_rows([generation_row()], [bad])
