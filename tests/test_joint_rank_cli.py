import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        ("scripts/prepare_joint_rank_grounding_data.py", "--negative-unit-fraction"),
        ("scripts/resample_joint_rank_training_data.py", "--negative-unit-fraction"),
        ("scripts/train_joint_rank_grounding.py", "--ranking-loss-weight"),
        ("scripts/infer_joint_rank_grounding.py", "--always-keep-top-n"),
        ("scripts/aggregate_joint_rank_scores.py", "--predictions"),
        ("scripts/evaluate_prepared_candidate_pool.py", "--top-k"),
    ],
)
def test_joint_rank_script_help_runs_from_project_root(script, expected):
    result = subprocess.run(
        [sys.executable, script, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
