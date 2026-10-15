import subprocess
import sys


def test_prepare_data_script_help_runs_from_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/prepare_data.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--data-dir" in result.stdout
    assert "--prompt-variant" in result.stdout
    assert "--retrieval-candidates-dir" in result.stdout
    assert "--no-filter-missing-law-candidates" in result.stdout


def test_prepare_retrieved_sft_script_help_runs_from_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/prepare_retrieved_sft.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--samples-dir" in result.stdout
    assert "--retrieval-candidates-dir" in result.stdout
    assert "--filter-missing-law-candidate-splits" in result.stdout


def test_evaluate_law_retrieval_script_help_runs_from_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_law_retrieval.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--law-map" in result.stdout
    assert "--samples" in result.stdout
    assert "--query-facts" in result.stdout


def test_evaluate_embedding_retrieval_script_help_runs_from_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_embedding_retrieval.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--embedding-model" in result.stdout
    assert "--batch-size" in result.stdout


def test_evaluate_hybrid_retrieval_script_help_runs_from_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_hybrid_retrieval.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--retrieval-files" in result.stdout
    assert "--rrf-k" in result.stdout


def test_evaluate_supervised_retrieval_script_help_runs_from_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_supervised_retrieval.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--train-samples" in result.stdout
    assert "--eval-samples" in result.stdout


def test_evaluate_supervised_oof_retrieval_script_help_runs_from_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_supervised_oof_retrieval.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--samples" in result.stdout
    assert "--num-folds" in result.stdout
    assert "--group-field" in result.stdout
    assert "--seed" in result.stdout
    assert "--fold-manifest-output" in result.stdout




def test_prepare_law_listwise_sft_script_help_runs_from_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/prepare_law_listwise_sft.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--samples-dir" in result.stdout
    assert "--group-size" in result.stdout
    assert "--top-k" in result.stdout




def test_infer_law_listwise_selector_script_help_runs_from_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/infer_law_listwise_selector.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--input" in result.stdout
    assert "--aggregate-output" in result.stdout


def test_aggregate_law_listwise_predictions_script_help_runs_from_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/aggregate_law_listwise_predictions.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--predictions" in result.stdout
    assert "--metrics-output" in result.stdout
    assert "--always-keep-top-n" in result.stdout


def test_rebalance_law_listwise_sft_script_help_runs_from_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/rebalance_law_listwise_sft.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--max-negative-groups-per-unit" in result.stdout
    assert "--empty-gold-unit-fraction" in result.stdout
    assert "--late-positive-copies" in result.stdout










def test_train_lora_script_supports_trainable_adapter_initialization():
    result = subprocess.run(
        [sys.executable, "scripts/train_lora.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--init-adapter" in result.stdout
    assert "--group-by-length" in result.stdout
    assert "--dataloader-num-workers" in result.stdout
    assert "--optim" in result.stdout








def test_split_jsonl_by_fields_script_help_runs_from_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/split_jsonl_by_fields.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--input" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--fields" in result.stdout


def test_merge_jsonl_script_help_runs_from_project_root():
    result = subprocess.run(
        [sys.executable, "scripts/merge_jsonl.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--inputs" in result.stdout
    assert "--output" in result.stdout


