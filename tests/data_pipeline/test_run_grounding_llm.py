import json
import os
from pathlib import Path

from run_grounding_llm import (
    build_request_messages,
    extract_json_object,
    get_deepseek_api_url,
    load_env_file,
    run_prompts_file,
    select_first_cases,
)


def prompt_row(case_id: str, task_id: str, defendant_name: str = "张某"):
    return {
        "task_id": task_id,
        "case_id": case_id,
        "defendant_index": 0,
        "defendant_name": defendant_name,
        "prompt": f"为{defendant_name}标注",
        "prompt_char_count": 6,
    }


def messages_prompt_row(case_id: str, task_id: str):
    row = prompt_row(case_id, task_id)
    row["messages"] = [
        {"role": "system", "content": "固定规则"},
        {"role": "user", "content": "当前任务"},
    ]
    row["messages_char_count"] = 8
    return row


def test_build_request_messages_prefers_messages_field_over_prompt():
    row = messages_prompt_row("case_0", "case_0__D0")

    messages = build_request_messages(row)

    assert messages == [
        {"role": "system", "content": "固定规则"},
        {"role": "user", "content": "当前任务"},
    ]


def test_build_request_messages_falls_back_to_legacy_prompt():
    messages = build_request_messages(prompt_row("case_0", "case_0__D0"))

    assert messages == [{"role": "user", "content": "为张某标注"}]


def test_load_env_file_sets_key_values_without_returning_secret(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    env_path = tmp_path / ".env.local"
    env_path.write_text(
        "DEEPSEEK_API_KEY='fake-key'\nDEEPSEEK_MODEL=deepseek-v4-flash\n",
        encoding="utf-8",
    )

    loaded = load_env_file(env_path)

    assert loaded == ["DEEPSEEK_API_KEY", "DEEPSEEK_MODEL"]
    assert os.environ["DEEPSEEK_API_KEY"] == "fake-key"
    assert os.environ["DEEPSEEK_MODEL"] == "deepseek-v4-flash"


def test_get_deepseek_api_url_uses_openai_compatible_base_url(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_URL", raising=False)
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://uni-api.cstcloud.cn/v1")

    assert get_deepseek_api_url() == "https://uni-api.cstcloud.cn/v1/chat/completions"


def test_get_deepseek_api_url_prefers_explicit_chat_url(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://uni-api.cstcloud.cn/v1")
    monkeypatch.setenv("DEEPSEEK_API_URL", "https://example.test/v1/chat/completions")

    assert get_deepseek_api_url() == "https://example.test/v1/chat/completions"


def test_select_first_cases_keeps_all_rows_for_selected_case_ids():
    rows = [
        prompt_row("case_0", "case_0__D0"),
        prompt_row("case_0", "case_0__D1"),
        prompt_row("case_1", "case_1__D0"),
        prompt_row("case_2", "case_2__D0"),
    ]

    selected = select_first_cases(rows, limit_cases=2)

    assert [row["task_id"] for row in selected] == ["case_0__D0", "case_0__D1", "case_1__D0"]


def test_extract_json_object_accepts_plain_json_and_markdown_fence():
    assert extract_json_object('{"grounding_links": []}') == {"grounding_links": []}
    assert extract_json_object('```json\n{"grounding_links": []}\n```') == {"grounding_links": []}


def test_run_prompts_file_writes_raw_and_parsed_outputs(tmp_path: Path):
    prompts_path = tmp_path / "prompts.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    prompts_path.write_text(
        "\n".join(
            [
                json.dumps(prompt_row("case_0", "case_0__D0"), ensure_ascii=False),
                json.dumps(prompt_row("case_0", "case_0__D1", "李某"), ensure_ascii=False),
                json.dumps(prompt_row("case_1", "case_1__D0"), ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_complete(messages: list[dict[str, str]]) -> dict:
        assert messages == [{"role": "user", "content": "为张某标注"}]
        return {
            "content": json.dumps({"grounding_links": [{"court_unit_id": "CV0", "fact_ids": ["F0"], "law_ids": ["264"]}]}),
            "usage": {"prompt_cache_hit_tokens": 12, "prompt_cache_miss_tokens": 3, "prompt_tokens": 15},
        }

    summary = run_prompts_file(prompts_path, output_path, limit_cases=1, complete_fn=fake_complete)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert summary == {"prompts_read": 3, "tasks_called": 2, "written": 2, "output": str(output_path)}
    assert rows[0]["task_id"] == "case_0__D0"
    assert rows[0]["parsed_output"]["grounding_links"][0]["court_unit_id"] == "CV0"
    assert rows[0]["usage"]["prompt_cache_hit_tokens"] == 12
    assert rows[0]["cache_hit_rate"] == 0.8
    assert rows[1]["defendant_name"] == "李某"


def test_run_prompts_file_uses_messages_field_when_present(tmp_path: Path):
    prompts_path = tmp_path / "prompts.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    prompts_path.write_text(json.dumps(messages_prompt_row("case_0", "case_0__D0"), ensure_ascii=False) + "\n", encoding="utf-8")

    def fake_complete(messages: list[dict[str, str]]) -> str:
        assert messages == [
            {"role": "system", "content": "固定规则"},
            {"role": "user", "content": "当前任务"},
        ]
        return json.dumps({"grounding_links": [{"court_unit_id": "CV0", "fact_ids": ["F0"], "law_ids": ["264"]}]})

    summary = run_prompts_file(prompts_path, output_path, limit_cases=1, complete_fn=fake_complete)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert summary == {"prompts_read": 1, "tasks_called": 1, "written": 1, "output": str(output_path)}
    assert rows[0]["task_id"] == "case_0__D0"
    assert rows[0]["parsed_output"]["grounding_links"][0]["court_unit_id"] == "CV0"


def test_run_prompts_file_stops_after_authentication_error(tmp_path: Path):
    prompts_path = tmp_path / "prompts.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    prompts_path.write_text(
        "\n".join(
            [
                json.dumps(prompt_row("case_0", "case_0__D0"), ensure_ascii=False),
                json.dumps(prompt_row("case_0", "case_0__D1", "李某"), ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_complete(prompt: str) -> str:
        raise RuntimeError("DeepSeek API request failed: HTTP 401: authentication failed")

    summary = run_prompts_file(prompts_path, output_path, limit_cases=1, complete_fn=fake_complete)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert summary["tasks_called"] == 1
    assert rows[0]["error"].startswith("DeepSeek API request failed: HTTP 401")
