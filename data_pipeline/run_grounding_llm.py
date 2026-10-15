"""Run DeepSeek on prepared defendant-level grounding prompts."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


JsonRecord = dict[str, Any]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS_INPUT = SCRIPT_DIR / "outputs" / "grounding_prompts.sample.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "grounding_model_outputs.sample.jsonl"
DEFAULT_ENV_FILE = SCRIPT_DIR / ".env.local"

DEFAULT_DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


def load_env_file(path: str | Path) -> list[str]:
    """Load KEY=VALUE lines into os.environ and return loaded key names only."""

    loaded_keys: list[str] = []
    env_path = Path(path)
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"invalid env line {line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid env line {line_number}: empty key")
        os.environ[key] = _strip_env_quotes(value.strip())
        loaded_keys.append(key)
    return loaded_keys


def select_first_cases(rows: list[JsonRecord], *, limit_cases: int | None) -> list[JsonRecord]:
    """Select all task rows belonging to the first N distinct case IDs."""

    if limit_cases is None:
        return rows
    if limit_cases < 1:
        raise ValueError("limit_cases must be at least 1")

    selected_case_ids: list[str] = []
    selected_case_set: set[str] = set()
    selected_rows: list[JsonRecord] = []
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if case_id not in selected_case_set:
            if len(selected_case_ids) >= limit_cases:
                break
            selected_case_ids.append(case_id)
            selected_case_set.add(case_id)
        selected_rows.append(row)
    return selected_rows


def extract_json_object(text: str) -> JsonRecord | None:
    """Parse a JSON object from a model response, including simple fenced blocks."""

    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
        if match:
            stripped = match.group(1).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return value


def build_request_messages(row: JsonRecord) -> list[JsonRecord]:
    """Return chat messages for a prompt row, preferring cache-friendly messages."""

    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        normalized_messages = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"messages[{index}] must be an object")
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", ""))
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"messages[{index}] has invalid role: {role}")
            normalized_messages.append({"role": role, "content": content})
        return normalized_messages
    return [{"role": "user", "content": str(row["prompt"])}]


def complete_deepseek(prompt: str | list[JsonRecord]) -> str:
    """Call DeepSeek chat completions and return message content."""

    return str(complete_deepseek_response(prompt)["content"])


def complete_deepseek_response(prompt: str | list[JsonRecord]) -> JsonRecord:
    """Call DeepSeek chat completions and return content plus response usage."""

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key or api_key == "replace-with-your-deepseek-api-key":
        raise RuntimeError("DEEPSEEK_API_KEY is missing or still set to placeholder")

    api_url = get_deepseek_api_url()
    payload = {
        "model": _env_value("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
        "messages": _normalize_messages(prompt),
        "response_format": {"type": "json_object"},
    }
    _apply_thinking_config(payload, api_url)
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek API request failed: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"DeepSeek API request failed: {exc}") from exc

    try:
        content = str(response_payload["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"DeepSeek API returned unexpected response: {response_payload}") from exc
    usage = response_payload.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    return {"content": content, "usage": usage}


def get_deepseek_api_url() -> str:
    """Resolve the chat completions URL from explicit URL or OpenAI-compatible base URL."""

    explicit_url = _env_value("DEEPSEEK_API_URL")
    if explicit_url:
        return explicit_url.rstrip("/")
    base_url = _env_value("DEEPSEEK_BASE_URL")
    if not base_url:
        return DEFAULT_DEEPSEEK_API_URL
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def run_prompts_file(
    prompts_input_path: str | Path,
    output_path: str | Path,
    *,
    limit_cases: int | None,
    complete_fn: Callable[[list[JsonRecord]], str | JsonRecord] = complete_deepseek_response,
    sleep_seconds: float = 0.0,
) -> JsonRecord:
    """Run model completion for selected prompt rows and write JSONL output."""

    rows = _read_jsonl(prompts_input_path)
    selected_rows = select_first_cases(rows, limit_cases=limit_cases)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    tasks_called = 0
    with output_path.open("w", encoding="utf-8") as target:
        for row in selected_rows:
            started_at = time.time()
            tasks_called += 1
            try:
                completion_result = complete_fn(build_request_messages(row))
                raw_output, usage = _normalize_completion_result(completion_result)
                error = None
            except Exception as exc:
                raw_output = ""
                usage = {}
                error = str(exc)
            parsed_output = extract_json_object(raw_output) if raw_output else None
            prompt_cache_hit_tokens = _usage_int(usage, "prompt_cache_hit_tokens")
            prompt_cache_miss_tokens = _usage_int(usage, "prompt_cache_miss_tokens")
            output = {
                "task_id": row.get("task_id"),
                "case_id": row.get("case_id"),
                "defendant_index": row.get("defendant_index"),
                "defendant_name": row.get("defendant_name"),
                "prompt_char_count": row.get("prompt_char_count"),
                "messages_char_count": row.get("messages_char_count"),
                "raw_output": raw_output,
                "parsed_output": parsed_output,
                "usage": usage,
                "prompt_cache_hit_tokens": prompt_cache_hit_tokens,
                "prompt_cache_miss_tokens": prompt_cache_miss_tokens,
                "cache_hit_rate": _cache_hit_rate(prompt_cache_hit_tokens, prompt_cache_miss_tokens),
                "error": error,
                "elapsed_seconds": round(time.time() - started_at, 3),
            }
            target.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            if error and _is_authentication_error(error):
                break
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    return {
        "prompts_read": len(rows),
        "tasks_called": tasks_called,
        "written": written,
        "output": str(output_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DeepSeek on prepared grounding prompts.")
    parser.add_argument("--prompts-input", type=Path, default=DEFAULT_PROMPTS_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--limit-cases", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--confirm-real-call", action="store_true", help="Required to call DeepSeek.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.confirm_real_call:
        raise SystemExit("Refusing real API call without --confirm-real-call")
    load_env_file(args.env_file)
    summary = run_prompts_file(
        args.prompts_input,
        args.output,
        limit_cases=args.limit_cases,
        sleep_seconds=args.sleep_seconds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _read_jsonl(path: str | Path) -> list[JsonRecord]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected JSON object")
            rows.append(row)
    return rows


def _strip_env_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _env_value(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _apply_thinking_config(payload: JsonRecord, api_url: str) -> None:
    thinking = str(_env_value("DEEPSEEK_THINKING", "disabled")).lower()
    if thinking in {"disabled", "false", "0", "none", "off"}:
        if "uni-api.cstcloud.cn" not in api_url:
            payload["thinking"] = {"type": "disabled"}
        return
    if "uni-api.cstcloud.cn" in api_url:
        payload["chat_template_kwargs"] = {"thinking": True}
    else:
        payload["thinking"] = {"type": thinking}


def _normalize_messages(prompt: str | list[JsonRecord]) -> list[JsonRecord]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    normalized_messages = []
    for index, message in enumerate(prompt):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be an object")
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", ""))
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"messages[{index}] has invalid role: {role}")
        normalized_messages.append({"role": role, "content": content})
    return normalized_messages


def _normalize_completion_result(result: str | JsonRecord) -> tuple[str, JsonRecord]:
    if isinstance(result, dict):
        content = str(result.get("content", ""))
        usage = result.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        return content, usage
    return str(result), {}


def _usage_int(usage: JsonRecord, key: str) -> int:
    value = usage.get(key, 0)
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _cache_hit_rate(prompt_cache_hit_tokens: int, prompt_cache_miss_tokens: int) -> float | None:
    total = prompt_cache_hit_tokens + prompt_cache_miss_tokens
    if total <= 0:
        return None
    return round(prompt_cache_hit_tokens / total, 6)


def _is_authentication_error(error: str) -> bool:
    lowered = error.lower()
    return "http 401" in lowered or "http 403" in lowered or "authentication" in lowered


if __name__ == "__main__":
    main()
