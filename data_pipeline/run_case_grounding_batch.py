"""Run a case-level grounding batch with offset/limit selection."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from build_cv_grounding_tasks import process_file as build_cv_tasks
from merge_cv_grounding_outputs import process_file as merge_cv_outputs
from run_grounding_llm import (
    build_request_messages,
    complete_deepseek_response,
    extract_json_object,
    get_deepseek_api_url,
    load_env_file,
)
from split_cmdl_sentences import transform_record


JsonRecord = dict[str, Any]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = SCRIPT_DIR / "outputs" / "cdml_big_high_quality_20k_all_law_refs" / "train_big.len8000.def6.high_quality.jsonl"
DEFAULT_LAW_MAP = SCRIPT_DIR / "law_article_map.criminal_law.json"
DEFAULT_ENV_FILE = SCRIPT_DIR / ".env.local"


def run_batch(
    source_path: str | Path,
    law_map_path: str | Path,
    output_dir: str | Path,
    *,
    offset_cases: int,
    limit_cases: int,
    env_file: str | Path,
    max_workers: int,
    cleanup_tmp: bool,
    rerun_errors: bool,
) -> JsonRecord:
    output_dir = Path(output_dir)
    split_dir = output_dir / "split"
    final_dir = output_dir / "final"
    tmp_dir = output_dir / "_tmp"
    split_path = split_dir / "train.jsonl"
    final_path = final_dir / "train.jsonl"
    tasks_path = tmp_dir / "cv_grounding_tasks.jsonl"
    prompts_path = tmp_dir / "cv_grounding_prompts.jsonl"
    model_outputs_path = tmp_dir / "cv_grounding_model_outputs.jsonl"
    merged_path = tmp_dir / "cv_grounding_merged.jsonl"
    summary_path = output_dir / "summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    load_env_file(env_file)
    selected_summary = write_split_selection(source_path, split_path, offset_cases=offset_cases, limit_cases=limit_cases)
    task_summary = build_cv_tasks(
        split_path,
        law_map_path,
        tasks_path,
        prompts_output_path=prompts_path,
        case_prefix="train",
        fact_candidates_mode="all",
    )
    model_summary = run_prompts_parallel(
        prompts_path,
        model_outputs_path,
        max_workers=max_workers,
        rerun_errors=rerun_errors,
    )
    merge_summary = merge_cv_outputs(tasks_path, model_outputs_path, merged_path)
    final_summary = write_final_cases(split_path, merged_path, final_path)

    warning_counts = count_final_warnings(final_path)
    summary = {
        "source": str(source_path),
        "output_dir": str(output_dir),
        "selection": {
            "split": "train",
            "offset_cases": offset_cases,
            "limit_cases": limit_cases,
            "line_start_1_based": offset_cases + 1,
            "line_end_1_based": offset_cases + limit_cases,
            "first_case_id": selected_summary.get("first_case_id"),
            "last_case_id": selected_summary.get("last_case_id"),
        },
        "config": {
            "api_url": get_deepseek_api_url(),
            "model": _env_value("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "thinking": _env_value("DEEPSEEK_THINKING", "disabled"),
            "fact_candidates_mode": "all",
            "max_workers": max_workers,
            "rerun_errors": rerun_errors,
        },
        "files": {
            "split_train": str(split_path),
            "final_train": str(final_path),
        },
        "split": selected_summary,
        "tasks": task_summary,
        "model": model_summary,
        "merge": merge_summary,
        "final": final_summary,
        "warnings": dict(warning_counts),
    }

    if cleanup_tmp:
        shutil.rmtree(tmp_dir)
    else:
        summary["files"].update(
            {
                "tasks": str(tasks_path),
                "prompts": str(prompts_path),
                "model_outputs": str(model_outputs_path),
                "merged": str(merged_path),
            }
        )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_split_selection(
    source_path: str | Path,
    output_path: str | Path,
    *,
    offset_cases: int,
    limit_cases: int,
) -> JsonRecord:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    read = 0
    written = 0
    first_case_id = None
    last_case_id = None
    with Path(source_path).open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            read += 1
            if read <= offset_cases:
                continue
            if written >= limit_cases:
                break
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: expected JSON object")
            split_record = transform_record(record)
            case_id = str(split_record.get("case_id", ""))
            first_case_id = first_case_id or case_id
            last_case_id = case_id
            target.write(json.dumps(split_record, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
    if written != limit_cases:
        raise ValueError(f"requested {limit_cases} cases after offset {offset_cases}, wrote {written}")
    return {
        "source_read_until": read,
        "written": written,
        "first_case_id": first_case_id,
        "last_case_id": last_case_id,
        "output": str(output_path),
    }


def run_prompts_parallel(
    prompts_path: str | Path,
    output_path: str | Path,
    *,
    max_workers: int,
    rerun_errors: bool,
) -> JsonRecord:
    rows = _read_jsonl(prompts_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_suffix(".progress.json")

    started_at = time.time()
    existing_by_task = latest_outputs_by_task(output_path)
    pending: list[tuple[int, JsonRecord]] = []
    skipped = 0
    for index, row in enumerate(rows):
        existing = existing_by_task.get(str(row.get("task_id", "")))
        if existing is not None and not should_rerun_output(existing, rerun_errors=rerun_errors):
            skipped += 1
            continue
        pending.append((index, row))

    completed_this_run = 0
    with output_path.open("a", encoding="utf-8") as target, ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(run_one_prompt, index, row): index
            for index, row in pending
        }
        for future in as_completed(future_to_index):
            result = future.result()
            target.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            target.flush()
            completed_this_run += 1
            if completed_this_run % 25 == 0 or completed_this_run == len(pending):
                write_progress(
                    progress_path,
                    total=len(rows),
                    skipped=skipped,
                    pending=len(pending),
                    completed_this_run=completed_this_run,
                    started_at=started_at,
                )

    compacted_rows = compact_model_outputs(output_path)
    desired_task_ids = {str(row.get("task_id", "")) for row in rows}
    latest_rows = [row for row in compacted_rows if str(row.get("task_id", "")) in desired_task_ids]

    usage = aggregate_usage(latest_rows)
    errors = sum(1 for result in latest_rows if result.get("error"))
    unparsed = sum(1 for result in latest_rows if not result.get("error") and result.get("parsed_output") is None)
    return {
        "prompts_read": len(rows),
        "existing_outputs": len(existing_by_task),
        "skipped": skipped,
        "pending": len(pending),
        "requests": len(pending),
        "written": len(latest_rows),
        "errors": errors,
        "unparsed": unparsed,
        "usage": usage,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "output": str(output_path),
        "progress": str(progress_path),
    }


def run_one_prompt(index: int, row: JsonRecord) -> JsonRecord:
    started_at = time.time()
    raw_output = ""
    parsed_output = None
    usage: JsonRecord = {}
    error = None
    attempts = 0
    for attempt in range(1, 3):
        attempts = attempt
        try:
            completion = complete_deepseek_response(build_request_messages(row))
            raw_output = str(completion.get("content", ""))
            usage = completion.get("usage", {})
            if not isinstance(usage, dict):
                usage = {}
            parsed_output = extract_json_object(raw_output) if raw_output else None
            error = None
            break
        except Exception as exc:  # noqa: BLE001 - preserve API error text in JSONL.
            error = str(exc)
            if attempt < 2:
                time.sleep(1.5 * attempt)

    return {
        "row_index": index,
        "task_id": row.get("task_id"),
        "case_id": row.get("case_id"),
        "defendant_index": row.get("defendant_index"),
        "defendant_name": row.get("defendant_name"),
        "court_unit_id": row.get("court_unit_id"),
        "prompt_char_count": row.get("prompt_char_count"),
        "messages_char_count": row.get("messages_char_count"),
        "raw_output": raw_output,
        "parsed_output": parsed_output,
        "usage": usage,
        "error": error,
        "attempts": attempts,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }


def latest_outputs_by_task(path: str | Path) -> dict[str, JsonRecord]:
    path = Path(path)
    if not path.exists():
        return {}
    latest: dict[str, JsonRecord] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            task_id = str(row.get("task_id", ""))
            if not task_id:
                continue
            latest[task_id] = row
    return latest


def should_rerun_output(output: JsonRecord, *, rerun_errors: bool) -> bool:
    if not rerun_errors:
        return False
    if output.get("error"):
        return True
    if output.get("parsed_output") is None:
        return True
    return False


def compact_model_outputs(path: str | Path) -> list[JsonRecord]:
    path = Path(path)
    latest = latest_outputs_by_task(path)
    rows = sorted(latest.values(), key=lambda row: (_as_int(row.get("row_index", 0)), str(row.get("task_id", ""))))
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp_path.replace(path)
    return rows


def write_progress(
    path: str | Path,
    *,
    total: int,
    skipped: int,
    pending: int,
    completed_this_run: int,
    started_at: float,
) -> None:
    progress = {
        "total": total,
        "skipped": skipped,
        "pending": pending,
        "completed_this_run": completed_this_run,
        "remaining_this_run": max(pending - completed_this_run, 0),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    Path(path).write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def write_final_cases(split_path: str | Path, merged_path: str | Path, output_path: str | Path) -> JsonRecord:
    cases = _read_jsonl(split_path)
    merged_rows = _read_jsonl(merged_path)
    merged_by_case: dict[str, dict[int, JsonRecord]] = {}
    for row in merged_rows:
        case_id = str(row.get("case_id", ""))
        defendant_index = int(row.get("defendant_index", -1))
        merged_by_case.setdefault(case_id, {})[defendant_index] = row

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    defendants_written = 0
    missing_groundings = 0
    with output_path.open("w", encoding="utf-8") as target:
        for case in cases:
            output_case = dict(case)
            output_defendants = []
            case_groundings = merged_by_case.get(str(case.get("case_id", "")), {})
            defendants = case.get("defendants", [])
            if not isinstance(defendants, list):
                defendants = []
            for defendant_index, defendant in enumerate(defendants):
                output_defendant = dict(defendant) if isinstance(defendant, dict) else {"name": str(defendant)}
                output_defendant["defendant_index"] = defendant_index
                grounding = case_groundings.get(defendant_index)
                if grounding is None:
                    output_defendant["grounding_links"] = []
                    output_defendant["warnings"] = [{"type": "missing_grounding"}]
                    missing_groundings += 1
                else:
                    output_defendant["grounding_links"] = grounding.get("grounding_links", [])
                    output_defendant["warnings"] = grounding.get("warnings", [])
                output_defendants.append(output_defendant)
                defendants_written += 1
            output_case["defendants"] = output_defendants
            target.write(json.dumps(output_case, ensure_ascii=False, separators=(",", ":")) + "\n")

    return {
        "cases_read": len(cases),
        "cases_written": len(cases),
        "defendants_written": defendants_written,
        "missing_groundings": missing_groundings,
        "output": str(output_path),
    }


def count_final_warnings(path: str | Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for case in _read_jsonl(path):
        defendants = case.get("defendants", [])
        if not isinstance(defendants, list):
            continue
        for defendant in defendants:
            if not isinstance(defendant, dict):
                continue
            warnings = defendant.get("warnings", [])
            if not isinstance(warnings, list):
                continue
            for warning in warnings:
                if isinstance(warning, dict):
                    counts[str(warning.get("type", "unknown"))] += 1
    return counts


def aggregate_usage(results: Any) -> JsonRecord:
    fields = [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    ]
    usage = {field: 0 for field in fields}
    for result in results:
        row_usage = result.get("usage", {})
        if not isinstance(row_usage, dict):
            continue
        for field in fields:
            usage[field] += _as_int(row_usage.get(field, 0))
    cache_total = usage["prompt_cache_hit_tokens"] + usage["prompt_cache_miss_tokens"]
    usage["cache_hit_rate"] = round(usage["prompt_cache_hit_tokens"] / cache_total, 6) if cache_total else None
    return usage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an offset/limit case grounding batch.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--law-map", type=Path, default=DEFAULT_LAW_MAP)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset-cases", type=int, required=True)
    parser.add_argument("--limit-cases", type=int, required=True)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--rerun-errors", action="store_true", help="Rerun existing outputs with API errors or unparsed JSON.")
    parser.add_argument("--cleanup-tmp", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.offset_cases < 0 or args.limit_cases < 1:
        raise SystemExit("offset-cases must be >= 0 and limit-cases must be >= 1")
    summary = run_batch(
        args.source,
        args.law_map,
        args.output_dir,
        offset_cases=args.offset_cases,
        limit_cases=args.limit_cases,
        env_file=args.env_file,
        max_workers=args.max_workers,
        cleanup_tmp=args.cleanup_tmp,
        rerun_errors=args.rerun_errors,
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


def _env_value(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _as_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
