"""Merge court-view-level grounding outputs into defendant-level links."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from build_grounding_tasks import JsonRecord, normalize_article_code


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TASKS_INPUT = SCRIPT_DIR / "outputs" / "cv_grounding_tasks.jsonl"
DEFAULT_MODEL_OUTPUTS_INPUT = SCRIPT_DIR / "outputs" / "cv_grounding_model_outputs.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "cv_grounding_merged.jsonl"


def merge_outputs(tasks: list[JsonRecord], model_outputs: list[JsonRecord]) -> list[JsonRecord]:
    """Merge parsed CV-level model outputs into one row per defendant."""

    tasks_by_id = {str(task.get("task_id")): task for task in tasks}
    groups: dict[tuple[str, int], JsonRecord] = {}
    links_by_group: dict[tuple[str, int], dict[str, JsonRecord]] = {}
    warnings_by_group: dict[tuple[str, int], list[JsonRecord]] = {}

    for output in model_outputs:
        task_id = str(output.get("task_id", ""))
        task = tasks_by_id.get(task_id)
        if task is None:
            continue
        group_key = (str(task.get("case_id", "")), int(task.get("defendant_index", -1)))
        groups.setdefault(
            group_key,
            {
                "case_id": task.get("case_id"),
                "defendant_index": task.get("defendant_index"),
                "defendant_name": task.get("defendant_name"),
            },
        )
        links_by_group.setdefault(group_key, {})
        warnings_by_group.setdefault(group_key, [])

        if output.get("error"):
            warnings_by_group[group_key].append({"task_id": task_id, "type": "model_error", "value": str(output["error"])})
            continue

        for link in _extract_links(output.get("parsed_output")):
            normalized_link, warnings = _normalize_link(task, link)
            warnings_by_group[group_key].extend({"task_id": task_id, **warning} for warning in warnings)
            if normalized_link is None:
                continue
            court_id = normalized_link["court_unit_id"]
            existing = links_by_group[group_key].get(court_id)
            if existing is None:
                links_by_group[group_key][court_id] = normalized_link
            else:
                existing["fact_ids"] = _ordered_union(
                    [*existing["fact_ids"], *normalized_link["fact_ids"]],
                    _candidate_unit_ids(task.get("fact_candidates", [])),
                )
                existing["law_ids"] = _ordered_union(
                    [*existing["law_ids"], *normalized_link["law_ids"]],
                    _candidate_law_ids(task.get("law_candidates", [])),
                )

    rows: list[JsonRecord] = []
    for group_key in sorted(groups, key=lambda item: (item[0], item[1])):
        group = groups[group_key]
        links = list(links_by_group.get(group_key, {}).values())
        links.sort(key=lambda link: _court_sort_key(str(link.get("court_unit_id", ""))))
        rows.append(
            {
                "case_id": group.get("case_id"),
                "defendant_index": group.get("defendant_index"),
                "defendant_name": group.get("defendant_name"),
                "grounding_links": links,
                "warnings": warnings_by_group.get(group_key, []),
            }
        )
    return rows


def process_file(
    tasks_input_path: str | Path,
    model_outputs_input_path: str | Path,
    output_path: str | Path,
) -> JsonRecord:
    tasks = _read_jsonl(tasks_input_path)
    model_outputs = _read_jsonl(model_outputs_input_path)
    rows = merge_outputs(tasks, model_outputs)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "tasks_read": len(tasks),
        "model_outputs_read": len(model_outputs),
        "defendants_written": len(rows),
        "output": str(output_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge CV-level grounding outputs into defendant-level rows.")
    parser.add_argument("--tasks-input", type=Path, default=DEFAULT_TASKS_INPUT)
    parser.add_argument("--model-outputs-input", type=Path, default=DEFAULT_MODEL_OUTPUTS_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = process_file(args.tasks_input, args.model_outputs_input, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _extract_links(parsed_output: Any) -> list[JsonRecord]:
    if not isinstance(parsed_output, dict):
        return []
    if isinstance(parsed_output.get("grounding_link"), dict):
        return [parsed_output["grounding_link"]]
    if isinstance(parsed_output.get("grounding_links"), list):
        return [link for link in parsed_output["grounding_links"] if isinstance(link, dict)]
    return []


def _normalize_link(task: JsonRecord, link: JsonRecord) -> tuple[JsonRecord | None, list[JsonRecord]]:
    warnings: list[JsonRecord] = []
    court_id = str(task.get("court_unit", {}).get("id", "")).strip()
    output_court_id = str(link.get("court_unit_id") or court_id).strip()
    if output_court_id != court_id:
        return None, [{"type": "bad_court", "value": output_court_id}]

    candidate_fact_ids = _candidate_unit_ids(task.get("fact_candidates", []))
    candidate_law_ids = _candidate_law_ids(task.get("law_candidates", []))
    fact_ids: list[str] = []
    law_ids: list[str] = []

    for fact_id in _as_string_list(link.get("fact_ids", [])):
        if fact_id not in candidate_fact_ids:
            warnings.append({"type": "bad_fact", "value": fact_id})
            continue
        fact_ids.append(fact_id)

    law_lookup = {normalize_article_code(article): article for article in candidate_law_ids}
    for law_id in _as_string_list(link.get("law_ids", [])):
        normalized_law_id = normalize_article_code(law_id)
        if law_id in candidate_law_ids:
            law_ids.append(law_id)
        elif normalized_law_id in law_lookup:
            law_ids.append(law_lookup[normalized_law_id])
        else:
            warnings.append({"type": "bad_law", "value": law_id})

    return (
        {
            "court_unit_id": court_id,
            "fact_ids": _ordered_union(fact_ids, candidate_fact_ids),
            "law_ids": _ordered_union(law_ids, candidate_law_ids),
        },
        warnings,
    )


def _candidate_unit_ids(units: Any) -> list[str]:
    if not isinstance(units, list):
        return []
    return [str(unit.get("id")) for unit in units if isinstance(unit, dict) and unit.get("id")]


def _candidate_law_ids(candidates: Any) -> list[str]:
    if not isinstance(candidates, list):
        return []
    return [str(candidate.get("article")) for candidate in candidates if isinstance(candidate, dict) and candidate.get("article")]


def _ordered_union(values: list[str], order: list[str]) -> list[str]:
    seen = {value for value in values if value}
    return [value for value in order if value in seen]


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _court_sort_key(court_id: str) -> tuple[int, str]:
    match = re.fullmatch(r"CV(\d+)", court_id)
    if match:
        return (int(match.group(1)), court_id)
    return (10**9, court_id)


def _read_jsonl(path: str | Path) -> list[JsonRecord]:
    rows: list[JsonRecord] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected JSON object")
            rows.append(row)
    return rows


if __name__ == "__main__":
    main()
