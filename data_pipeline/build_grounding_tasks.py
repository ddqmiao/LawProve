"""Build defendant-level grounding tasks and readable prompts without calling a model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


JsonRecord = dict[str, Any]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "outputs" / "train_small_split.jsonl"
DEFAULT_LAW_MAP = SCRIPT_DIR / "law_article_map.criminal_law.json"
DEFAULT_TASKS_OUTPUT = SCRIPT_DIR / "outputs" / "grounding_tasks.jsonl"
DEFAULT_PROMPTS_OUTPUT = SCRIPT_DIR / "outputs" / "grounding_prompts.jsonl"

DOT_SUFFIX_NUMERALS = {
    1: "一",
    2: "二",
    3: "三",
    4: "四",
    5: "五",
    6: "六",
    7: "七",
    8: "八",
    9: "九",
    10: "十",
    11: "十一",
    12: "十二",
    13: "十三",
    14: "十四",
    15: "十五",
    16: "十六",
    17: "十七",
    18: "十八",
    19: "十九",
    20: "二十",
}


def load_law_map(path: str | Path) -> dict[str, JsonRecord]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("law map must be a JSON object")
    return {str(key): _validate_law_unit(str(key), unit) for key, unit in value.items()}


def normalize_article_code(article: Any) -> str:
    article_id = str(article).strip()
    match = re.fullmatch(r"(\d+)\.(\d+)(-\d+)?", article_id)
    if not match:
        return article_id

    suffix_number = int(match.group(2))
    suffix_text = DOT_SUFFIX_NUMERALS.get(suffix_number)
    if suffix_text is None:
        return article_id
    paragraph_suffix = match.group(3) or ""
    return f"{match.group(1)}之{suffix_text}{paragraph_suffix}"


def build_law_candidates(
    defendant_articles: Iterable[Any],
    relevant_articles: Iterable[Any],
    law_map: dict[str, JsonRecord],
) -> tuple[list[JsonRecord], list[str]]:
    """Build defendant-level law candidates from outcome articles then case articles."""

    source_by_resolved: dict[str, list[str]] = {}
    ordered_resolved: list[str] = []
    missing: list[str] = []

    for article in [*defendant_articles, *relevant_articles]:
        original = str(article).strip()
        if not original:
            continue
        law_unit = _resolve_law_unit(original, law_map)
        if law_unit is None:
            if original not in missing:
                missing.append(original)
            continue

        resolved_article = str(law_unit["article"])
        if resolved_article not in source_by_resolved:
            ordered_resolved.append(resolved_article)
            source_by_resolved[resolved_article] = []
        if original not in source_by_resolved[resolved_article]:
            source_by_resolved[resolved_article].append(original)

    candidates = []
    for article in ordered_resolved:
        law_unit = _resolve_law_unit(article, law_map)
        if law_unit is None:
            continue
        candidates.append(
            {
                "article": str(law_unit["article"]),
                "text": str(law_unit["text"]),
                "source_articles": source_by_resolved[article],
            }
        )
    return candidates, missing


def build_grounding_tasks(case: JsonRecord, law_map: dict[str, JsonRecord], *, case_id: str) -> list[JsonRecord]:
    """Create one defendant-level task per defendant in a sentence-split case."""

    defendants = case.get("defendants", [])
    if not isinstance(defendants, list):
        raise ValueError(f"{case_id}: defendants must be a list")

    tasks = []
    for defendant_index, defendant_name in enumerate(defendants):
        name = str(defendant_name).strip()
        if not name:
            continue
        outcome = _find_outcome(case.get("outcomes", []), name, defendant_index)
        defendant_articles = _judgment_articles(outcome)
        law_candidates, missing_law_articles = build_law_candidates(
            defendant_articles,
            case.get("relevant_articles", []),
            law_map,
        )
        tasks.append(
            {
                "task_id": f"{case_id}__D{defendant_index}",
                "case_id": case_id,
                "defendant_index": defendant_index,
                "defendant_name": name,
                "target_outcome": outcome,
                "fact_units": list(case.get("fact", [])),
                "court_view_units": list(case.get("court_view", [])),
                "law_candidates": law_candidates,
                "missing_law_articles": missing_law_articles,
            }
        )
    return tasks


def render_prompt(task: JsonRecord) -> str:
    """Render a simple text prompt for one defendant-level grounding task."""

    output_schema = {
        "grounding_links": [
            {
                "court_unit_id": "CV0",
                "fact_ids": ["F0"],
                "law_ids": ["264"],
            }
        ]
    }
    empty_output_schema = {"grounding_links": []}
    sections = [
        "你是法律裁判文书细粒度溯源标注助手。",
        "",
        "任务：",
        "给定一个刑事案件中的目标被告人，请只为该目标被告人选择相关的法院观点、支撑事实和法条编号。",
        "",
        "模型只负责选择编号，不负责改写文本、补充事实或生成解释。",
        "所有输出编号都必须来自候选列表，不得编造。",
        "",
        "【标注目标】",
        "",
        "输出目标被告人的 grounding_links。",
        "",
        "每个 grounding_link 表示：",
        "",
        "* court_unit_id：法院对目标被告人的一条实质性说理；",
        "* fact_ids：直接支持该法院说理的事实依据；",
        "* law_ids：直接支持该法院说理的法条依据。",
        "",
        "【court_unit_id 选择规则】",
        "",
        "1. 只选择与目标被告人直接相关的法院观点。",
        "2. 如果法院观点列出多个被告人，且包含目标被告人，可以选择。",
        "3. 如果法院观点适用于全体被告人，且目标被告人在其中，可以选择。",
        "4. 如果法院观点只涉及其他被告人，不要选择。",
        "5. 一般不要选择纯程序性或总结性表述，例如“据此”“本院予以支持”“量刑建议予以采纳”等，除非该句包含实质性定罪、量刑或处罚理由。",
        "6. 内容不完整、无法判断含义的法院观点，不要选择。",
        "",
        "【fact_ids 选择规则】",
        "",
        "1. fact_ids 必须直接支持该 court_unit_id 对目标被告人的说理。",
        "2. fact_ids 不是机械追求最少，应选择直接、充分、不过度重复的事实；避免堆砌间接或重复证据。",
        "3. 如果法院观点涉及目标被告人的具体行为、角色、分工、作用、金额、时间、规模、认罪态度、自首、坦白、退赔、谅解、缓刑等关键细节，应选择能够支撑这些细节的事实。",
        "4. 可以同时选择概括性事实和关键细节事实，但不要堆砌大量重复证据。",
        "5. 优先选择法院认定或采信的事实，例如“经审理查明”“上述事实经证实”“经法庭质证、认证的证据证实”等。",
        "6. 如果法院认定事实不够具体，可以补充选择被告人供述、证人证言、辨认笔录、书证等具体证据。",
        "7. 一般不要选择公诉机关指控、量刑建议、辩护意见，除非没有更直接的事实或证据材料。",
        "8. 不要把其他被告人的个人行为、供述、角色事实作为目标被告人的依据。",
        "9. 必须精确区分姓名相近的被告人，例如“陈某”和“陈某某”、“杨某甲”和“杨某乙”。",
        "10. 如果没有明确事实依据，可以让 fact_ids 为空数组。",
        "",
        "【law_ids 选择规则】",
        "",
        "1. law_ids 只能从当前案件的法条候选中选择。",
        "2. 不要从候选之外新增法条编号。",
        "3. law_ids 必须是该 court_unit_id 的直接法律依据。",
        "4. 不要把本案所有法条都放入同一个 grounding_link。",
        "5. 定罪说理选择罪名法条；共同犯罪、主犯、从犯、自首、坦白、累犯、缓刑、罚金、违法所得处理等说理，选择对应法条。",
        "6. 如果没有明确对应法条，可以让 law_ids 为空数组。",
        "",
        "【输出要求】",
        "",
        "只输出一个 JSON 对象。",
        "不要输出 Markdown。",
        "不要输出解释。",
        "不要输出原文。",
        "",
        "JSON 格式如下：",
        "",
        json.dumps(output_schema, ensure_ascii=False, indent=2),
        "",
        "要求：",
        "",
        "1. court_unit_id 必须来自法院观点候选。",
        "2. fact_ids 必须来自事实候选。",
        "3. law_ids 必须来自法条候选。",
        "4. fact_ids 和 law_ids 可以为空数组。",
        "5. grounding_links 不要重复。",
        "6. grounding_links 按法院观点候选顺序排列。",
        "7. 如果没有相关实质性法院观点，输出：",
        json.dumps(empty_output_schema, ensure_ascii=False, indent=2),
        "",
        "【目标被告人】",
        "",
        f"姓名：{task['defendant_name']}",
        "",
        _format_outcome(task.get("target_outcome", {})),
        "",
        "【法院观点候选】",
        "",
        _format_units(task.get("court_view_units", [])),
        "",
        "【事实候选】",
        "",
        _format_units(task.get("fact_units", [])),
        "",
        "【法条候选】",
        "",
        _format_law_candidates(task.get("law_candidates", [])),
    ]
    return "\n".join(section for section in sections if section is not None)


def process_file(
    input_path: str | Path,
    law_map_path: str | Path,
    tasks_output_path: str | Path,
    prompts_output_path: str | Path,
    *,
    case_prefix: str = "train",
    limit_cases: int | None = None,
) -> JsonRecord:
    """Write structured tasks and readable prompts for sentence-split cases."""

    law_map = load_law_map(law_map_path)
    input_path = Path(input_path)
    tasks_output_path = Path(tasks_output_path)
    prompts_output_path = Path(prompts_output_path)
    tasks_output_path.parent.mkdir(parents=True, exist_ok=True)
    prompts_output_path.parent.mkdir(parents=True, exist_ok=True)

    cases_read = 0
    tasks_written = 0
    prompts_written = 0
    with input_path.open("r", encoding="utf-8") as source, tasks_output_path.open(
        "w", encoding="utf-8"
    ) as task_target, prompts_output_path.open("w", encoding="utf-8") as prompt_target:
        for line_index, line in enumerate(source):
            if limit_cases is not None and cases_read >= limit_cases:
                break
            if not line.strip():
                continue

            case = json.loads(line)
            if not isinstance(case, dict):
                raise ValueError(f"line {line_index + 1}: expected JSON object")
            case_id = str(case.get("case_id") or f"{case_prefix}_{line_index:06d}")
            cases_read += 1

            for task in build_grounding_tasks(case, law_map, case_id=case_id):
                prompt = render_prompt(task)
                task_target.write(json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n")
                prompt_target.write(
                    json.dumps(
                        {
                            "task_id": task["task_id"],
                            "case_id": task["case_id"],
                            "defendant_index": task["defendant_index"],
                            "defendant_name": task["defendant_name"],
                            "prompt": prompt,
                            "prompt_char_count": len(prompt),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                tasks_written += 1
                prompts_written += 1

    return {
        "cases_read": cases_read,
        "tasks_written": tasks_written,
        "prompts_written": prompts_written,
        "tasks_output": str(tasks_output_path),
        "prompts_output": str(prompts_output_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build defendant-level grounding tasks and prompts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Sentence-split JSONL. Default: {DEFAULT_INPUT}")
    parser.add_argument("--law-map", type=Path, default=DEFAULT_LAW_MAP, help=f"Law map JSON. Default: {DEFAULT_LAW_MAP}")
    parser.add_argument(
        "--tasks-output",
        type=Path,
        default=DEFAULT_TASKS_OUTPUT,
        help=f"Structured task JSONL output. Default: {DEFAULT_TASKS_OUTPUT}",
    )
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=DEFAULT_PROMPTS_OUTPUT,
        help=f"Readable prompt JSONL output. Default: {DEFAULT_PROMPTS_OUTPUT}",
    )
    parser.add_argument("--case-prefix", default="train", help="Prefix for generated case IDs when input lacks case_id.")
    parser.add_argument("--limit-cases", type=int, default=None, help="Optional number of cases to process.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = process_file(
        args.input,
        args.law_map,
        args.tasks_output,
        args.prompts_output,
        case_prefix=args.case_prefix,
        limit_cases=args.limit_cases,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _validate_law_unit(article_id: str, value: Any) -> JsonRecord:
    if not isinstance(value, dict):
        raise ValueError(f"law map entry for {article_id} must be an object")
    article = str(value.get("article", "")).strip()
    text = str(value.get("text", "")).strip()
    if not article or not text:
        raise ValueError(f"law map entry for {article_id} must contain article and text")
    return {"article": article, "text": text}


def _resolve_law_unit(article: Any, law_map: dict[str, JsonRecord]) -> JsonRecord | None:
    article_id = str(article).strip()
    normalized = normalize_article_code(article_id)
    candidates = [article_id]
    if normalized != article_id:
        candidates.append(normalized)
    if "-" in normalized:
        candidates.append(normalized.rsplit("-", 1)[0])

    for candidate in candidates:
        law_unit = law_map.get(candidate)
        if law_unit is not None:
            return law_unit
    return None


def _find_outcome(outcomes: Any, defendant_name: str, defendant_index: int) -> JsonRecord:
    if not isinstance(outcomes, list):
        return {}
    for outcome in outcomes:
        if isinstance(outcome, dict) and str(outcome.get("name", "")).strip() == defendant_name:
            return dict(outcome)
    if defendant_index < len(outcomes) and isinstance(outcomes[defendant_index], dict):
        return dict(outcomes[defendant_index])
    return {}


def _judgment_articles(outcome: JsonRecord) -> list[str]:
    articles: list[str] = []
    judgments = outcome.get("judgment", [])
    if not isinstance(judgments, list):
        return articles
    for judgment in judgments:
        if not isinstance(judgment, dict):
            continue
        for article in judgment.get("article", []):
            article_id = str(article).strip()
            if article_id:
                articles.append(article_id)
    return articles


def _format_outcome(outcome: JsonRecord) -> str:
    judgments = outcome.get("judgment", [])
    if not isinstance(judgments, list) or not judgments:
        return "判决信息：无"

    lines = ["判决信息："]
    for index, judgment in enumerate(judgments, start=1):
        if not isinstance(judgment, dict):
            continue
        accusation = judgment.get("standard_accusation") or judgment.get("accusation") or "未知罪名"
        articles = "、".join(str(article) for article in judgment.get("article", [])) or "无"
        penalty = _format_penalty(judgment.get("penalty", {}))
        lines.append(f"{index}. 罪名：{accusation}；适用法条：{articles}；刑罚：{penalty}")
    return "\n".join(lines)


def _format_penalty(penalty: Any) -> str:
    if not isinstance(penalty, dict):
        return "无"
    parts = []
    if penalty.get("death_penalty"):
        parts.append("死刑")
    if penalty.get("life_imprisonment"):
        parts.append("无期徒刑")
    if penalty.get("imprisonment"):
        parts.append(f"有期徒刑{penalty['imprisonment']}个月")
    if penalty.get("detention"):
        parts.append(f"拘役{penalty['detention']}个月")
    if penalty.get("surveillance"):
        parts.append(f"管制{penalty['surveillance']}个月")
    if penalty.get("fine"):
        parts.append(f"罚金{penalty['fine']}元")
    if penalty.get("fine_without_amount"):
        parts.append("罚金")
    return "；".join(parts) if parts else "无"


def _format_units(units: Any) -> str:
    if not isinstance(units, list) or not units:
        return "无"
    lines = []
    for unit in units:
        if isinstance(unit, dict):
            unit_id = str(unit.get("id", "")).strip()
            text = str(unit.get("text", "")).strip()
            if unit_id and text:
                lines.append(f"{unit_id} {text}")
    return "\n".join(lines) if lines else "无"


def _format_law_candidates(candidates: Any) -> str:
    if not isinstance(candidates, list) or not candidates:
        return "无"
    lines = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        article = str(candidate.get("article", "")).strip()
        text = str(candidate.get("text", "")).strip()
        if article and text:
            lines.append(f"{article} {text}")
    return "\n".join(lines) if lines else "无"


if __name__ == "__main__":
    main()
