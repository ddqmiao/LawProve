"""Build court-view-level grounding tasks with rule-based candidate filtering."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from build_grounding_tasks import (
    JsonRecord,
    _find_outcome,
    _format_law_candidates,
    _format_outcome,
    _format_units,
    _judgment_articles,
    build_law_candidates,
    load_law_map,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "outputs" / "train_small_split.jsonl"
DEFAULT_LAW_MAP = SCRIPT_DIR / "law_article_map.criminal_law.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "cv_grounding_tasks.jsonl"
DEFAULT_PROMPTS_OUTPUT = SCRIPT_DIR / "outputs" / "cv_grounding_prompts.jsonl"

CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

SUBSTANTIVE_COURT_KEYWORDS = [
    "构成",
    "触犯",
    "定罪",
    "主犯",
    "从犯",
    "共同犯罪",
    "从轻",
    "从重",
    "减轻",
    "免除",
    "坦白",
    "自首",
    "累犯",
    "缓刑",
    "罚金",
    "退赔",
    "谅解",
    "认罪",
    "认罚",
    "犯罪事实",
    "量刑",
    "处罚",
    "没收",
    "追缴",
]

FACT_KEYWORDS = [
    "经审理查明",
    "上述事实",
    "证实",
    "证明",
    "供述",
    "证言",
    "辨认",
    "书证",
    "质证",
    "认证",
    "盗窃",
    "抢劫",
    "诈骗",
    "开设赌场",
    "非法拘禁",
    "共同",
    "主犯",
    "从犯",
    "望风",
    "帮助",
    "如实供述",
    "坦白",
    "自首",
    "累犯",
    "退赔",
    "谅解",
    "认罪",
    "认罚",
    "罚金",
    "缓刑",
]

GROUP_COURT_PATTERNS = [
    "各被告",
    "全体被告",
    "二被告",
    "两被告",
    "三被告",
    "四被告",
    "五被告",
    "六被告",
    "七被告",
    "八被告",
    "九被告",
    "十被告",
    "被告二人",
    "被告三人",
    "被告四人",
    "被告五人",
    "被告六人",
    "被告七人",
    "被告八人",
    "被告九人",
    "被告十人",
    "均能",
    "均已",
    "均可",
    "均自愿",
]

SUMMARY_ONLY_PATTERNS = [
    "据此",
    "综上",
    "本院予以支持",
    "本院予以采纳",
    "予以采纳",
    "予以确认",
]

NEGATIVE_FACT_PATTERNS = [
    "无关",
    "另案处理",
]


def extract_law_articles_from_text(text: str) -> list[str]:
    """Extract article ids such as 26-1 from Chinese law references in text."""

    results: list[str] = []
    seen: set[str] = set()
    pattern = re.compile(r"第([零〇一二两三四五六七八九十百千]+)条(?:第([零〇一二两三四五六七八九十百千]+)款)?")
    for match in pattern.finditer(text):
        article_number = _parse_chinese_number(match.group(1))
        if article_number is None:
            continue
        paragraph_number = _parse_chinese_number(match.group(2)) if match.group(2) else None
        article_id = str(article_number)
        if paragraph_number is not None:
            article_id = f"{article_id}-{paragraph_number}"
        if article_id not in seen:
            results.append(article_id)
            seen.add(article_id)
    return results


def select_defendant_court_units(case: JsonRecord, defendant_name: str) -> list[JsonRecord]:
    """Select court_view units that are likely relevant to one defendant."""

    court_units = _list_units(case.get("court_view", []))
    defendants = [str(name).strip() for name in case.get("defendants", []) if str(name).strip()]
    selected: list[JsonRecord] = []
    for unit in court_units:
        text = str(unit.get("text", "")).strip()
        if not text or _is_incomplete_court_view(text):
            continue
        if _is_summary_only_court_view(text):
            continue

        mentioned_names = _mentioned_defendants(text, defendants)
        if defendant_name in mentioned_names:
            selected.append(unit)
            continue
        if mentioned_names and defendant_name not in mentioned_names:
            continue
        if _is_group_court_view(text) and _has_substantive_court_content(text):
            selected.append(unit)
    return selected


def select_fact_candidates_for_court_unit(
    fact_units: Iterable[Any],
    court_unit: JsonRecord,
    *,
    defendant_name: str,
    defendants: Iterable[Any],
    max_fact_candidates: int = 50,
    context_window: int = 1,
) -> list[JsonRecord]:
    """Filter fact units for a single defendant and court_view unit."""

    facts = _list_units(fact_units)
    defendant_names = [str(name).strip() for name in defendants if str(name).strip()]
    court_text = str(court_unit.get("text", ""))
    court_keywords = _matched_keywords(court_text, SUBSTANTIVE_COURT_KEYWORDS + FACT_KEYWORDS)
    scored: dict[int, int] = {}

    for index, fact in enumerate(facts):
        text = str(fact.get("text", "")).strip()
        if not text:
            continue
        if _is_negative_other_fact(text, defendant_name):
            continue
        mentioned_names = _mentioned_defendants(text, defendant_names)
        if mentioned_names and defendant_name not in mentioned_names and not _allows_group_fact(court_text, text):
            continue

        score = 0
        if _contains_defendant_name(text, defendant_name):
            score += 8
        if _allows_group_fact(court_text, text):
            score += 4
        if any(keyword in text for keyword in court_keywords):
            score += 3
        if any(keyword in text for keyword in FACT_KEYWORDS):
            score += 2
        if "证实" in text or "证明" in text or "上述事实" in text:
            score += 3

        if score > 0:
            scored[index] = score

    candidate_indexes = set(scored)
    for index in list(scored):
        for nearby in range(index - context_window, index + context_window + 1):
            if 0 <= nearby < len(facts):
                candidate_indexes.add(nearby)
                scored.setdefault(nearby, 1)

    ranked_indexes = sorted(candidate_indexes, key=lambda item: (-scored.get(item, 0), item))
    if max_fact_candidates > 0:
        ranked_indexes = ranked_indexes[:max_fact_candidates]
    return [facts[index] for index in sorted(ranked_indexes)]


def build_cv_grounding_tasks(
    case: JsonRecord,
    law_map: dict[str, JsonRecord],
    *,
    case_id: str,
    max_fact_candidates: int = 50,
    context_window: int = 1,
    fact_candidates_mode: str = "filtered",
) -> list[JsonRecord]:
    """Create one task per selected defendant/court_view pair."""

    defendants = case.get("defendants", [])
    if not isinstance(defendants, list):
        raise ValueError(f"{case_id}: defendants must be a list")
    if fact_candidates_mode not in {"filtered", "all"}:
        raise ValueError("fact_candidates_mode must be 'filtered' or 'all'")

    tasks: list[JsonRecord] = []
    all_court_laws = _extract_law_articles_from_units(case.get("court_view", []))
    all_fact_candidates = _list_units(case.get("fact", []))
    for defendant_index, raw_name in enumerate(defendants):
        defendant_name = str(raw_name).strip()
        if not defendant_name:
            continue
        outcome = _find_outcome(case.get("outcomes", []), defendant_name, defendant_index)
        defendant_articles = _judgment_articles(outcome)
        selected_courts = select_defendant_court_units(case, defendant_name)
        for court_unit in selected_courts:
            court_laws = extract_law_articles_from_text(str(court_unit.get("text", "")))
            law_candidates, missing_law_articles = build_law_candidates(
                defendant_articles,
                [*case.get("relevant_articles", []), *all_court_laws, *court_laws],
                law_map,
            )
            if fact_candidates_mode == "all":
                fact_candidates = all_fact_candidates
            else:
                fact_candidates = select_fact_candidates_for_court_unit(
                    case.get("fact", []),
                    court_unit,
                    defendant_name=defendant_name,
                    defendants=defendants,
                    max_fact_candidates=max_fact_candidates,
                    context_window=context_window,
                )
            tasks.append(
                {
                    "task_id": f"{case_id}__D{defendant_index}__{court_unit['id']}",
                    "case_id": case_id,
                    "defendant_index": defendant_index,
                    "defendant_name": defendant_name,
                    "target_outcome": outcome,
                    "court_unit": court_unit,
                    "fact_candidates": fact_candidates,
                    "fact_candidates_mode": fact_candidates_mode,
                    "law_candidates": law_candidates,
                    "missing_law_articles": missing_law_articles,
                }
            )
    return tasks


def render_cv_system_prompt() -> str:
    """Render static grounding rules shared by all CV-level tasks."""

    output_schema = {
        "grounding_link": {
            "court_unit_id": "CV_ID",
            "fact_ids": ["F0"],
            "law_ids": ["264"],
        }
    }
    sections = [
        "你是法律裁判文书细粒度溯源标注助手。",
        "",
        "任务：",
        "给定一个刑事案件中的目标被告人和一个法院观点，请只为该法院观点选择直接支撑事实和法条编号。",
        "每次只处理一个法院观点。模型只负责选择编号，不负责改写文本、补充事实或生成解释。",
        "所有输出编号都必须来自候选列表，不得编造。",
        "",
        "【标注目标】",
        "",
        "输出一个 grounding_link。",
        "",
        "每个 grounding_link 表示：",
        "",
        "* court_unit_id：当前法院观点编号；",
        "* fact_ids：直接支持当前法院观点对目标被告人的说理的事实依据；",
        "* law_ids：直接支持当前法院观点对目标被告人的说理的法条依据。",
        "",
        "【选择规则】",
        "",
        "1. court_unit_id 必须等于当前法院观点编号（见当前任务数据）。",
        "2. fact_ids 必须直接支持当前法院观点，不要选择只涉及其他被告人的事实。",
        "3. fact_ids 不要求最少，也不要机械多选；应选择与当前法院观点直接相关、必要且不重复的事实。",
        "4. 如果一条法院观点同时涉及多个关键点，例如行为、角色、金额、认罪态度、退赔、谅解、缓刑等，可以选择多条事实分别支撑这些关键点。",
        "5. 如果已有概括性事实能够支持法院观点，但缺少关键细节，可以补充选择具体证据或细节事实。",
        "6. 不要选择与当前法院观点关系较远的背景事实、重复证据，或只能间接支持该观点的事实。",
        "7. 如果当前法院观点主要讨论认罪、坦白、退赔、谅解、累犯、缓刑等量刑情节，应优先选择支撑这些情节的事实；基础犯罪事实可以选择，但应与该观点存在直接关系。",
        "8. 优先选择法院认定或采信的事实，例如“经审理查明”“上述事实经证实”“经法庭质证、认证的证据证实”等。",
        "9. 如果法院认定事实不够具体，可以补充选择被告人供述、证人证言、辨认笔录、书证等具体证据。",
        "10. 一般不要选择公诉机关指控、量刑建议、辩护意见，除非没有更直接的事实或证据材料。",
        "11. 必须精确区分姓名相近的被告人，例如“陈某”和“陈某某”、“杨某甲”和“杨某乙”。",
        "12. law_ids 只能从当前法条候选中选择，且必须是当前法院观点的直接法律依据。",
        "13. 不要把本案所有法条都放入同一个 grounding_link；定罪、主犯、从犯、坦白、累犯、缓刑、罚金等分别选择直接对应法条。",
        "14. 如果没有明确事实依据或法条依据，对应数组可以为空。",
        "",
        "【宽泛法院观点处理规则】",
        "",
        "1. 如果当前法院观点只是“指控事实成立”“公诉机关指控成立”“本院予以确认”等概括性总结，且没有具体定罪、量刑或处罚理由，不要展开选择全部犯罪事实；优先选择与目标被告人直接相关的1-3条概括性事实或核心事实。如果事实候选中没有这类直接相关事实，再让 fact_ids 为空。",
        "2. 如果当前法院观点是宽泛定罪句，例如同时概括行为性质、次数、金额、共同犯罪和罪名，不要选择所有犯罪事实和所有证据。",
        "3. 对宽泛定罪句，应优先选择能够覆盖犯罪构成要件的事实：行为方式、关键次数、关键金额或损失、共同犯罪关系、目标被告人参与情况。",
        "4. 如果事实候选中已有直接概括目标被告人作案次数、总金额、总价值或参与情况的事实句，应优先选择该概括性事实，不要再逐条选择全部单次犯罪事实。",
        "5. 多次同类犯罪不需要逐条选择全部细节；选择能体现“多次”“数额较大”“情节严重”等判断的代表性事实或概括性事实即可。",
        "6. 不要选择大量证据目录、证人证言、辨认笔录等重复证明材料，除非事实认定句不足以支撑当前法院观点。",
        "",
        "【输出要求】",
        "",
        "只输出一个 JSON 对象。",
        "不要输出 Markdown。",
        "不要输出解释。",
        "不要输出原文。",
        "",
        "JSON 格式如下，示例编号不代表答案：",
        "",
        json.dumps(output_schema, ensure_ascii=False, indent=2),
        "",
        "要求：",
        "",
        "1. court_unit_id 必须等于当前法院观点编号（见当前任务数据）。",
        "2. fact_ids 必须来自事实候选。",
        "3. law_ids 必须来自法条候选。",
        "4. fact_ids 和 law_ids 可以为空数组。",
    ]
    return "\n".join(section for section in sections if section is not None)


def render_cv_user_prompt(task: JsonRecord) -> str:
    """Render task-specific data for one defendant/court_view grounding task."""

    court_unit = task.get("court_unit", {})
    court_unit_id = str(court_unit.get("id", "")).strip()
    shared_candidate_sections = [
        "【事实候选】",
        "",
        _format_units(task.get("fact_candidates", [])),
        "",
        "【法条候选】",
        "",
        _format_law_candidates(task.get("law_candidates", [])),
    ]
    task_sections = [
        "【当前任务数据】",
        "",
        f"court_unit_id 必须等于当前法院观点编号：{court_unit_id}",
        "",
        "【目标被告人】",
        "",
        f"姓名：{task['defendant_name']}",
        "",
        _format_outcome(task.get("target_outcome", {})),
        "",
        "【当前法院观点】",
        "",
        _format_units([court_unit]),
    ]
    if task.get("fact_candidates_mode") == "all":
        sections = [*shared_candidate_sections, "", *task_sections]
    else:
        sections = [*task_sections, "", *shared_candidate_sections]
    return "\n".join(section for section in sections if section is not None)


def render_cv_messages(task: JsonRecord) -> list[JsonRecord]:
    """Render DeepSeek chat messages with a cache-friendly static prefix."""

    return [
        {"role": "system", "content": render_cv_system_prompt()},
        {"role": "user", "content": render_cv_user_prompt(task)},
    ]


def render_cv_prompt(task: JsonRecord) -> str:
    """Render a readable combined prompt for compatibility with legacy callers."""

    return "\n\n".join(message["content"] for message in render_cv_messages(task))


def process_file(
    input_path: str | Path,
    law_map_path: str | Path,
    output_path: str | Path,
    *,
    prompts_output_path: str | Path | None = None,
    case_prefix: str = "train",
    limit_cases: int | None = None,
    max_fact_candidates: int = 50,
    context_window: int = 1,
    fact_candidates_mode: str = "filtered",
) -> JsonRecord:
    """Write CV-level grounding tasks, optionally with readable prompts."""

    law_map = load_law_map(law_map_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_target = None
    if prompts_output_path is not None:
        prompts_output_path = Path(prompts_output_path)
        prompts_output_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_target = prompts_output_path.open("w", encoding="utf-8")

    cases_read = 0
    tasks_written = 0
    prompts_written = 0
    try:
        with Path(input_path).open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
            for line_index, line in enumerate(source):
                if limit_cases is not None and cases_read >= limit_cases:
                    break
                if not line.strip():
                    continue
                case = json.loads(line)
                if not isinstance(case, dict):
                    raise ValueError(f"line {line_index + 1}: expected JSON object")
                case_id = str(case.get("case_id") or f"{case_prefix}_{line_index:06d}")
                for task in build_cv_grounding_tasks(
                    case,
                    law_map,
                    case_id=case_id,
                    max_fact_candidates=max_fact_candidates,
                    context_window=context_window,
                    fact_candidates_mode=fact_candidates_mode,
                ):
                    target.write(json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n")
                    tasks_written += 1
                    if prompt_target is not None:
                        messages = render_cv_messages(task)
                        prompt = render_cv_prompt(task)
                        prompt_target.write(
                            json.dumps(
                                {
                                    "task_id": task["task_id"],
                                    "case_id": task["case_id"],
                                    "defendant_index": task["defendant_index"],
                                    "defendant_name": task["defendant_name"],
                                    "court_unit_id": task["court_unit"]["id"],
                                    "messages": messages,
                                    "messages_char_count": sum(len(message["content"]) for message in messages),
                                    "prompt": prompt,
                                    "prompt_char_count": len(prompt),
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        prompts_written += 1
                cases_read += 1
    finally:
        if prompt_target is not None:
            prompt_target.close()

    summary = {
        "cases_read": cases_read,
        "tasks_written": tasks_written,
        "tasks_output": str(output_path),
    }
    if prompts_output_path is not None:
        summary["prompts_written"] = prompts_written
        summary["prompts_output"] = str(prompts_output_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build court-view-level grounding tasks.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--law-map", type=Path, default=DEFAULT_LAW_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompts-output", type=Path)
    parser.add_argument("--case-prefix", default="train")
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--max-fact-candidates", type=int, default=50)
    parser.add_argument("--context-window", type=int, default=1)
    parser.add_argument("--fact-candidates-mode", choices=["filtered", "all"], default="filtered")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = process_file(
        args.input,
        args.law_map,
        args.output,
        prompts_output_path=args.prompts_output,
        case_prefix=args.case_prefix,
        limit_cases=args.limit_cases,
        max_fact_candidates=args.max_fact_candidates,
        context_window=args.context_window,
        fact_candidates_mode=args.fact_candidates_mode,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _parse_chinese_number(text: str | None) -> int | None:
    if not text:
        return None
    total = 0
    section = 0
    number = 0
    units = {"十": 10, "百": 100, "千": 1000}
    for char in text:
        if char in CHINESE_DIGITS:
            number = CHINESE_DIGITS[char]
        elif char in units:
            unit = units[char]
            section += (number or 1) * unit
            number = 0
        else:
            return None
    total += section + number
    return total if total > 0 else None


def _extract_law_articles_from_units(units: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for unit in _list_units(units):
        for article in extract_law_articles_from_text(str(unit.get("text", ""))):
            if article not in seen:
                result.append(article)
                seen.add(article)
    return result


def _list_units(units: Any) -> list[JsonRecord]:
    if not isinstance(units, list):
        return []
    return [dict(unit) for unit in units if isinstance(unit, dict) and unit.get("id")]


def _is_incomplete_court_view(text: str) -> bool:
    return text.endswith(("，", "、"))


def _is_summary_only_court_view(text: str) -> bool:
    if _has_substantive_court_content(text):
        return False
    return any(pattern in text for pattern in SUMMARY_ONLY_PATTERNS)


def _has_substantive_court_content(text: str) -> bool:
    return any(keyword in text for keyword in SUBSTANTIVE_COURT_KEYWORDS)


def _mentioned_defendants(text: str, defendants: Iterable[str]) -> set[str]:
    return {name for name in defendants if name and _contains_defendant_name(text, name)}


def _is_group_court_view(text: str) -> bool:
    return any(pattern in text for pattern in GROUP_COURT_PATTERNS)


def _allows_group_fact(court_text: str, fact_text: str) -> bool:
    if _is_group_court_view(court_text) and any(pattern in fact_text for pattern in GROUP_COURT_PATTERNS):
        return True
    if "共同" in court_text and "共同" in fact_text:
        return True
    if "上述事实" in fact_text or "证实" in fact_text or "证明" in fact_text:
        return True
    return False


def _matched_keywords(text: str, keywords: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        if keyword in text and keyword not in seen:
            result.append(keyword)
            seen.add(keyword)
    return result


def _is_negative_other_fact(text: str, defendant_name: str) -> bool:
    if not _contains_defendant_name(text, defendant_name):
        return False
    return any(pattern in text for pattern in NEGATIVE_FACT_PATTERNS)


def _contains_defendant_name(text: str, defendant_name: str) -> bool:
    if not defendant_name:
        return False
    suffix_guard = r"(?![某甲乙丙丁戊己庚辛壬癸A-Za-z0-9０-９])"
    return re.search(re.escape(defendant_name) + suffix_guard, text) is not None


if __name__ == "__main__":
    main()
