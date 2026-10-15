import json
from pathlib import Path

from build_cv_grounding_tasks import (
    build_cv_grounding_tasks,
    extract_law_articles_from_text,
    process_file,
    render_cv_messages,
    render_cv_prompt,
    select_defendant_court_units,
    select_fact_candidates_for_court_unit,
)


def tiny_law_map():
    return {
        "25-1": {"id": "L25_1", "article": "25-1", "text": "共同犯罪是指二人以上共同故意犯罪。"},
        "26-1": {"id": "L26_1", "article": "26-1", "text": "组织、领导犯罪集团进行犯罪活动的或者在共同犯罪中起主要作用的，是主犯。"},
        "27": {"id": "L27", "article": "27", "text": "在共同犯罪中起次要或者辅助作用的，是从犯。"},
        "52": {"id": "L52", "article": "52", "text": "判处罚金，应当根据犯罪情节决定罚金数额。"},
        "67-3": {"id": "L67_3", "article": "67-3", "text": "如实供述自己罪行的，可以从轻处罚。"},
        "264": {"id": "L264", "article": "264", "text": "盗窃公私财物，数额较大的，处三年以下有期徒刑。"},
    }


def tiny_case():
    return {
        "fact": [
            {"id": "F0", "text": "经审理查明，张某、王某共同盗窃手机一部。"},
            {"id": "F1", "text": "张某负责进入店内实施盗窃，王某在门口望风。"},
            {"id": "F2", "text": "王某到案后如实供述自己的罪行。"},
            {"id": "F3", "text": "李某另案处理，与张某无关。"},
            {"id": "F4", "text": "上述事实，有被告人供述、证人证言等证据证实。"},
        ],
        "court_view": [
            {"id": "CV0", "text": "本院认为，被告人张某、王某以非法占有为目的，盗窃他人财物，其行为均已构成盗窃罪。"},
            {"id": "CV1", "text": "张某在共同犯罪中起主要作用，是主犯。"},
            {"id": "CV2", "text": "王某到案后如实供述，可以从轻处罚。"},
            {"id": "CV3", "text": "据此，公诉机关指控的罪名成立，本院予以支持。"},
            {"id": "CV4", "text": "三被告人均自愿认罪认罚，可以从宽处理。"},
            {"id": "CV5", "text": "对张某依照《中华人民共和国刑法》第二十五条第一款、第二十六条第一款、第五十二条"},
        ],
        "defendants": ["张某", "王某", "赵某"],
        "outcomes": [
            {
                "name": "张某",
                "judgment": [
                    {
                        "standard_accusation": "盗窃罪",
                        "article": ["264"],
                        "penalty": {"imprisonment": 10, "fine": 2000},
                    }
                ],
            },
            {
                "name": "王某",
                "judgment": [
                    {
                        "standard_accusation": "盗窃罪",
                        "article": ["264"],
                        "penalty": {"imprisonment": 8, "fine": 1000},
                    }
                ],
            },
        ],
        "relevant_articles": ["25-1", "27", "67-3"],
    }


def test_extract_law_articles_from_chinese_court_view_text():
    text = "依照《中华人民共和国刑法》第二十五条第一款、第二十六条第一款、第六十七条第三款、第五十二条之规定。"

    assert extract_law_articles_from_text(text) == ["25-1", "26-1", "67-3", "52"]


def test_select_defendant_court_units_keeps_target_and_group_views_but_drops_other_and_summary():
    selected = select_defendant_court_units(tiny_case(), "张某")

    assert [unit["id"] for unit in selected] == ["CV0", "CV1", "CV4", "CV5"]


def test_select_defendant_court_units_does_not_match_prefix_name_inside_longer_name():
    case = {
        "court_view": [
            {"id": "CV0", "text": "被告人陈某某起次要作用，是从犯。"},
            {"id": "CV1", "text": "被告人陈某起主要作用，是主犯。"},
        ],
        "defendants": ["陈某", "陈某某"],
    }

    selected = select_defendant_court_units(case, "陈某")

    assert [unit["id"] for unit in selected] == ["CV1"]


def test_select_fact_candidates_for_single_court_unit_filters_other_defendant_facts():
    case = tiny_case()
    court_unit = case["court_view"][1]

    facts = select_fact_candidates_for_court_unit(
        case["fact"],
        court_unit,
        defendant_name="张某",
        defendants=case["defendants"],
        max_fact_candidates=10,
        context_window=0,
    )

    assert [fact["id"] for fact in facts] == ["F0", "F1", "F4"]


def test_build_cv_grounding_tasks_creates_one_task_per_selected_court_unit_and_adds_court_laws():
    tasks = build_cv_grounding_tasks(tiny_case(), tiny_law_map(), case_id="train_000001")

    assert [task["task_id"] for task in tasks[:4]] == [
        "train_000001__D0__CV0",
        "train_000001__D0__CV1",
        "train_000001__D0__CV4",
        "train_000001__D0__CV5",
    ]
    first = tasks[0]
    assert first["defendant_name"] == "张某"
    assert first["court_unit"]["id"] == "CV0"
    assert "fact_candidates" in first
    assert "court_view_units" not in first
    assert "fact_units" not in first
    law_articles = [candidate["article"] for candidate in tasks[3]["law_candidates"]]
    assert law_articles == ["264", "25-1", "27", "67-3", "26-1", "52"]


def test_build_cv_grounding_tasks_can_use_all_case_facts_for_cache_experiment():
    tasks = build_cv_grounding_tasks(tiny_case(), tiny_law_map(), case_id="train_000001", fact_candidates_mode="all")

    assert [fact["id"] for fact in tasks[1]["fact_candidates"]] == ["F0", "F1", "F2", "F3", "F4"]
    assert tasks[1]["fact_candidates_mode"] == "all"


def test_render_cv_prompt_focuses_on_single_court_unit():
    task = build_cv_grounding_tasks(tiny_case(), tiny_law_map(), case_id="train_000001")[1]

    prompt = render_cv_prompt(task)

    assert prompt.startswith("你是法律裁判文书细粒度溯源标注助手。")
    assert "每次只处理一个法院观点" in prompt
    assert "【当前法院观点】" in prompt
    assert "CV1 张某在共同犯罪中起主要作用，是主犯。" in prompt
    assert "CV0 本院认为" not in prompt
    assert "【事实候选】" in prompt
    assert "F0 经审理查明" in prompt
    assert "不要求最少，也不要机械多选" in prompt
    assert "基础犯罪事实可以选择，但应与该观点存在直接关系" in prompt
    assert "【宽泛法院观点处理规则】" in prompt
    assert "指控事实成立" in prompt
    assert "优先选择与目标被告人直接相关的1-3条概括性事实或核心事实" in prompt
    assert "如果事实候选中没有这类直接相关事实，再让 fact_ids 为空" in prompt
    assert "可以让 fact_ids 为空，或只选择最直接的概括性认定事实" not in prompt
    assert "如果事实候选中已有直接概括目标被告人作案次数、总金额、总价值或参与情况的事实句" in prompt
    assert "多次同类犯罪不需要逐条选择全部细节" in prompt
    assert "【法条候选】" in prompt
    assert '"grounding_link"' in prompt
    assert '"grounding_links"' not in prompt
    assert "court_unit_id 必须等于当前法院观点编号：CV1" in prompt


def test_render_cv_messages_separates_static_rules_from_dynamic_task_data():
    task = build_cv_grounding_tasks(tiny_case(), tiny_law_map(), case_id="train_000001")[1]

    messages = render_cv_messages(task)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"].startswith("你是法律裁判文书细粒度溯源标注助手。")
    assert "court_unit_id 必须等于当前法院观点编号" in messages[0]["content"]
    assert "CV1" not in messages[0]["content"]
    assert "张某" not in messages[0]["content"]
    assert "【当前法院观点】" in messages[1]["content"]
    assert "CV1 张某在共同犯罪中起主要作用，是主犯。" in messages[1]["content"]


def test_render_cv_messages_puts_all_case_facts_before_dynamic_task_data_for_cache_prefix():
    task = build_cv_grounding_tasks(tiny_case(), tiny_law_map(), case_id="train_000001", fact_candidates_mode="all")[1]

    user_content = render_cv_messages(task)[1]["content"]

    assert user_content.index("【事实候选】") < user_content.index("court_unit_id 必须等于当前法院观点编号：CV1")
    assert "F3 李某另案处理，与张某无关。" in user_content


def test_process_file_writes_cv_level_tasks(tmp_path: Path):
    input_path = tmp_path / "split.jsonl"
    law_map_path = tmp_path / "law_map.json"
    output_path = tmp_path / "outputs" / "cv_tasks.jsonl"
    input_path.write_text(json.dumps(tiny_case(), ensure_ascii=False) + "\n", encoding="utf-8")
    law_map_path.write_text(json.dumps(tiny_law_map(), ensure_ascii=False), encoding="utf-8")

    summary = process_file(input_path, law_map_path, output_path, case_prefix="train")

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert summary == {
        "cases_read": 1,
        "tasks_written": len(rows),
        "tasks_output": str(output_path),
    }
    assert rows[0]["task_id"] == "train_000000__D0__CV0"


def test_process_file_can_also_write_cv_level_prompts(tmp_path: Path):
    input_path = tmp_path / "split.jsonl"
    law_map_path = tmp_path / "law_map.json"
    tasks_path = tmp_path / "outputs" / "cv_tasks.jsonl"
    prompts_path = tmp_path / "outputs" / "cv_prompts.jsonl"
    input_path.write_text(json.dumps(tiny_case(), ensure_ascii=False) + "\n", encoding="utf-8")
    law_map_path.write_text(json.dumps(tiny_law_map(), ensure_ascii=False), encoding="utf-8")

    summary = process_file(input_path, law_map_path, tasks_path, prompts_output_path=prompts_path, case_prefix="train")

    prompt_rows = [json.loads(line) for line in prompts_path.read_text(encoding="utf-8").splitlines()]
    assert summary["prompts_written"] == len(prompt_rows)
    assert prompt_rows[0]["task_id"] == "train_000000__D0__CV0"
    assert prompt_rows[0]["court_unit_id"] == "CV0"
    assert prompt_rows[0]["prompt"].startswith("你是法律裁判文书细粒度溯源标注助手。")
    assert prompt_rows[0]["prompt_char_count"] == len(prompt_rows[0]["prompt"])
    assert [message["role"] for message in prompt_rows[0]["messages"]] == ["system", "user"]
    assert prompt_rows[0]["messages_char_count"] == sum(len(message["content"]) for message in prompt_rows[0]["messages"])
