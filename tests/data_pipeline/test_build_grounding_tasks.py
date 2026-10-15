import json
from pathlib import Path

from build_grounding_tasks import (
    build_grounding_tasks,
    build_law_candidates,
    process_file,
    render_prompt,
)


def tiny_law_map():
    return {
        "264": {
            "id": "L264",
            "article": "264",
            "text": "盗窃公私财物，数额较大的，处三年以下有期徒刑、拘役或者管制。",
        },
        "67-3": {
            "id": "L67_3",
            "article": "67-3",
            "text": "如实供述自己罪行的，可以从轻处罚。",
        },
        "25-1": {
            "id": "L25_1",
            "article": "25-1",
            "text": "共同犯罪是指二人以上共同故意犯罪。",
        },
        "287之二-1": {
            "id": "L287Z2_1",
            "article": "287之二-1",
            "text": "明知他人利用信息网络实施犯罪，为其犯罪提供帮助，情节严重的，处三年以下有期徒刑或者拘役。",
        },
    }


def tiny_case():
    return {
        "fact": [
            {"id": "F0", "text": "张某盗窃手机一部。"},
            {"id": "F1", "text": "李某帮助张某望风。"},
            {"id": "F2", "text": "张某到案后如实供述。"},
        ],
        "court_view": [
            {"id": "CV0", "text": "本院认为，张某的行为构成盗窃罪。"},
            {"id": "CV1", "text": "李某在共同犯罪中起帮助作用。"},
            {"id": "CV2", "text": "张某到案后如实供述，可以从轻处罚。"},
        ],
        "defendants": ["张某", "李某"],
        "outcomes": [
            {
                "name": "张某",
                "judgment": [
                    {
                        "accusation": "盗窃罪",
                        "standard_accusation": "盗窃罪",
                        "article": ["264"],
                        "penalty": {"imprisonment": 10, "fine": 2000},
                    }
                ],
            },
            {
                "name": "李某",
                "judgment": [
                    {
                        "accusation": "盗窃罪",
                        "standard_accusation": "盗窃罪",
                        "article": ["264", "25-1"],
                        "penalty": {"imprisonment": 8, "fine": 1000},
                    }
                ],
            },
        ],
        "relevant_articles": ["67-3", "25-1"],
    }


def test_build_law_candidates_puts_defendant_articles_first_and_deduplicates():
    candidates, missing = build_law_candidates(["264", "67-3"], ["67-3", "25-1"], tiny_law_map())

    assert [candidate["article"] for candidate in candidates] == ["264", "67-3", "25-1"]
    assert missing == []


def test_build_law_candidates_resolves_cmdl_dot_article_codes():
    candidates, missing = build_law_candidates(["287.2-1"], [], tiny_law_map())

    assert candidates == [
        {
            "article": "287之二-1",
            "text": "明知他人利用信息网络实施犯罪，为其犯罪提供帮助，情节严重的，处三年以下有期徒刑或者拘役。",
            "source_articles": ["287.2-1"],
        }
    ]
    assert missing == []


def test_build_grounding_tasks_creates_one_task_per_defendant():
    tasks = build_grounding_tasks(tiny_case(), tiny_law_map(), case_id="train_000001")

    assert [task["defendant_name"] for task in tasks] == ["张某", "李某"]
    assert tasks[0]["task_id"] == "train_000001__D0"
    assert tasks[0]["defendant_index"] == 0
    assert tasks[0]["fact_units"][0] == {"id": "F0", "text": "张某盗窃手机一部。"}
    assert tasks[0]["court_view_units"][0] == {"id": "CV0", "text": "本院认为，张某的行为构成盗窃罪。"}
    assert [candidate["article"] for candidate in tasks[0]["law_candidates"]] == ["264", "67-3", "25-1"]
    assert tasks[0]["missing_law_articles"] == []


def test_render_prompt_is_readable_text_not_raw_task_json():
    task = build_grounding_tasks(tiny_case(), tiny_law_map(), case_id="train_000001")[0]

    prompt = render_prompt(task)

    assert "【目标被告人】" in prompt
    assert "姓名：张某" in prompt
    assert "CV0 本院认为，张某的行为构成盗窃罪。" in prompt
    assert "F0 张某盗窃手机一部。" in prompt
    assert "264 盗窃公私财物" in prompt
    assert prompt.count("判决信息：") == 1
    assert '"grounding_links"' in prompt
    assert '"fact_units"' not in prompt
    assert '"court_view_units"' not in prompt


def test_render_prompt_uses_grounding_rules_without_encouraging_extra_facts():
    task = build_grounding_tasks(tiny_case(), tiny_law_map(), case_id="train_000001")[0]

    prompt = render_prompt(task)

    assert prompt.startswith("你是法律裁判文书细粒度溯源标注助手。")
    assert "【标注目标】" in prompt
    assert "【court_unit_id 选择规则】" in prompt
    assert "【fact_ids 选择规则】" in prompt
    assert "【law_ids 选择规则】" in prompt
    assert "fact_ids 不是机械追求最少，应选择直接、充分、不过度重复的事实；避免堆砌间接或重复证据。" in prompt
    assert "fact_ids 不是越少越好" not in prompt


def test_process_file_writes_tasks_and_prompt_files(tmp_path: Path):
    input_path = tmp_path / "split.jsonl"
    law_map_path = tmp_path / "law_map.json"
    tasks_path = tmp_path / "outputs" / "grounding_tasks.jsonl"
    prompts_path = tmp_path / "outputs" / "grounding_prompts.jsonl"
    input_path.write_text(json.dumps(tiny_case(), ensure_ascii=False) + "\n", encoding="utf-8")
    law_map_path.write_text(json.dumps(tiny_law_map(), ensure_ascii=False), encoding="utf-8")

    summary = process_file(input_path, law_map_path, tasks_path, prompts_path, case_prefix="train")

    assert summary == {
        "cases_read": 1,
        "tasks_written": 2,
        "prompts_written": 2,
        "tasks_output": str(tasks_path),
        "prompts_output": str(prompts_path),
    }
    task_rows = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines()]
    prompt_rows = [json.loads(line) for line in prompts_path.read_text(encoding="utf-8").splitlines()]
    assert task_rows[1]["task_id"] == "train_000000__D1"
    assert prompt_rows[0]["task_id"] == "train_000000__D0"
    assert prompt_rows[0]["prompt"].startswith("你是法律裁判文书细粒度溯源标注助手。")
