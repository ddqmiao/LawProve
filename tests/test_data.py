import json

from law_grounding.data import (
    apply_retrieval_candidates_to_samples,
    build_inference_messages,
    build_sample,
    build_sft_record,
    expand_case,
    prepare_splits_from_prepared_samples,
    prepare_splits,
    sample_has_candidate_coverage,
)


LAW_MAP = {
    "25": {"type": "article", "article_id": "25", "article_label": "第二十五条", "text": "共同犯罪是指二人以上共同故意犯罪。"},
    "67": {"type": "article", "article_id": "67", "article_label": "第六十七条", "text": "自首和坦白整条。"},
    "67-1": {"type": "paragraph", "article_id": "67", "article_label": "第六十七条", "paragraph_no": 1, "text": "自首。"},
    "67-2": {"type": "paragraph", "article_id": "67", "article_label": "第六十七条", "paragraph_no": 2, "text": "以自首论。"},
    "67-3": {"type": "paragraph", "article_id": "67", "article_label": "第六十七条", "paragraph_no": 3, "text": "如实供述可以从轻。"},
    "72": {"type": "article", "article_id": "72", "article_label": "第七十二条", "text": "缓刑整条。"},
    "72-1": {"type": "paragraph", "article_id": "72", "article_label": "第七十二条", "paragraph_no": 1, "text": "可以宣告缓刑。"},
    "125": {"type": "article", "article_id": "125", "article_label": "第一百二十五条", "text": "危险物质整条。"},
    "125-1": {"type": "paragraph", "article_id": "125", "article_label": "第一百二十五条", "paragraph_no": 1, "text": "枪支弹药。"},
    "125-2": {"type": "paragraph", "article_id": "125", "article_label": "第一百二十五条", "paragraph_no": 2, "text": "毒害性物质。"},
    "224.1": {"type": "article", "article_id": "224.1", "article_label": "第二百二十四条之一", "text": "组织、领导传销活动。"},
}


def sample_case():
    return {
        "case_id": "valid_000001",
        "fact": [
            {"id": "F0", "text": "被告人汪某某购买毒狗丸。"},
            {"id": "F1", "text": "汪某某将部分毒狗丸卖给他人。"},
            {"id": "F2", "text": "汪某某到案后如实供述。"},
        ],
        "court_view": [
            {"id": "CV0", "text": "汪某某非法买卖毒害性物质，构成非法买卖危险物质罪。"},
            {"id": "CV1", "text": "汪某某到案后如实供述，可以从轻处罚。"},
        ],
        "defendants": [
            {
                "name": "汪某某",
                "defendant_index": 0,
                "grounding_links": [
                    {"court_unit_id": "CV0", "fact_ids": ["F0", "F1"], "law_ids": ["125-2"]},
                    {"court_unit_id": "CV1", "fact_ids": ["F2"], "law_ids": ["67-3"]},
                ],
                "warnings": [],
            }
        ],
        "outcomes": [{"name": "汪某某", "judgment": [{"article": ["125-2"]}]}],
        "relevant_articles": ["125-2", "67-3", "72-1"],
    }


def test_expand_case_creates_one_sample_per_grounding_link():
    samples, warnings = expand_case(sample_case(), LAW_MAP)

    assert warnings == []
    assert len(samples) == 2
    assert samples[0]["case_id"] == "valid_000001"
    assert samples[0]["defendant"] == "汪某某"
    assert samples[0]["court_unit_id"] == "CV0"
    assert samples[0]["gold_facts"] == ["F0", "F1"]
    assert samples[0]["gold_laws"] == ["125-2"]


def test_law_candidates_come_only_from_relevant_articles():
    sample = build_sample(sample_case(), sample_case()["defendants"][0], sample_case()["defendants"][0]["grounding_links"][1], LAW_MAP)
    candidate_ids = [item["id"] for item in sample["law_candidates"]]

    assert candidate_ids == ["125-2", "67-3", "72-1"]
    assert "67-3" in candidate_ids
    assert "67" not in candidate_ids


def test_apply_retrieval_candidates_replaces_law_candidates_by_sample_key_and_top_k():
    case = sample_case()
    samples, _ = expand_case(case, LAW_MAP)
    rows = [
        {
            "case_id": "valid_000001",
            "defendant": "汪某某",
            "defendant_index": 0,
            "court_unit_id": "CV0",
            "retrieved_laws": [
                {"id": "72-1", "rank": 1, "score": 0.9, "text": "72-1: cached"},
                {"id": "125-2", "rank": 2, "score": 0.8, "text": ""},
                {"id": "67-3", "rank": 3, "score": 0.7, "text": "67-3: ignored by top_k"},
            ],
        },
        {
            "case_id": "valid_000001",
            "defendant": "汪某某",
            "defendant_index": 0,
            "court_unit_id": "CV1",
            "retrieved_law_ids": ["67-3", "72-1"],
        },
    ]

    updated = apply_retrieval_candidates_to_samples(samples, rows, LAW_MAP, top_k=2)

    assert [item["id"] for item in updated[0]["law_candidates"]] == ["72-1", "125-2"]
    assert updated[0]["law_candidates"][0]["text"] == "72-1: cached"
    assert updated[0]["law_candidates"][1]["text"] == "125-2: 第一百二十五条第2款：毒害性物质。"
    assert updated[0]["gold_laws"] == ["125-2"]
    assert updated[0]["law_candidate_source"] == "retrieval"
    assert updated[0]["retrieval_candidate_top_k"] == 2
    assert [item["id"] for item in updated[1]["law_candidates"]] == ["67-3", "72-1"]


def test_missing_gold_law_is_warned_but_not_added_to_candidates():
    case = sample_case()
    case["relevant_articles"] = ["72-1"]
    sample = build_sample(case, case["defendants"][0], case["defendants"][0]["grounding_links"][0], LAW_MAP)

    assert [item["id"] for item in sample["law_candidates"]] == ["72-1"]
    assert sample["gold_laws"] == ["125-2"]
    assert not sample_has_candidate_coverage(sample)
    assert {"type": "missing_gold_law_candidate", "value": "125-2"} in sample["warnings"]


def test_gold_law_ids_are_mapped_to_matching_relevant_article_ids():
    case = sample_case()
    case["relevant_articles"] = ["224.1", "72-1"]
    link = {"court_unit_id": "CV0", "fact_ids": ["F0"], "law_ids": ["224之一"]}
    sample = build_sample(case, case["defendants"][0], link, LAW_MAP)

    assert [item["id"] for item in sample["law_candidates"]] == ["224.1", "72-1"]
    assert sample["gold_laws"] == ["224.1"]
    assert sample["original_gold_laws"] == ["224之一"]
    assert sample_has_candidate_coverage(sample)


def test_build_sft_record_uses_json_answer_and_candidate_text():
    sample = build_sample(sample_case(), sample_case()["defendants"][0], sample_case()["defendants"][0]["grounding_links"][0], LAW_MAP)
    record = build_sft_record(sample)

    assert [message["role"] for message in record["messages"]] == ["system", "user", "assistant"]
    assert "被告人：汪某某" in record["messages"][1]["content"]
    assert "F0: 被告人汪某某购买毒狗丸。" in record["messages"][1]["content"]
    assert "125-2: 第一百二十五条第2款：毒害性物质。" in record["messages"][1]["content"]
    assert json.loads(record["messages"][2]["content"]) == {
        "support_facts": ["F0", "F1"],
        "support_laws": ["125-2"],
    }


def test_build_sft_record_accepts_strict_fact_prompt_variant():
    sample = build_sample(sample_case(), sample_case()["defendants"][0], sample_case()["defendants"][0]["grounding_links"][0], LAW_MAP)

    record = build_sft_record(sample, prompt_variant="strict_fact")

    assert "最小必要事实集合" in record["messages"][0]["content"]
    assert "宁可少选，不要多选" in record["messages"][1]["content"]
    assert json.loads(record["messages"][2]["content"]) == {
        "support_facts": ["F0", "F1"],
        "support_laws": ["125-2"],
    }


def test_strict_fact_prompt_variant_adds_minimal_fact_selection_rules():
    sample = build_sample(sample_case(), sample_case()["defendants"][0], sample_case()["defendants"][0]["grounding_links"][0], LAW_MAP)

    messages = build_inference_messages(sample, prompt_variant="strict_fact")

    assert "最小必要事实集合" in messages[0]["content"]
    assert "不要选择背景事实、量刑情节、其他被告人的事实或重复表达的事实" in messages[0]["content"]
    assert "宁可少选，不要多选" in messages[1]["content"]


def test_prepare_splits_filters_samples_with_missing_law_candidates(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    law_map_path = tmp_path / "law_map.json"
    output_dir = tmp_path / "out"
    case = sample_case()
    case["relevant_articles"] = ["67-3"]
    (data_dir / "train.jsonl").write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
    law_map_path.write_text(json.dumps(LAW_MAP, ensure_ascii=False), encoding="utf-8")

    stats = prepare_splits(data_dir, law_map_path, output_dir)
    samples = [
        json.loads(line)
        for line in (output_dir / "samples" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert stats["splits"]["train"]["raw_samples"] == 2
    assert stats["splits"]["train"]["samples"] == 1
    assert stats["splits"]["train"]["filtered_missing_law_candidates"] == 1
    assert samples[0]["gold_laws"] == ["67-3"]


def test_prepare_splits_writes_strict_fact_sft_records(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    law_map_path = tmp_path / "law_map.json"
    output_dir = tmp_path / "out"
    (data_dir / "train.jsonl").write_text(json.dumps(sample_case(), ensure_ascii=False) + "\n", encoding="utf-8")
    law_map_path.write_text(json.dumps(LAW_MAP, ensure_ascii=False), encoding="utf-8")

    prepare_splits(data_dir, law_map_path, output_dir, prompt_variant="strict_fact")
    records = [
        json.loads(line)
        for line in (output_dir / "sft" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert "最小必要事实集合" in records[0]["messages"][0]["content"]
    assert "宁可少选，不要多选" in records[0]["messages"][1]["content"]


def test_prepare_splits_can_use_retrieval_candidates_without_filtering_missing_gold(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    retrieval_dir = tmp_path / "retrieval"
    retrieval_dir.mkdir()
    law_map_path = tmp_path / "law_map.json"
    output_dir = tmp_path / "out"
    (data_dir / "train.jsonl").write_text(json.dumps(sample_case(), ensure_ascii=False) + "\n", encoding="utf-8")
    law_map_path.write_text(json.dumps(LAW_MAP, ensure_ascii=False), encoding="utf-8")
    retrieval_rows = [
        {
            "case_id": "valid_000001",
            "defendant": "汪某某",
            "defendant_index": 0,
            "court_unit_id": "CV0",
            "retrieved_law_ids": ["72-1"],
        },
        {
            "case_id": "valid_000001",
            "defendant": "汪某某",
            "defendant_index": 0,
            "court_unit_id": "CV1",
            "retrieved_law_ids": ["67-3"],
        },
    ]
    (retrieval_dir / "train.top100.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in retrieval_rows) + "\n",
        encoding="utf-8",
    )

    stats = prepare_splits(
        data_dir,
        law_map_path,
        output_dir,
        retrieval_candidates_dir=retrieval_dir,
        retrieval_top_k=1,
        filter_missing_law_candidates=False,
    )
    samples = [
        json.loads(line)
        for line in (output_dir / "samples" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert stats["splits"]["train"]["samples"] == 2
    assert stats["splits"]["train"]["filtered_missing_law_candidates"] == 0
    assert stats["splits"]["train"]["samples_missing_law_candidates"] == 1
    assert [item["id"] for item in samples[0]["law_candidates"]] == ["72-1"]
    assert {"type": "missing_gold_law_candidate", "value": "125-2"} in samples[0]["warnings"]


def test_prepare_splits_from_prepared_samples_rebuilds_sft_with_retrieval_candidates(tmp_path):
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    retrieval_dir = tmp_path / "retrieval"
    retrieval_dir.mkdir()
    law_map_path = tmp_path / "law_map.json"
    output_dir = tmp_path / "out"
    source_samples, _ = expand_case(sample_case(), LAW_MAP)
    (samples_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(sample, ensure_ascii=False) for sample in source_samples) + "\n",
        encoding="utf-8",
    )
    retrieval_rows = [
        {
            "case_id": "valid_000001",
            "defendant": "汪某某",
            "defendant_index": 0,
            "court_unit_id": "CV0",
            "retrieved_law_ids": ["125-2", "72-1"],
        },
        {
            "case_id": "valid_000001",
            "defendant": "汪某某",
            "defendant_index": 0,
            "court_unit_id": "CV1",
            "retrieved_law_ids": ["67-3", "72-1"],
        },
    ]
    (retrieval_dir / "train.top100.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in retrieval_rows) + "\n",
        encoding="utf-8",
    )
    law_map_path.write_text(json.dumps(LAW_MAP, ensure_ascii=False), encoding="utf-8")

    stats = prepare_splits_from_prepared_samples(
        samples_dir,
        law_map_path,
        output_dir,
        retrieval_candidates_dir=retrieval_dir,
        retrieval_top_k=1,
        prompt_variant="strict_fact",
    )
    samples = [
        json.loads(line)
        for line in (output_dir / "samples" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [
        json.loads(line)
        for line in (output_dir / "sft" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert stats["splits"]["train"]["raw_samples"] == 2
    assert stats["splits"]["train"]["samples"] == 2
    assert [item["id"] for item in samples[0]["law_candidates"]] == ["125-2"]
    assert "最小必要事实集合" in records[0]["messages"][0]["content"]


def test_prepare_splits_from_prepared_samples_can_filter_missing_only_for_train(tmp_path):
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    retrieval_dir = tmp_path / "retrieval"
    retrieval_dir.mkdir()
    law_map_path = tmp_path / "law_map.json"
    output_dir = tmp_path / "out"
    source_samples, _ = expand_case(sample_case(), LAW_MAP)
    for split in ["train", "valid"]:
        (samples_dir / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(sample, ensure_ascii=False) for sample in source_samples) + "\n",
            encoding="utf-8",
        )
        retrieval_rows = [
            {
                "case_id": "valid_000001",
                "defendant": "汪某某",
                "defendant_index": 0,
                "court_unit_id": "CV0",
                "retrieved_law_ids": ["72-1"],
            },
            {
                "case_id": "valid_000001",
                "defendant": "汪某某",
                "defendant_index": 0,
                "court_unit_id": "CV1",
                "retrieved_law_ids": ["67-3"],
            },
        ]
        (retrieval_dir / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in retrieval_rows) + "\n",
            encoding="utf-8",
        )
    law_map_path.write_text(json.dumps(LAW_MAP, ensure_ascii=False), encoding="utf-8")

    stats = prepare_splits_from_prepared_samples(
        samples_dir,
        law_map_path,
        output_dir,
        retrieval_candidates_dir=retrieval_dir,
        retrieval_candidates_suffix=".jsonl",
        retrieval_top_k=1,
        filter_missing_law_candidates=False,
        filter_missing_law_candidate_splits={"train"},
    )
    train_samples = [
        json.loads(line)
        for line in (output_dir / "samples" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    valid_samples = [
        json.loads(line)
        for line in (output_dir / "samples" / "valid.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert stats["splits"]["train"]["samples"] == 1
    assert stats["splits"]["train"]["filtered_missing_law_candidates"] == 1
    assert stats["splits"]["valid"]["samples"] == 2
    assert stats["splits"]["valid"]["filtered_missing_law_candidates"] == 0
    assert train_samples[0]["gold_laws"] == ["67-3"]
    assert valid_samples[0]["gold_laws"] == ["125-2"]
