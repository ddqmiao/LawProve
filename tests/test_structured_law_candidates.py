from law_grounding.structured_law import build_structured_law_group, group_candidate_ids_by_article


LAW_MAP = {
    "67": {
        "type": "article",
        "article_id": "67",
        "article_label": "第六十七条",
        "text": "第六十七条完整内容。",
    },
    "67-1": {
        "type": "paragraph",
        "article_id": "67",
        "article_label": "第六十七条",
        "paragraph_no": 1,
        "text": "自首。",
    },
    "67-3": {
        "type": "paragraph",
        "article_id": "67",
        "article_label": "第六十七条",
        "paragraph_no": 3,
        "text": "如实供述。",
    },
    "25-1": {
        "type": "paragraph",
        "article_id": "25",
        "article_label": "第二十五条",
        "paragraph_no": 1,
        "text": "共同故意犯罪。",
    },
}


def test_build_structured_law_group_uses_article_context_but_only_given_selectable_ids():
    group = build_structured_law_group("67", ["67-3", "67"], LAW_MAP)

    assert group["article_id"] == "67"
    assert group["article_label"] == "第六十七条"
    assert group["article_text"] == "第六十七条完整内容。"
    assert [item["id"] for item in group["selectable_items"]] == ["67", "67-3"]
    assert group["selectable_items"][0]["type"] == "article"
    assert group["selectable_items"][1]["type"] == "paragraph"
    assert group["selectable_items"][1]["paragraph_no"] == 3


def test_build_structured_law_group_does_not_make_article_selectable_when_absent():
    group = build_structured_law_group("67", ["67-3"], LAW_MAP)

    assert [item["id"] for item in group["selectable_items"]] == ["67-3"]
    assert group["article_text"] == "第六十七条完整内容。"


def test_group_candidate_ids_by_article_preserves_first_article_order_and_sorts_within_article():
    groups = group_candidate_ids_by_article(["67-3", "25-1", "67-1"], LAW_MAP)

    assert [(group["article_id"], [item["id"] for item in group["selectable_items"]]) for group in groups] == [
        ("67", ["67-1", "67-3"]),
        ("25", ["25-1"]),
    ]
