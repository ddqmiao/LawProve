from law_grounding.law import (
    article_key,
    format_law_text,
    is_paragraph_law_id,
    normalize_law_id,
    sibling_paragraph_ids,
)


def test_normalize_chinese_amendment_ids():
    assert normalize_law_id("224之一") == "224.1"
    assert normalize_law_id("253之一-1") == "253.1-1"
    assert normalize_law_id("287之二-1") == "287.2-1"
    assert normalize_law_id("125-2") == "125-2"


def test_article_key_uses_law_map_article_id_after_normalization():
    law_map = {
        "253.1-1": {
            "type": "paragraph",
            "article_id": "253.1",
            "article_label": "第二百五十三条之一",
            "paragraph_no": 1,
            "text": "违反国家有关规定，向他人出售或者提供公民个人信息。",
        }
    }

    assert article_key("253之一-1", law_map) == "253.1"
    assert is_paragraph_law_id("253之一-1", law_map)


def test_sibling_paragraph_ids_for_gold_law():
    law_map = {
        "125": {"type": "article", "article_id": "125", "article_label": "第一百二十五条", "text": "整条"},
        "125-1": {"type": "paragraph", "article_id": "125", "article_label": "第一百二十五条", "paragraph_no": 1, "text": "第一款"},
        "125-2": {"type": "paragraph", "article_id": "125", "article_label": "第一百二十五条", "paragraph_no": 2, "text": "第二款"},
        "126-1": {"type": "paragraph", "article_id": "126", "article_label": "第一百二十六条", "paragraph_no": 1, "text": "别的条"},
    }

    assert sibling_paragraph_ids("125-2", law_map) == ["125-1", "125-2"]


def test_format_law_text_preserves_original_id_but_uses_normalized_lookup():
    law_map = {
        "224.1": {
            "type": "article",
            "article_id": "224.1",
            "article_label": "第二百二十四条之一",
            "text": "组织、领导传销活动的，依法处罚。",
        },
        "67-3": {
            "type": "paragraph",
            "article_id": "67",
            "article_label": "第六十七条",
            "paragraph_no": 3,
            "text": "如实供述自己的罪行的，可以从轻处罚。",
        },
    }

    assert format_law_text("224之一", law_map).startswith("224之一: 第二百二十四条之一：组织、领导传销活动")
    assert format_law_text("67-3", law_map) == "67-3: 第六十七条第3款：如实供述自己的罪行的，可以从轻处罚。"
    assert format_law_text("999", law_map) == "999: 法条ID：999"
