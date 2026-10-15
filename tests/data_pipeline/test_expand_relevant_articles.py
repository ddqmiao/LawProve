import json

from expand_relevant_articles import (
    expand_record_relevant_articles,
    extract_criminal_law_refs_from_text,
    ordered_unique,
)


def test_ordered_unique_keeps_first_occurrence():
    assert ordered_unique(["67", "264", "67", "25-1"]) == ["67", "264", "25-1"]


def test_extract_criminal_law_refs_from_text_requires_criminal_law_context():
    law_map = {"25-1": {}, "67-3": {}, "225-1": {}, "328-1": {}, "4": {}}
    text = (
        "依照《中华人民共和国刑法》第二十五条第一款、第六十七条第三款之规定。"
        "根据《最高人民法院、最高人民检察院》第四条的规定，不应提取。"
        "根据我国刑法第二百二十五条第（一）项，应认定非法经营。"
        "符合刑法第三百二十八条第（四）项情形。"
    )

    assert extract_criminal_law_refs_from_text(text, law_map) == ["25-1", "67-3", "225-1", "328-1"]


def test_expand_record_relevant_articles_merges_relevant_judgment_and_text_refs():
    law_map = {"25-1": {}, "67-3": {}, "264": {}, "52": {}}
    record = {
        "case_id": "train_000001",
        "fact": "经审理查明，被告人共同盗窃。",
        "court_view": "依照《中华人民共和国刑法》第二十五条第一款、第六十七条第三款之规定。",
        "relevant_articles": ["67-3", "52"],
        "outcomes": [
            {"judgment": [{"article": ["264", "264"]}]},
            {"judgment": [{"article": ["264"]}]},
        ],
    }

    expanded, detail = expand_record_relevant_articles(record, law_map)

    assert expanded["relevant_articles"] == ["67-3", "52", "264", "25-1"]
    assert record["relevant_articles"] == ["67-3", "52"]
    assert detail["added_from_judgment"] == ["264"]
    assert detail["added_from_text"] == ["25-1"]
    assert json.dumps(expanded, ensure_ascii=False)
