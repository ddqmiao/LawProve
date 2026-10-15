from build_criminal_law_mapping import (
    build_flat_map,
    build_lookup,
    cn_number_to_int,
    parse_law_paragraphs,
    resolve_reference,
)


def test_cn_number_to_int_handles_law_article_numbers():
    assert cn_number_to_int("六十七") == 67
    assert cn_number_to_int("一百三十三") == 133
    assert cn_number_to_int("二百零五") == 205
    assert cn_number_to_int("四百五十二") == 452


def test_parse_law_paragraphs_groups_enumerated_items_under_current_paragraph():
    articles = parse_law_paragraphs(
        [
            "目录",
            "第一编 总则",
            "第二编 分则",
            "第五章 侵犯财产罪",
            "第二百九十三条 有下列寻衅滋事行为之一，破坏社会秩序的，处五年以下有期徒刑：",
            "（一）随意殴打他人，情节恶劣的；",
            "（二）追逐、拦截、辱骂、恐吓他人，情节恶劣的；",
            "纠集他人多次实施前款行为，严重破坏社会秩序的，处五年以上十年以下有期徒刑。",
            "第二百九十三条之一 有下列情形之一，催收高利放贷等产生的非法债务，情节严重的，处三年以下有期徒刑：",
            "（一）使用暴力、胁迫方法的；",
        ],
        source_name="刑法.docx",
    )

    assert set(articles) == {"293", "293.1"}
    assert articles["293"]["chapter"] == "第五章 侵犯财产罪"
    assert len(articles["293"]["paragraphs"]) == 2
    assert articles["293"]["paragraphs"][0]["paragraph_id"] == "293-1"
    assert "（一）随意殴打他人" in articles["293"]["paragraphs"][0]["text"]
    assert "（二）追逐、拦截" in articles["293"]["paragraphs"][0]["text"]
    assert articles["293"]["paragraphs"][1]["paragraph_id"] == "293-2"
    assert articles["293.1"]["article_label"] == "第二百九十三条之一"
    assert articles["293.1"]["paragraphs"][0]["paragraph_id"] == "293.1-1"


def test_build_lookup_returns_article_and_paragraph_entries():
    articles = parse_law_paragraphs(
        [
            "第一编 总则",
            "第六十七条 犯罪以后自动投案，如实供述自己的罪行的，是自首。",
            "被采取强制措施的犯罪嫌疑人，如实供述司法机关还未掌握的本人其他罪行的，以自首论。",
            "犯罪嫌疑人虽不具有前两款规定的自首情节，但是如实供述自己罪行的，可以从轻处罚。",
            "第二百零五条之一 虚开本法第二百零五条规定以外的其他发票，情节严重的，处二年以下有期徒刑。",
            "单位犯前款罪的，对单位判处罚金。",
        ]
    )

    lookup = build_lookup(articles)

    assert lookup["67"]["scope"] == "article"
    assert lookup["67-3"]["scope"] == "paragraph"
    assert lookup["67-3"]["paragraph_no"] == 3
    assert "可以从轻处罚" in lookup["67-3"]["text"]
    assert lookup["205.1-2"]["article_id"] == "205.1"
    assert lookup["205.1-2"]["article_label"] == "第二百零五条之一"


def test_build_flat_map_keeps_simple_lookup_shape():
    articles = parse_law_paragraphs(
        [
            "第一编 总则",
            "第六十七条 犯罪以后自动投案，如实供述自己的罪行的，是自首。",
            "被采取强制措施的犯罪嫌疑人，如实供述司法机关还未掌握的本人其他罪行的，以自首论。",
        ],
        source_name="中华人民共和国刑法_20201226.docx",
    )

    flat_map = build_flat_map(articles)

    assert flat_map["67"] == {
        "type": "article",
        "article_id": "67",
        "article_label": "第六十七条",
        "text": (
            "第六十七条 犯罪以后自动投案，如实供述自己的罪行的，是自首。\n"
            "被采取强制措施的犯罪嫌疑人，如实供述司法机关还未掌握的本人其他罪行的，以自首论。"
        ),
        "source": "中华人民共和国刑法_20201226.docx",
    }
    assert flat_map["67-2"] == {
        "type": "paragraph",
        "article_id": "67",
        "article_label": "第六十七条",
        "paragraph_no": 2,
        "text": "被采取强制措施的犯罪嫌疑人，如实供述司法机关还未掌握的本人其他罪行的，以自首论。",
        "source": "中华人民共和国刑法_20201226.docx",
    }


def test_resolve_reference_falls_back_to_article_for_bad_paragraph_number():
    articles = parse_law_paragraphs(
        [
            "第一编 总则",
            "第二十五条 共同犯罪是指二人以上共同故意犯罪。",
            "二人以上共同过失犯罪，不以共同犯罪论处。",
        ]
    )
    lookup = build_lookup(articles)

    exact = resolve_reference("25-2", articles, lookup)
    fallback = resolve_reference("25-61", articles, lookup)
    missing = resolve_reference("0", articles, lookup)

    assert exact["matched"] is True
    assert exact["scope"] == "paragraph"
    assert fallback["matched"] is False
    assert fallback["reason"] == "paragraph_not_found"
    assert fallback["fallback_article_id"] == "25"
    assert "共同犯罪是指" in fallback["fallback_text"]
    assert missing["matched"] is False
    assert missing["reason"] == "article_not_found"
