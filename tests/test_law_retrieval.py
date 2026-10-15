import math

import numpy as np

from law_grounding.law_retrieval import (
    BM25LawRetriever,
    EmbeddingLawRetriever,
    LawDocument,
    SupervisedLawProfileRetriever,
    assign_group_folds,
    build_oof_fold_manifest,
    build_law_corpus,
    build_retrieval_query,
    compute_retrieval_metrics,
    fuse_retrieval_rows,
    retrieve_supervised_oof_for_samples,
)


LAW_MAP = {
    "25": {
        "type": "article",
        "article_id": "25",
        "article_label": "第二十五条",
        "text": "共同犯罪是指二人以上共同故意犯罪。",
    },
    "67-1": {
        "type": "paragraph",
        "article_id": "67",
        "article_label": "第六十七条",
        "paragraph_no": 1,
        "text": "犯罪以后自动投案，如实供述自己的罪行的，是自首。对于自首的犯罪分子，可以从轻或者减轻处罚。",
    },
    "72-1": {
        "type": "paragraph",
        "article_id": "72",
        "article_label": "第七十二条",
        "paragraph_no": 1,
        "text": "对于被判处拘役、三年以下有期徒刑的犯罪分子，可以宣告缓刑。",
    },
    "125-2": {
        "type": "paragraph",
        "article_id": "125",
        "article_label": "第一百二十五条",
        "paragraph_no": 2,
        "text": "非法制造、买卖、运输、储存毒害性、放射性、传染病病原体等物质，危害公共安全的。",
    },
}


def test_build_law_corpus_keeps_ids_and_formatted_text():
    corpus = build_law_corpus(LAW_MAP)

    assert [doc.law_id for doc in corpus] == ["25", "67-1", "72-1", "125-2"]
    assert corpus[1].article_id == "67"
    assert "第六十七条第1款" in corpus[1].text
    assert "自首" in corpus[1].text


def test_bm25_retriever_ranks_keyword_matching_law_first():
    retriever = BM25LawRetriever(build_law_corpus(LAW_MAP))

    results = retriever.search("被告人自动投案并如实供述，可以从轻处罚", top_k=3)

    assert results[0].law_id == "67-1"
    assert results[0].score > results[1].score


def test_embedding_retriever_ranks_by_cosine_similarity():
    documents = [
        LawDocument(law_id="67-1", article_id="67", text="自首", search_text="自首"),
        LawDocument(law_id="72-1", article_id="72", text="缓刑", search_text="缓刑"),
        LawDocument(law_id="125-2", article_id="125", text="危险物质", search_text="危险物质"),
    ]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    retriever = EmbeddingLawRetriever(documents, embeddings)

    results = retriever.search_by_vector(np.array([0.9, 0.1, 0.0], dtype=np.float32), top_k=2)

    assert [result.law_id for result in results] == ["67-1", "72-1"]
    assert results[0].score > results[1].score


def test_supervised_law_profile_retriever_learns_from_gold_training_queries():
    documents = build_law_corpus(LAW_MAP)
    train_samples = [
        {
            "defendant": "张某",
            "court_view": "张某到案后如实供述自己的罪行，可以从轻处罚。",
            "fact_candidates": [],
            "gold_laws": ["67-1"],
        },
        {
            "defendant": "李某",
            "court_view": "李某符合缓刑条件，可以宣告缓刑。",
            "fact_candidates": [],
            "gold_laws": ["72-1"],
        },
    ]
    retriever = SupervisedLawProfileRetriever.fit(documents, train_samples, fact_mode="none")

    results = retriever.search("被告人主动到案并如实供述，可以从轻处罚", top_k=2)

    assert results[0].law_id == "67-1"
    assert results[0].score > 0


def test_supervised_oof_retrieval_holds_out_each_eval_fold():
    documents = build_law_corpus(LAW_MAP)
    samples = [
        {
            "case_id": "c1",
            "defendant": "张某",
            "court_unit_id": "CV0",
            "court_view": "相同裁判观点唯一词",
            "fact_candidates": [],
            "gold_laws": ["67-1"],
        },
        {
            "case_id": "c2",
            "defendant": "李某",
            "court_unit_id": "CV0",
            "court_view": "相同裁判观点唯一词",
            "fact_candidates": [],
            "gold_laws": ["72-1"],
        },
    ]

    rows = retrieve_supervised_oof_for_samples(
        samples,
        documents,
        num_folds=2,
        top_k=2,
        fact_mode="none",
        law_text_weight=0.0,
    )

    assert rows[0]["retrieved_law_ids"][0] == "72-1"
    assert rows[1]["retrieved_law_ids"][0] == "67-1"
    assert {rows[0]["oof_fold"], rows[1]["oof_fold"]} == {0, 1}
    assert rows[0]["oof_group_field"] == "case_id"
    assert rows[0]["oof_seed"] == 42


def test_assign_group_folds_keeps_every_case_in_one_fold_and_is_deterministic():
    samples = [
        {"case_id": "large", "court_unit_id": f"CV{i}"} for i in range(5)
    ] + [
        {"case_id": f"case-{i}", "court_unit_id": "CV0"} for i in range(7)
    ]

    first = assign_group_folds(samples, num_folds=3, group_field="case_id", seed=17)
    second = assign_group_folds(samples, num_folds=3, group_field="case_id", seed=17)

    assert first == second
    case_folds = {}
    for sample, fold in zip(samples, first, strict=True):
        case_folds.setdefault(sample["case_id"], set()).add(fold)
    assert all(len(folds) == 1 for folds in case_folds.values())
    fold_sizes = [first.count(fold) for fold in range(3)]
    assert max(fold_sizes) - min(fold_sizes) <= 5


def test_assign_group_folds_rejects_missing_groups_and_too_many_folds():
    try:
        assign_group_folds([{"case_id": ""}, {"case_id": "c2"}], num_folds=2)
    except ValueError as exc:
        assert "case_id" in str(exc)
    else:
        raise AssertionError("expected missing case_id to fail")

    try:
        assign_group_folds([{"case_id": "c1"}, {"case_id": "c1"}], num_folds=2)
    except ValueError as exc:
        assert "unique groups" in str(exc)
    else:
        raise AssertionError("expected too many folds to fail")


def test_build_oof_fold_manifest_reports_balanced_non_crossing_groups():
    samples = [
        {"case_id": "c1"},
        {"case_id": "c1"},
        {"case_id": "c2"},
        {"case_id": "c3"},
    ]
    assignments = assign_group_folds(samples, num_folds=2, seed=9)

    manifest = build_oof_fold_manifest(samples, assignments, num_folds=2, seed=9)

    assert manifest["group_field"] == "case_id"
    assert manifest["seed"] == 9
    assert manifest["num_samples"] == 4
    assert manifest["num_groups"] == 3
    assert manifest["groups_crossing_folds"] == 0
    assert sum(fold["num_samples"] for fold in manifest["folds"]) == 4


def test_build_retrieval_query_uses_court_view_and_all_fact_candidates():
    sample = {
        "defendant": "汪某某",
        "court_view": "汪某某非法买卖毒害性物质，构成非法买卖危险物质罪。",
        "fact_candidates": [
            {"id": "F0", "text": "汪某某购买毒狗丸。"},
            {"id": "F1", "text": "汪某某将毒狗丸卖给他人。"},
        ],
    }

    query = build_retrieval_query(sample)

    assert "汪某某" in query
    assert "非法买卖毒害性物质" in query
    assert "购买毒狗丸" in query
    assert "卖给他人" in query


def test_build_retrieval_query_can_omit_fact_candidates():
    sample = {
        "defendant": "汪某某",
        "court_view": "汪某某非法买卖毒害性物质，构成非法买卖危险物质罪。",
        "fact_candidates": [{"id": "F0", "text": "背景事实噪声。"}],
    }

    query = build_retrieval_query(sample, fact_mode="none")

    assert "非法买卖毒害性物质" in query
    assert "背景事实噪声" not in query


def test_compute_retrieval_metrics_handles_multiple_gold_laws():
    rows = [
        {"gold_laws": ["25", "67-1"], "retrieved_law_ids": ["25", "72-1", "67-1"]},
        {"gold_laws": ["125-2"], "retrieved_law_ids": ["72-1", "67-1", "25"]},
    ]

    metrics = compute_retrieval_metrics(rows, cutoffs=[1, 3])

    assert metrics["num_samples"] == 2
    assert metrics["num_gold_laws"] == 3
    assert math.isclose(metrics["micro_recall@1"], 1 / 3)
    assert math.isclose(metrics["sample_recall@1"], 0.25)
    assert math.isclose(metrics["all_hit@1"], 0.0)
    assert math.isclose(metrics["micro_recall@3"], 2 / 3)
    assert math.isclose(metrics["sample_recall@3"], 0.5)
    assert math.isclose(metrics["all_hit@3"], 0.5)
    assert math.isclose(metrics["mrr"], 0.5)


def test_fuse_retrieval_rows_uses_rrf_and_preserves_sample_metadata():
    bm25_rows = [
        {
            "case_id": "c1",
            "defendant": "张某",
            "court_unit_id": "CV0",
            "gold_laws": ["67-1"],
            "retrieved_laws": [
                {"id": "25", "rank": 1, "score": 3.0, "text": "共同犯罪"},
                {"id": "67-1", "rank": 2, "score": 2.0, "text": "自首"},
            ],
        }
    ]
    embedding_rows = [
        {
            "case_id": "c1",
            "defendant": "张某",
            "court_unit_id": "CV0",
            "gold_laws": ["67-1"],
            "retrieved_laws": [
                {"id": "67-1", "rank": 1, "score": 0.9, "text": "自首"},
                {"id": "72-1", "rank": 2, "score": 0.8, "text": "缓刑"},
            ],
        }
    ]

    fused = fuse_retrieval_rows([bm25_rows, embedding_rows], top_k=3, rrf_k=60)

    assert fused[0]["case_id"] == "c1"
    assert fused[0]["gold_laws"] == ["67-1"]
    assert fused[0]["retrieved_law_ids"] == ["67-1", "25", "72-1"]
    assert [item["rank"] for item in fused[0]["retrieved_laws"]] == [1, 2, 3]
