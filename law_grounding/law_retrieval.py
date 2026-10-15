"""BM25 law retrieval utilities for grounding candidate generation."""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .law import format_law_text, law_sort_key, normalize_law_id


DEFAULT_CUTOFFS = (5, 10, 20, 50)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class LawDocument:
    law_id: str
    article_id: str
    text: str
    search_text: str


@dataclass(frozen=True)
class SearchResult:
    law_id: str
    score: float
    rank: int
    text: str


def tokenize_text(text: str) -> list[str]:
    """Tokenize mixed Chinese/legal text with CJK chars, CJK bigrams, and ASCII words."""
    tokens: list[str] = []
    cjk_run: list[str] = []
    word_run: list[str] = []

    def flush_cjk() -> None:
        if not cjk_run:
            return
        tokens.extend(cjk_run)
        tokens.extend("".join(cjk_run[index : index + 2]) for index in range(len(cjk_run) - 1))
        cjk_run.clear()

    def flush_word() -> None:
        if word_run:
            tokens.append("".join(word_run))
            word_run.clear()

    for char in text.lower():
        if _CJK_RE.match(char):
            flush_word()
            cjk_run.append(char)
        elif _WORD_RE.match(char):
            flush_cjk()
            word_run.append(char)
        else:
            flush_cjk()
            flush_word()
    flush_cjk()
    flush_word()
    return tokens


def build_law_corpus(
    law_map: dict[str, dict[str, Any]],
    *,
    include_types: Iterable[str] = ("article", "paragraph"),
) -> list[LawDocument]:
    """Build a searchable corpus from criminal_law_map entries."""
    allowed_types = set(include_types)
    documents: list[LawDocument] = []
    for law_id in sorted(law_map, key=law_sort_key):
        info = law_map[law_id]
        if info.get("type") not in allowed_types:
            continue
        formatted = format_law_text(law_id, law_map)
        search_parts = [
            law_id,
            str(info.get("article_id") or ""),
            str(info.get("article_label") or ""),
            str(info.get("paragraph_no") or ""),
            str(info.get("text") or ""),
            formatted,
        ]
        documents.append(
            LawDocument(
                law_id=str(law_id),
                article_id=str(info.get("article_id") or law_id),
                text=formatted,
                search_text=" ".join(part for part in search_parts if part),
            )
        )
    return documents


class BM25LawRetriever:
    """Small dependency-free BM25 retriever for the criminal law corpus."""

    def __init__(self, documents: list[LawDocument], *, k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._doc_tokens = [tokenize_text(document.search_text) for document in documents]
        self._doc_term_counts = [Counter(tokens) for tokens in self._doc_tokens]
        self._doc_lengths = [len(tokens) for tokens in self._doc_tokens]
        self._avg_doc_len = sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        self._idf = self._compute_idf(self._doc_tokens)
        self._doc_norms = [
            self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_len)
            for doc_len in self._doc_lengths
        ]
        self._postings = self._build_postings()

    def _compute_idf(self, tokenized_documents: list[list[str]]) -> dict[str, float]:
        doc_count = len(tokenized_documents)
        doc_freq: Counter[str] = Counter()
        for tokens in tokenized_documents:
            doc_freq.update(set(tokens))
        return {
            token: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freq.items()
        }

    def _build_postings(self) -> dict[str, list[tuple[int, int]]]:
        postings: dict[str, list[tuple[int, int]]] = {}
        for index, term_counts in enumerate(self._doc_term_counts):
            for term, freq in term_counts.items():
                postings.setdefault(term, []).append((index, freq))
        return postings

    def _score_terms(self, query_terms: Counter[str], document_index: int) -> float:
        if not query_terms or not self.documents:
            return 0.0
        term_counts = self._doc_term_counts[document_index]
        score = 0.0
        for term, query_freq in query_terms.items():
            freq = term_counts.get(term, 0)
            if freq == 0:
                continue
            denominator = freq + self._doc_norms[document_index]
            query_weight = 1.0 + math.log(query_freq)
            score += query_weight * self._idf.get(term, 0.0) * freq * (self.k1 + 1) / denominator
        return score

    def score(self, query: str, document_index: int) -> float:
        return self._score_terms(Counter(tokenize_text(query)), document_index)

    def search(self, query: str, *, top_k: int = 20) -> list[SearchResult]:
        query_terms = Counter(tokenize_text(query))
        scores = [0.0] * len(self.documents)
        for term, query_freq in query_terms.items():
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            query_weight = 1.0 + math.log(query_freq)
            for index, freq in self._postings.get(term, []):
                denominator = freq + self._doc_norms[index]
                scores[index] += query_weight * idf * freq * (self.k1 + 1) / denominator
        scored = [(score, document) for score, document in zip(scores, self.documents, strict=True)]
        scored.sort(key=lambda item: (-item[0], law_sort_key(item[1].law_id)))
        return [
            SearchResult(law_id=document.law_id, score=score, rank=rank, text=document.text)
            for rank, (score, document) in enumerate(scored[:top_k], start=1)
        ]


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Return L2-normalized embeddings, preserving zero rows."""
    values = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


class EmbeddingLawRetriever:
    """Cosine-similarity retriever over precomputed law embeddings."""

    def __init__(self, documents: list[LawDocument], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length")
        self.documents = documents
        self.embeddings = normalize_embeddings(embeddings)

    def search_by_vector(self, query_embedding: np.ndarray, *, top_k: int = 20) -> list[SearchResult]:
        query = normalize_embeddings(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))[0]
        scores = self.embeddings @ query
        order = sorted(
            range(len(self.documents)),
            key=lambda index: (-float(scores[index]), law_sort_key(self.documents[index].law_id)),
        )[:top_k]
        return [
            SearchResult(
                law_id=self.documents[index].law_id,
                score=float(scores[index]),
                rank=rank,
                text=self.documents[index].text,
            )
            for rank, index in enumerate(order, start=1)
        ]

    def search_many(self, query_embeddings: np.ndarray, *, top_k: int = 20) -> list[list[SearchResult]]:
        queries = normalize_embeddings(query_embeddings)
        return [self.search_by_vector(query, top_k=top_k) for query in queries]


class SupervisedLawProfileRetriever:
    """Sparse trainable retriever that learns law-label query profiles from gold laws."""

    def __init__(self, documents: list[LawDocument], profiles: list[Counter[str]]):
        if len(documents) != len(profiles):
            raise ValueError("documents and profiles must have the same length")
        self.documents = documents
        self.profiles = profiles
        self._postings = self._build_postings()
        self._idf = self._compute_profile_idf()

    @classmethod
    def fit(
        cls,
        documents: list[LawDocument],
        samples: list[dict[str, Any]],
        *,
        fact_mode: str = "none",
        law_text_weight: float = 0.15,
    ) -> "SupervisedLawProfileRetriever":
        law_to_index = {document.law_id: index for index, document in enumerate(documents)}
        normalized_to_index = {normalize_law_id(document.law_id): index for index, document in enumerate(documents)}
        profiles = [Counter(tokenize_text(document.search_text)) for document in documents]
        if law_text_weight != 1.0:
            profiles = [
                Counter({term: count * law_text_weight for term, count in profile.items()})
                for profile in profiles
            ]
        for sample in samples:
            query_terms = Counter(tokenize_text(build_retrieval_query(sample, fact_mode=fact_mode)))
            if not query_terms:
                continue
            for law_id in sample.get("gold_laws") or []:
                index = law_to_index.get(str(law_id), normalized_to_index.get(normalize_law_id(str(law_id))))
                if index is not None:
                    profiles[index].update(query_terms)
        return cls(documents, profiles)

    def _build_postings(self) -> dict[str, list[tuple[int, float]]]:
        postings: dict[str, list[tuple[int, float]]] = {}
        for index, profile in enumerate(self.profiles):
            for term, freq in profile.items():
                postings.setdefault(term, []).append((index, float(freq)))
        return postings

    def _compute_profile_idf(self) -> dict[str, float]:
        label_count = len(self.profiles)
        return {
            term: math.log(1 + (label_count - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self._postings.items()
        }

    def search(self, query: str, *, top_k: int = 20) -> list[SearchResult]:
        query_terms = Counter(tokenize_text(query))
        scores = [0.0] * len(self.documents)
        for term, query_freq in query_terms.items():
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            query_weight = 1.0 + math.log(query_freq)
            for index, profile_freq in self._postings.get(term, []):
                scores[index] += query_weight * idf * math.log1p(profile_freq)
        order = sorted(
            range(len(self.documents)),
            key=lambda index: (-scores[index], law_sort_key(self.documents[index].law_id)),
        )[:top_k]
        return [
            SearchResult(
                law_id=self.documents[index].law_id,
                score=float(scores[index]),
                rank=rank,
                text=self.documents[index].text,
            )
            for rank, index in enumerate(order, start=1)
        ]


def retrieve_supervised_for_samples(
    samples: list[dict[str, Any]],
    retriever: SupervisedLawProfileRetriever,
    *,
    top_k: int,
    fact_mode: str = "none",
    max_fact_chars: int | None = 12000,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        query = build_retrieval_query(sample, fact_mode=fact_mode, max_fact_chars=max_fact_chars)
        results = retriever.search(query, top_k=top_k)
        rows.append(
            {
                "case_id": sample.get("case_id"),
                "defendant": sample.get("defendant"),
                "defendant_index": sample.get("defendant_index"),
                "court_unit_id": sample.get("court_unit_id"),
                "gold_laws": sample.get("gold_laws", []),
                "retrieved_law_ids": [result.law_id for result in results],
                "retrieved_laws": [
                    {"id": result.law_id, "rank": result.rank, "score": result.score, "text": result.text}
                    for result in results
                ],
            }
        )
    return rows


def assign_group_folds(
    samples: list[dict[str, Any]],
    *,
    num_folds: int,
    group_field: str = "case_id",
    seed: int = 42,
) -> list[int]:
    """Assign deterministic, size-balanced folds without splitting a group."""
    if num_folds < 2:
        raise ValueError("num_folds must be >= 2 for out-of-fold retrieval")
    if not group_field:
        raise ValueError("group_field cannot be empty")
    if not samples:
        return []

    indices_by_group: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        value = sample.get(group_field)
        if value is None or not str(value).strip():
            raise ValueError(f"sample at index {index} has no non-empty {group_field}")
        indices_by_group[str(value)].append(index)

    if len(indices_by_group) < num_folds:
        raise ValueError("num_folds cannot exceed number of unique groups")

    groups = list(indices_by_group.items())
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)

    fold_sizes = [0] * num_folds
    fold_group_counts = [0] * num_folds
    assignments = [-1] * len(samples)
    for _, indices in groups:
        fold = min(
            range(num_folds),
            key=lambda candidate: (fold_sizes[candidate], fold_group_counts[candidate], candidate),
        )
        for index in indices:
            assignments[index] = fold
        fold_sizes[fold] += len(indices)
        fold_group_counts[fold] += 1

    if any(fold < 0 for fold in assignments):
        raise RuntimeError("failed to assign every sample to an OOF fold")
    return assignments


def build_oof_fold_manifest(
    samples: list[dict[str, Any]],
    assignments: list[int],
    *,
    num_folds: int,
    group_field: str = "case_id",
    seed: int = 42,
) -> dict[str, Any]:
    """Summarize a grouped OOF assignment and prove groups do not cross folds."""
    if len(samples) != len(assignments):
        raise ValueError("samples and assignments must have the same length")

    fold_groups: list[set[str]] = [set() for _ in range(num_folds)]
    fold_sample_counts = [0] * num_folds
    group_to_fold: dict[str, int] = {}
    for index, (sample, fold) in enumerate(zip(samples, assignments, strict=True)):
        if fold < 0 or fold >= num_folds:
            raise ValueError(f"invalid fold {fold} at sample index {index}")
        value = sample.get(group_field)
        if value is None or not str(value).strip():
            raise ValueError(f"sample at index {index} has no non-empty {group_field}")
        group = str(value)
        previous = group_to_fold.setdefault(group, fold)
        if previous != fold:
            raise ValueError(f"group {group!r} crosses folds {previous} and {fold}")
        fold_groups[fold].add(group)
        fold_sample_counts[fold] += 1

    return {
        "assignment": "grouped_size_balanced",
        "group_field": group_field,
        "seed": seed,
        "num_folds": num_folds,
        "num_samples": len(samples),
        "num_groups": len(group_to_fold),
        "groups_crossing_folds": 0,
        "folds": [
            {
                "fold": fold,
                "num_samples": fold_sample_counts[fold],
                "num_groups": len(fold_groups[fold]),
                "groups": sorted(fold_groups[fold]),
            }
            for fold in range(num_folds)
        ],
    }


def retrieve_supervised_oof_for_samples(
    samples: list[dict[str, Any]],
    documents: list[LawDocument],
    *,
    num_folds: int = 5,
    top_k: int,
    fact_mode: str = "none",
    max_fact_chars: int | None = 12000,
    law_text_weight: float = 0.15,
    group_field: str = "case_id",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Retrieve train candidates with grouped out-of-fold supervised profiles."""
    if not samples:
        return []

    assignments = assign_group_folds(
        samples,
        num_folds=num_folds,
        group_field=group_field,
        seed=seed,
    )
    rows_by_index: list[dict[str, Any] | None] = [None] * len(samples)
    for fold in range(num_folds):
        eval_indices = [index for index, assigned_fold in enumerate(assignments) if assigned_fold == fold]
        train_samples = [
            sample for sample, assigned_fold in zip(samples, assignments, strict=True) if assigned_fold != fold
        ]
        retriever = SupervisedLawProfileRetriever.fit(
            documents,
            train_samples,
            fact_mode=fact_mode,
            law_text_weight=law_text_weight,
        )
        fold_rows = retrieve_supervised_for_samples(
            [samples[index] for index in eval_indices],
            retriever,
            top_k=top_k,
            fact_mode=fact_mode,
            max_fact_chars=max_fact_chars,
        )
        for index, row in zip(eval_indices, fold_rows, strict=True):
            row["oof_fold"] = fold
            row["oof_num_folds"] = num_folds
            row["oof_group_field"] = group_field
            row["oof_seed"] = seed
            rows_by_index[index] = row

    return [row for row in rows_by_index if row is not None]


def build_retrieval_query(
    sample: dict[str, Any],
    *,
    fact_mode: str = "all",
    max_fact_chars: int | None = 12000,
) -> str:
    """Build a law retrieval query from a prepared grounding sample."""
    if fact_mode not in {"all", "none"}:
        raise ValueError(f"unsupported fact_mode: {fact_mode}")
    facts: list[str] = []
    if fact_mode == "all":
        total_chars = 0
        for fact in sample.get("fact_candidates") or []:
            text = str(fact.get("text") or "")
            if max_fact_chars is not None and max_fact_chars > 0 and total_chars + len(text) > max_fact_chars:
                remaining = max_fact_chars - total_chars
                if remaining > 0:
                    facts.append(text[:remaining])
                break
            facts.append(text)
            total_chars += len(text)
    return "\n".join(
        part
        for part in [
            f'被告人：{sample.get("defendant") or ""}',
            f'裁判观点：{sample.get("court_view") or ""}',
            "案件事实：" + "\n".join(facts),
        ]
        if part.strip()
    )


def _gold_laws(row: dict[str, Any]) -> list[str]:
    if "gold_laws" in row:
        return [str(item) for item in row.get("gold_laws") or []]
    return [str(item) for item in row.get("gold", {}).get("support_laws", [])]


def _retrieved_law_ids(row: dict[str, Any]) -> list[str]:
    if "retrieved_law_ids" in row:
        return [str(item) for item in row.get("retrieved_law_ids") or []]
    return [str(item.get("id")) for item in row.get("retrieved_laws") or []]


def _normalize_ids(law_ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for law_id in law_ids:
        normalized = normalize_law_id(law_id)
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def compute_retrieval_metrics(rows: list[dict[str, Any]], *, cutoffs: Iterable[int] = DEFAULT_CUTOFFS) -> dict[str, Any]:
    """Compute recall@k metrics for multi-label law retrieval rows."""
    cutoffs = sorted(set(int(item) for item in cutoffs))
    num_samples = len(rows)
    num_gold_laws = 0
    micro_hits = {cutoff: 0 for cutoff in cutoffs}
    sample_recall = {cutoff: 0.0 for cutoff in cutoffs}
    all_hit = {cutoff: 0 for cutoff in cutoffs}
    reciprocal_rank_total = 0.0

    for row in rows:
        gold = set(_normalize_ids(_gold_laws(row)))
        retrieved = _normalize_ids(_retrieved_law_ids(row))
        num_gold_laws += len(gold)
        first_rank = 0
        for rank, law_id in enumerate(retrieved, start=1):
            if law_id in gold:
                first_rank = rank
                break
        if first_rank:
            reciprocal_rank_total += 1 / first_rank

        for cutoff in cutoffs:
            top_ids = set(retrieved[:cutoff])
            hits = len(gold & top_ids)
            micro_hits[cutoff] += hits
            sample_recall[cutoff] += hits / len(gold) if gold else 1.0
            if gold.issubset(top_ids):
                all_hit[cutoff] += 1

    metrics: dict[str, Any] = {
        "num_samples": num_samples,
        "num_gold_laws": num_gold_laws,
        "mrr": reciprocal_rank_total / num_samples if num_samples else 0.0,
    }
    for cutoff in cutoffs:
        metrics[f"micro_recall@{cutoff}"] = micro_hits[cutoff] / num_gold_laws if num_gold_laws else 0.0
        metrics[f"sample_recall@{cutoff}"] = sample_recall[cutoff] / num_samples if num_samples else 0.0
        metrics[f"all_hit@{cutoff}"] = all_hit[cutoff] / num_samples if num_samples else 0.0
    return metrics


def retrieve_for_samples(
    samples: list[dict[str, Any]],
    retriever: BM25LawRetriever,
    *,
    top_k: int,
    fact_mode: str = "all",
    max_fact_chars: int | None = 12000,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        query = build_retrieval_query(sample, fact_mode=fact_mode, max_fact_chars=max_fact_chars)
        results = retriever.search(query, top_k=top_k)
        rows.append(
            {
                "case_id": sample.get("case_id"),
                "defendant": sample.get("defendant"),
                "defendant_index": sample.get("defendant_index"),
                "court_unit_id": sample.get("court_unit_id"),
                "gold_laws": sample.get("gold_laws", []),
                "retrieved_law_ids": [result.law_id for result in results],
                "retrieved_laws": [
                    {"id": result.law_id, "rank": result.rank, "score": result.score, "text": result.text}
                    for result in results
                ],
            }
        )
    return rows


def retrieval_rows_from_results(
    samples: list[dict[str, Any]],
    results_by_sample: list[list[SearchResult]],
) -> list[dict[str, Any]]:
    if len(samples) != len(results_by_sample):
        raise ValueError("samples and results_by_sample must have the same length")
    rows: list[dict[str, Any]] = []
    for sample, results in zip(samples, results_by_sample, strict=True):
        rows.append(
            {
                "case_id": sample.get("case_id"),
                "defendant": sample.get("defendant"),
                "defendant_index": sample.get("defendant_index"),
                "court_unit_id": sample.get("court_unit_id"),
                "gold_laws": sample.get("gold_laws", []),
                "retrieved_law_ids": [result.law_id for result in results],
                "retrieved_laws": [
                    {"id": result.law_id, "rank": result.rank, "score": result.score, "text": result.text}
                    for result in results
                ],
            }
        )
    return rows


def _row_retrieved_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    items = row.get("retrieved_laws") or []
    if items:
        return [dict(item) for item in items]
    return [
        {"id": law_id, "rank": rank, "score": 0.0, "text": ""}
        for rank, law_id in enumerate(row.get("retrieved_law_ids") or [], start=1)
    ]


def fuse_retrieval_rows(
    runs: list[list[dict[str, Any]]],
    *,
    top_k: int,
    rrf_k: int = 60,
    weights: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Fuse multiple retrieval runs with reciprocal rank fusion."""
    if not runs:
        return []
    row_count = len(runs[0])
    if any(len(run) != row_count for run in runs):
        raise ValueError("all retrieval runs must have the same number of rows")
    if weights is None:
        weights = [1.0] * len(runs)
    if len(weights) != len(runs):
        raise ValueError("weights must match number of retrieval runs")

    fused_rows: list[dict[str, Any]] = []
    for row_index in range(row_count):
        base = runs[0][row_index]
        scores: dict[str, float] = {}
        texts: dict[str, str] = {}
        source_scores: dict[str, list[dict[str, Any]]] = {}
        for run_index, run in enumerate(runs):
            row = run[row_index]
            if row.get("case_id") != base.get("case_id") or row.get("court_unit_id") != base.get("court_unit_id"):
                raise ValueError(f"retrieval rows are not aligned at index {row_index}")
            for rank, item in enumerate(_row_retrieved_items(row), start=1):
                law_id = str(item.get("id"))
                scores[law_id] = scores.get(law_id, 0.0) + weights[run_index] / (rrf_k + rank)
                texts.setdefault(law_id, str(item.get("text") or ""))
                source_scores.setdefault(law_id, []).append(
                    {
                        "source": run_index,
                        "rank": int(item.get("rank") or rank),
                        "score": float(item.get("score") or 0.0),
                    }
                )
        law_ids = sorted(scores, key=lambda law_id: (-scores[law_id], law_sort_key(law_id)))[:top_k]
        fused_rows.append(
            {
                "case_id": base.get("case_id"),
                "defendant": base.get("defendant"),
                "defendant_index": base.get("defendant_index"),
                "court_unit_id": base.get("court_unit_id"),
                "gold_laws": base.get("gold_laws", []),
                "retrieved_law_ids": law_ids,
                "retrieved_laws": [
                    {
                        "id": law_id,
                        "rank": rank,
                        "score": scores[law_id],
                        "text": texts.get(law_id, ""),
                        "source_scores": source_scores.get(law_id, []),
                    }
                    for rank, law_id in enumerate(law_ids, start=1)
                ],
            }
        )
    return fused_rows
