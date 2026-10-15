#!/usr/bin/env python
"""Evaluate neural embedding retrieval recall on prepared grounding samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from law_grounding.data import load_law_map, read_jsonl, write_jsonl
from law_grounding.law_retrieval import (
    DEFAULT_CUTOFFS,
    EmbeddingLawRetriever,
    build_law_corpus,
    build_retrieval_query,
    compute_retrieval_metrics,
    retrieval_rows_from_results,
)


DEFAULT_LAW_MAP = "data/criminal_law_map.json"
DEFAULT_EMBEDDING_MODEL = "models/Qwen3-Embedding-0.6B"
DEFAULT_QUERY_PREFIX = "为这个刑事裁判观点检索相关刑法条文："
DEFAULT_LAW_PREFIX = "刑法条文："


def parse_cutoffs(value: str) -> list[int]:
    cutoffs = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not cutoffs:
        raise argparse.ArgumentTypeError("at least one cutoff is required")
    return cutoffs


class TransformerTextEmbedder:
    """Lazy transformers-based text embedder with no import-time torch dependency."""

    def __init__(self, model_path: str, *, device: str = "auto", max_length: int = 1024, pooling: str = "last"):
        if pooling not in {"last", "mean"}:
            raise ValueError(f"unsupported pooling: {pooling}")
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.max_length = max_length
        self.pooling = pooling
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    def _pool(self, hidden_states: Any, attention_mask: Any) -> Any:
        torch = self.torch
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            return summed / counts
        last_indices = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, last_indices]

    def encode(self, texts: list[str], *, batch_size: int = 16) -> Any:
        torch = self.torch
        vectors = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded, output_hidden_states=True, return_dict=True)
                hidden = getattr(outputs, "last_hidden_state", None)
                if hidden is None:
                    hidden = outputs.hidden_states[-1]
                pooled = self._pool(hidden, encoded["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.float().cpu())
        return torch.cat(vectors, dim=0).numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--law-map", default=DEFAULT_LAW_MAP, help="criminal_law_map.json path")
    parser.add_argument("--samples", required=True, help="Prepared samples JSONL path")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Local embedding model path")
    parser.add_argument("--top-k", type=int, default=50, help="Number of laws to retrieve per sample")
    parser.add_argument(
        "--cutoffs",
        type=parse_cutoffs,
        default=list(DEFAULT_CUTOFFS),
        help="Comma-separated recall cutoffs, e.g. 5,10,20,50",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    parser.add_argument("--max-length", type=int, default=1024, help="Tokenizer max length")
    parser.add_argument("--pooling", choices=("last", "mean"), default="last")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--query-facts",
        choices=("all", "none"),
        default="none",
        help="Whether retrieval query includes all fact candidates or only defendant/court view",
    )
    parser.add_argument("--max-fact-chars", type=int, default=12000)
    parser.add_argument("--query-prefix", default=DEFAULT_QUERY_PREFIX)
    parser.add_argument("--law-prefix", default=DEFAULT_LAW_PREFIX)
    parser.add_argument("--output", help="Optional metrics JSON output path")
    parser.add_argument("--retrieved-output", help="Optional retrieved law JSONL output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    law_map = load_law_map(args.law_map)
    samples = read_jsonl(args.samples)
    documents = build_law_corpus(law_map)
    max_fact_chars = None if args.max_fact_chars <= 0 else args.max_fact_chars
    law_texts = [args.law_prefix + document.search_text for document in documents]
    query_texts = [
        args.query_prefix
        + build_retrieval_query(sample, fact_mode=args.query_facts, max_fact_chars=max_fact_chars)
        for sample in samples
    ]

    embedder = TransformerTextEmbedder(
        args.embedding_model,
        device=args.device,
        max_length=args.max_length,
        pooling=args.pooling,
    )
    law_embeddings = embedder.encode(law_texts, batch_size=args.batch_size)
    query_embeddings = embedder.encode(query_texts, batch_size=args.batch_size)
    retriever = EmbeddingLawRetriever(documents, law_embeddings)
    rows = retrieval_rows_from_results(samples, retriever.search_many(query_embeddings, top_k=args.top_k))
    metrics = compute_retrieval_metrics(rows, cutoffs=args.cutoffs)
    metrics.update(
        {
            "retriever": "embedding",
            "embedding_model": str(args.embedding_model),
            "top_k": args.top_k,
            "cutoffs": args.cutoffs,
            "query_facts": args.query_facts,
            "pooling": args.pooling,
            "max_length": args.max_length,
            "num_law_documents": len(documents),
            "samples": str(args.samples),
            "law_map": str(args.law_map),
        }
    )
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if args.retrieved_output:
        write_jsonl(args.retrieved_output, rows)


if __name__ == "__main__":
    main()
