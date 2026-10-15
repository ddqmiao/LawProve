#!/usr/bin/env python
"""Score joint listwise records from the first assistant-token logits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from law_grounding.joint_rank_grounding import (
    CANDIDATE_MARKERS,
    NULL_MARKER,
    aggregate_joint_rank_scores,
)
from law_grounding.law_listwise_selector import candidate_pool_metrics
from law_grounding.resource_monitor import ResourceMonitor
from scripts.train_joint_rank_grounding import apply_chat_template, marker_token_id


DEFAULT_MODEL = "models/Qwen3-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--input", required=True, help="Prepared all/{split}.jsonl ranking records")
    parser.add_argument("--output", required=True, help="Group-level score JSONL")
    parser.add_argument("--aggregate-output", help="Optional unit-level candidate-pool JSONL")
    parser.add_argument("--metrics-output", help="Optional candidate-pool metrics JSON")
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--always-keep-top-n", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--monitor-log")
    parser.add_argument("--monitor-every", type=int, default=200)
    parser.add_argument("--max-system-mem-percent", type=float, default=80.0)
    parser.add_argument("--max-gpu-mem-percent", type=float, default=90.0)
    parser.add_argument("--monitor-gpu-indices")
    return parser.parse_args()


def parse_gpu_indices(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def unit_shard(record: dict[str, Any], num_shards: int) -> int:
    meta = record.get("meta", {})
    key = "|".join(
        str(meta.get(field, ""))
        for field in ("case_id", "defendant_index", "defendant", "court_unit_id")
    )
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % num_shards


def iter_records(
    path: str | Path,
    *,
    num_shards: int,
    shard_index: int,
    limit: int | None,
) -> Iterator[dict[str, Any]]:
    selected = 0
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if unit_shard(row, num_shards) != shard_index:
                continue
            if limit is not None and selected >= limit:
                break
            selected += 1
            yield row


def iter_batches(records: Iterator[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_length < 2:
        raise ValueError("batch-size must be >= 1 and max-length must be >= 2")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require 0 <= --shard-index < --num-shards")
    if args.aggregate_output and args.limit is not None:
        raise ValueError("cannot aggregate a record-limited inference run")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding makes every prompt end at the final sequence position, so
    # Qwen3 can skip the vocabulary projection for all earlier tokens.
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    records = iter_records(
        args.input,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        limit=args.limit,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    monitor = None
    if args.monitor_every > 0:
        monitor = ResourceMonitor(
            args.monitor_log or str(output.with_suffix(".resource_monitor.jsonl")),
            max_system_mem_percent=args.max_system_mem_percent,
            max_gpu_mem_percent=args.max_gpu_mem_percent,
            gpu_indices=parse_gpu_indices(args.monitor_gpu_indices),
        )
        monitor.log(
            "infer_begin",
            extra={"num_shards": args.num_shards, "shard_index": args.shard_index},
        )

    completed = 0
    prediction_rows: list[dict[str, Any]] = []
    marker_ids = {
        marker: marker_token_id(tokenizer, marker)
        for marker in [*CANDIDATE_MARKERS, NULL_MARKER]
    }
    if len(set(marker_ids.values())) != len(marker_ids):
        raise ValueError(f"ranking markers do not map to unique token IDs: {marker_ids}")
    with output.open("w", encoding="utf-8") as f:
        for batch in iter_batches(records, args.batch_size):
            prompts = [
                apply_chat_template(tokenizer, row["messages"], add_generation_prompt=True)
                for row in batch
            ]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            if inputs["input_ids"].shape[1] > args.max_length:
                raise ValueError(
                    f"ranking prompt exceeds max-length={args.max_length}; "
                    f"longest encountered={inputs['input_ids'].shape[1]}"
                )
            inputs = inputs.to(model.device)
            with torch.inference_mode():
                logits = model(**inputs, logits_to_keep=1).logits[:, 0, :]
            for row_index, record in enumerate(batch):
                ranking = record["ranking"]
                candidate_token_ids = torch.tensor(
                    [marker_ids[str(marker)] for marker in ranking["candidate_markers"]],
                    dtype=torch.long,
                    device=logits.device,
                )
                candidate_logits = logits[row_index, candidate_token_ids].float()
                null_logit = logits[
                    row_index,
                    marker_ids[str(ranking["null_marker"])],
                ].float()
                all_logits = torch.cat([candidate_logits, null_logit.view(1)])
                probabilities = torch.softmax(all_logits, dim=0)
                prediction = {
                    "meta": record.get("meta", {}),
                    "candidate_logits": candidate_logits.tolist(),
                    "null_logit": float(null_logit.item()),
                    "candidate_scores": (candidate_logits - null_logit).tolist(),
                    "candidate_probabilities": probabilities[:-1].tolist(),
                    "null_probability": float(probabilities[-1].item()),
                }
                prediction_rows.append(prediction)
                f.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
            completed += len(batch)
            if monitor and completed % args.monitor_every < len(batch):
                monitor.log("infer_step", step=completed)
            if completed % max(args.monitor_every, 200) < len(batch):
                print(f"scored {completed}", flush=True)

    if args.aggregate_output:
        pools = aggregate_joint_rank_scores(
            prediction_rows,
            max_candidates=args.max_candidates,
            always_keep_top_n=args.always_keep_top_n,
        )
        _write_jsonl(args.aggregate_output, pools)
        metrics = candidate_pool_metrics(pools)
        metrics_text = json.dumps(metrics, ensure_ascii=False, indent=2)
        print(metrics_text)
        if args.metrics_output:
            metrics_path = Path(args.metrics_output)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(metrics_text + "\n", encoding="utf-8")
    if monitor:
        monitor.log("infer_end", step=completed)
    print(f"completed shard {args.shard_index}/{args.num_shards}: {completed} records", flush=True)


if __name__ == "__main__":
    main()
