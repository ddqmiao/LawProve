#!/usr/bin/env python
"""Run Qwen inference for prepared grounding samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from law_grounding.data import (
    PROMPT_VARIANTS,
    apply_retrieval_candidates_to_samples,
    build_inference_messages,
    load_law_map,
    read_jsonl,
)
from law_grounding.postprocess import sanitize_prediction
from law_grounding.resource_monitor import ResourceMonitor


DEFAULT_MODEL = "models/Qwen3-4B"
DEFAULT_LAW_MAP = "data/criminal_law_map.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", help="Optional LoRA adapter path")
    parser.add_argument("--input", required=True, help="Prepared samples JSONL")
    parser.add_argument("--output", required=True, help="Predictions JSONL")
    parser.add_argument("--limit", type=int, help="Optional number of samples for smoke tests")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of samples to generate at once")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default="default")
    parser.add_argument("--retrieval-candidates", help="Optional retrieval JSONL used to replace law candidates")
    parser.add_argument("--retrieval-top-k", type=int, default=50, help="Number of retrieved law candidates to keep")
    parser.add_argument("--law-map", default=DEFAULT_LAW_MAP, help="Law map used to format retrieved candidate text")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--monitor-log", help="JSONL resource monitor log path")
    parser.add_argument("--monitor-every", type=int, default=20, help="Log resources every N generated samples; <=0 disables")
    parser.add_argument("--max-system-mem-percent", type=float, default=80.0, help="Stop when system memory usage reaches this percent; <=0 disables")
    parser.add_argument("--max-gpu-mem-percent", type=float, default=80.0, help="Stop when GPU memory usage reaches this percent; <=0 disables")
    parser.add_argument(
        "--monitor-gpu-indices",
        help="Comma-separated physical GPU indices to guard; default checks all GPUs reported by nvidia-smi",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser()
    return parser.parse_args()


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model(model_path: str, adapter_path: str | None, device_map: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def ensure_batch_padding(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


def decode_batch_new_tokens(tokenizer: Any, generated: torch.Tensor, prompt_width: int) -> list[str]:
    outputs: list[str] = []
    for row in generated:
        new_tokens = row[prompt_width:]
        outputs.append(tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True).strip())
    return outputs


def parse_gpu_indices(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def iter_batches(items: list[dict[str, Any]], batch_size: int) -> Any:
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def main() -> None:
    args = parse_args()
    samples = read_jsonl(args.input)
    if args.limit is not None:
        samples = samples[: args.limit]

    if args.retrieval_candidates:
        law_map = load_law_map(args.law_map) if args.law_map else None
        samples = apply_retrieval_candidates_to_samples(
            samples,
            read_jsonl(args.retrieval_candidates),
            law_map,
            top_k=args.retrieval_top_k,
        )

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    tokenizer, model = load_model(args.model, args.adapter, args.device_map)
    ensure_batch_padding(tokenizer)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    monitor = None
    if args.monitor_every > 0:
        monitor_log = args.monitor_log or str(output.with_suffix(".resource_monitor.jsonl"))
        monitor = ResourceMonitor(
            monitor_log,
            max_system_mem_percent=args.max_system_mem_percent,
            max_gpu_mem_percent=args.max_gpu_mem_percent,
            gpu_indices=parse_gpu_indices(args.monitor_gpu_indices),
        )
        monitor.log("infer_begin", extra={"num_samples": len(samples)})

    do_sample = args.temperature > 0
    with output.open("w", encoding="utf-8") as f:
        next_monitor_step = args.monitor_every
        next_progress_step = 20
        for start, batch in iter_batches(samples, args.batch_size):
            prompts = [
                apply_chat_template(tokenizer, build_inference_messages(sample, prompt_variant=args.prompt_variant))
                for sample in batch
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            prompt_width = inputs["input_ids"].shape[-1]
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = args.temperature
            with torch.no_grad():
                generated = model.generate(**inputs, **generation_kwargs)
            raw_outputs = decode_batch_new_tokens(tokenizer, generated, prompt_width)
            for sample, raw_output in zip(batch, raw_outputs, strict=True):
                parsed = sanitize_prediction(raw_output, sample)
                row = {
                    "case_id": sample["case_id"],
                    "defendant": sample["defendant"],
                    "defendant_index": sample["defendant_index"],
                    "court_unit_id": sample["court_unit_id"],
                    "gold": {
                        "support_facts": sample["gold_facts"],
                        "support_laws": sample["gold_laws"],
                    },
                    **parsed,
                }
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            completed = start + len(batch)
            if monitor:
                while next_monitor_step <= completed:
                    monitor.log("infer_step", step=next_monitor_step)
                    next_monitor_step += args.monitor_every
            while next_progress_step <= completed:
                print(f"generated {next_progress_step}/{len(samples)}", flush=True)
                next_progress_step += 20
    if monitor:
        monitor.log("infer_end", step=len(samples))


if __name__ == "__main__":
    main()
