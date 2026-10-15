#!/usr/bin/env python
"""Train a BF16 LoRA adapter for Qwen grounding SFT data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from law_grounding.resource_monitor import ResourceMonitor


DEFAULT_MODEL = "models/Qwen3-4B"
DEFAULT_OUTPUT = "outputs/lora"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--init-adapter",
        help="Optional existing LoRA adapter used as the trainable initialization instead of a fresh adapter.",
    )
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--valid-file")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device-map", default="auto", help="Single-process device_map; omitted under torchrun DDP. Use 'none' to disable.")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--group-by-length",
        action="store_true",
        help="Bucket examples by token length to reduce dynamic-padding waste.",
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--optim", default="adamw_torch", help="Transformers optimizer name")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=-1, help="Set to a positive value for smoke training")
    parser.add_argument("--resume-from-checkpoint", help="Optional Trainer checkpoint path to resume from")
    parser.add_argument("--max-train-samples", type=int, help="Optional training sample cap for smoke tests")
    parser.add_argument("--max-valid-samples", type=int, help="Optional validation sample cap for smoke tests")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for model initialization and training")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--monitor-log", help="JSONL resource monitor log path")
    parser.add_argument("--monitor-steps", type=int, default=20, help="Log resources every N training steps; <=0 disables")
    parser.add_argument("--max-system-mem-percent", type=float, default=80.0, help="Stop when system memory usage reaches this percent; <=0 disables")
    parser.add_argument("--max-gpu-mem-percent", type=float, default=80.0, help="Stop when GPU memory usage reaches this percent; <=0 disables")
    parser.add_argument(
        "--monitor-gpu-indices",
        help="Comma-separated physical GPU indices to guard; default checks all GPUs reported by nvidia-smi",
    )
    return parser.parse_args()


def read_jsonl(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def parse_gpu_indices(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def env_int(name: str, default: int, env: dict[str, str] | None = None) -> int:
    env = env or os.environ
    return int(env.get(name, default))


def is_distributed(env: dict[str, str] | None = None) -> bool:
    return env_int("WORLD_SIZE", 1, env) > 1


def should_write_monitor(env: dict[str, str] | None = None) -> bool:
    return env_int("RANK", 0, env) == 0


def configure_torchrun_device(env: dict[str, str] | None = None) -> None:
    if torch.cuda.is_available() and is_distributed(env):
        torch.cuda.set_device(env_int("LOCAL_RANK", 0, env))


def model_load_kwargs(torch_dtype: Any, device_map: str | None, env: dict[str, str] | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if not is_distributed(env) and device_map and device_map.lower() != "none":
        kwargs["device_map"] = device_map
    return kwargs


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


class ChatSftDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        messages = self.rows[index]["messages"]
        prompt_text = apply_chat_template(self.tokenizer, messages[:-1], add_generation_prompt=True)
        full_text = apply_chat_template(self.tokenizer, messages, add_generation_prompt=False)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full = self.tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=self.max_length)
        input_ids = full["input_ids"]
        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        return {
            "input_ids": input_ids,
            "attention_mask": full["attention_mask"],
            "labels": labels,
        }


@dataclass
class DataCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad_len = max_len - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad_len)
            batch["labels"].append(item["labels"] + [-100] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


class ResourceMonitorCallback(TrainerCallback):
    def __init__(self, monitor: ResourceMonitor, every_steps: int):
        self.monitor = monitor
        self.every_steps = every_steps

    def on_train_begin(self, args, state, control, **kwargs):
        self.monitor.log("train_begin", step=int(state.global_step))

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if self.every_steps > 0 and step > 0 and step % self.every_steps == 0:
            self.monitor.log("train_step", step=step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.monitor.log("eval", step=int(state.global_step), extra=metrics or {})

    def on_save(self, args, state, control, **kwargs):
        self.monitor.log("save", step=int(state.global_step))

    def on_train_end(self, args, state, control, **kwargs):
        self.monitor.log("train_end", step=int(state.global_step))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_torchrun_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_load_kwargs(
            torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            args.device_map,
        ),
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_rows = read_jsonl(args.train_file, limit=args.max_train_samples)
    valid_rows = read_jsonl(args.valid_file, limit=args.max_valid_samples) if args.valid_file else []

    train_dataset = ChatSftDataset(train_rows, tokenizer, args.max_length)
    eval_dataset = ChatSftDataset(valid_rows, tokenizer, args.max_length) if valid_rows else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        group_by_length=args.group_by_length,
        dataloader_num_workers=args.dataloader_num_workers,
        optim=args.optim,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False if is_distributed() else None,
        report_to=[],
        remove_unused_columns=False,
    )
    callbacks = []
    if args.monitor_steps > 0 and should_write_monitor():
        monitor_log = args.monitor_log or str(Path(args.output_dir) / "resource_monitor.jsonl")
        gpu_indices = parse_gpu_indices(args.monitor_gpu_indices)
        callbacks.append(
            ResourceMonitorCallback(
                ResourceMonitor(
                    monitor_log,
                    max_system_mem_percent=args.max_system_mem_percent,
                    max_gpu_mem_percent=args.max_gpu_mem_percent,
                    gpu_indices=gpu_indices,
                ),
                args.monitor_steps,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(tokenizer),
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
