#!/usr/bin/env python
"""Continue a grounding Causal LM with joint JSON-SFT and listwise law-ranking loss."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MODEL = "models/Qwen3-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--init-adapter", required=True, help="Trusted Flat CAUSAL_LM LoRA initialization")
    parser.add_argument("--generation-file", required=True)
    parser.add_argument("--ranking-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--group-by-length", action="store_true")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--optim", default="adamw_torch_fused")
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-generation-samples", type=int)
    parser.add_argument("--max-ranking-samples", type=int)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.25)
    parser.add_argument("--pairwise-loss-weight", type=float, default=0.5)
    parser.add_argument("--hierarchy-loss-weight", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--monitor-log")
    parser.add_argument("--monitor-steps", type=int, default=50)
    parser.add_argument("--max-system-mem-percent", type=float, default=80.0)
    parser.add_argument("--max-gpu-mem-percent", type=float, default=0.0)
    parser.add_argument("--monitor-gpu-indices")
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


def env_int(name: str, default: int, env: dict[str, str] | None = None) -> int:
    env = env or os.environ
    return int(env.get(name, default))


def is_distributed(env: dict[str, str] | None = None) -> bool:
    return env_int("WORLD_SIZE", 1, env) > 1


def should_write_artifacts(env: dict[str, str] | None = None) -> bool:
    return env_int("RANK", 0, env) == 0


def parse_gpu_indices(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def model_load_kwargs(torch_dtype: Any, device_map: str | None, env: dict[str, str] | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype, "trust_remote_code": True}
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


def marker_token_id(tokenizer: Any, marker: str) -> int:
    token_ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
    if len(token_ids) != 1:
        raise ValueError(f"ranking marker must be exactly one token, got {marker!r}: {token_ids}")
    return int(token_ids[0])


def validate_rows(generation_rows: list[dict[str, Any]], ranking_rows: list[dict[str, Any]]) -> None:
    if not generation_rows:
        raise ValueError("generation training file is empty")
    if not ranking_rows:
        raise ValueError("ranking training file is empty")
    for index, row in enumerate(generation_rows):
        messages = row.get("messages") or []
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            raise ValueError(f"generation row {index} must end with an assistant answer")
    for index, row in enumerate(ranking_rows):
        if row.get("task_type") != "law_listwise_rank":
            raise ValueError(f"ranking row {index} has an invalid task_type")
        messages = row.get("messages") or []
        if not messages or any(message.get("role") == "assistant" for message in messages):
            raise ValueError(f"ranking row {index} must contain prompt messages only")
        ranking = row.get("ranking") or {}
        count = len(ranking.get("candidate_markers") or [])
        if count < 1:
            raise ValueError(f"ranking row {index} has no candidate markers")
        if len(ranking.get("candidate_family_indices") or []) != count:
            raise ValueError(f"ranking row {index} candidate/family lengths differ")
        positives = [int(item) for item in ranking.get("exact_positive_indices") or []]
        if any(item < 0 or item >= count for item in positives):
            raise ValueError(f"ranking row {index} has an out-of-range exact positive")
        if bool(positives) == bool(ranking.get("target_is_null")):
            raise ValueError(f"ranking row {index} must choose exact positives xor NULL")


def main() -> None:
    args = parse_args()
    if args.max_length < 2:
        raise ValueError("--max-length must be >= 2")
    for name in ("ranking_loss_weight", "pairwise_loss_weight", "hierarchy_loss_weight"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    from law_grounding.resource_monitor import ResourceMonitor

    set_seed(args.seed)
    if torch.cuda.is_available() and is_distributed():
        torch.cuda.set_device(env_int("LOCAL_RANK", 0))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
    model.print_trainable_parameters()

    generation_rows = read_jsonl(args.generation_file, limit=args.max_generation_samples)
    ranking_rows = read_jsonl(args.ranking_file, limit=args.max_ranking_samples)
    validate_rows(generation_rows, ranking_rows)
    rows = [dict(row, task_type="grounding_generation") for row in generation_rows] + ranking_rows

    all_markers = sorted(
        {
            str(marker)
            for row in ranking_rows
            for marker in [*(row["ranking"]["candidate_markers"]), row["ranking"]["null_marker"]]
        }
    )
    marker_ids = {marker: marker_token_id(tokenizer, marker) for marker in all_markers}
    if len(set(marker_ids.values())) != len(marker_ids):
        raise ValueError(f"ranking markers do not map to unique token IDs: {marker_ids}")

    class JointDataset(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            row = rows[index]
            if row["task_type"] == "grounding_generation":
                messages = row["messages"]
                prompt_text = apply_chat_template(tokenizer, messages[:-1], add_generation_prompt=True)
                full_text = apply_chat_template(tokenizer, messages, add_generation_prompt=False)
                prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                encoded = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=args.max_length)
                labels = list(encoded["input_ids"])
                prompt_length = min(len(prompt_ids), len(labels))
                labels[:prompt_length] = [-100] * prompt_length
                if not any(label != -100 for label in labels):
                    raise ValueError(f"generation answer was fully truncated for row {index}")
                return {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "labels": labels,
                    "task_id": 0,
                }

            ranking = row["ranking"]
            prompt_text = apply_chat_template(tokenizer, row["messages"], add_generation_prompt=True)
            encoded = tokenizer(prompt_text, add_special_tokens=False)
            if len(encoded["input_ids"]) > args.max_length:
                raise ValueError(
                    f"ranking prompt exceeds max-length={args.max_length} for row {index}: "
                    f"{len(encoded['input_ids'])} tokens"
                )
            count = len(ranking["candidate_markers"])
            exact_mask = [False] * count
            for positive in ranking["exact_positive_indices"]:
                exact_mask[int(positive)] = True
            family_count = max((int(item) for item in ranking["candidate_family_indices"]), default=-1) + 1
            gold_family_mask = [False] * family_count
            for positive in ranking["gold_family_indices"]:
                gold_family_mask[int(positive)] = True
            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": [-100] * len(encoded["input_ids"]),
                "task_id": 1,
                "candidate_token_ids": [marker_ids[str(marker)] for marker in ranking["candidate_markers"]],
                "candidate_family_indices": [int(item) for item in ranking["candidate_family_indices"]],
                "exact_positive_mask": exact_mask,
                "target_is_null": bool(ranking["target_is_null"]),
                "gold_family_mask": gold_family_mask,
                "family_target_is_null": bool(ranking["family_target_is_null"]),
                "null_token_id": marker_ids[str(ranking["null_marker"])],
            }

    class JointCollator:
        def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
            max_length = max(len(item["input_ids"]) for item in features)
            max_candidates = max((len(item.get("candidate_token_ids", [])) for item in features), default=0)
            max_families = max((len(item.get("gold_family_mask", [])) for item in features), default=0)
            batch: dict[str, list[Any]] = {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
                "task_id": [],
                "candidate_token_ids": [],
                "candidate_mask": [],
                "candidate_family_indices": [],
                "exact_positive_mask": [],
                "target_is_null": [],
                "gold_family_mask": [],
                "family_mask": [],
                "family_target_is_null": [],
                "null_token_id": [],
            }
            for item in features:
                pad_length = max_length - len(item["input_ids"])
                batch["input_ids"].append(item["input_ids"] + [tokenizer.pad_token_id] * pad_length)
                batch["attention_mask"].append(item["attention_mask"] + [0] * pad_length)
                batch["labels"].append(item["labels"] + [-100] * pad_length)
                batch["task_id"].append(item["task_id"])
                candidate_count = len(item.get("candidate_token_ids", []))
                candidate_pad = max_candidates - candidate_count
                batch["candidate_token_ids"].append(item.get("candidate_token_ids", []) + [0] * candidate_pad)
                batch["candidate_mask"].append([True] * candidate_count + [False] * candidate_pad)
                batch["candidate_family_indices"].append(
                    item.get("candidate_family_indices", []) + [-1] * candidate_pad
                )
                batch["exact_positive_mask"].append(
                    item.get("exact_positive_mask", []) + [False] * candidate_pad
                )
                batch["target_is_null"].append(item.get("target_is_null", False))
                family_count = len(item.get("gold_family_mask", []))
                family_pad = max_families - family_count
                batch["gold_family_mask"].append(item.get("gold_family_mask", []) + [False] * family_pad)
                batch["family_mask"].append([True] * family_count + [False] * family_pad)
                batch["family_target_is_null"].append(item.get("family_target_is_null", False))
                batch["null_token_id"].append(item.get("null_token_id", 0))
            return {
                "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(batch["labels"], dtype=torch.long),
                "task_id": torch.tensor(batch["task_id"], dtype=torch.long),
                "candidate_token_ids": torch.tensor(batch["candidate_token_ids"], dtype=torch.long),
                "candidate_mask": torch.tensor(batch["candidate_mask"], dtype=torch.bool),
                "candidate_family_indices": torch.tensor(batch["candidate_family_indices"], dtype=torch.long),
                "exact_positive_mask": torch.tensor(batch["exact_positive_mask"], dtype=torch.bool),
                "target_is_null": torch.tensor(batch["target_is_null"], dtype=torch.bool),
                "gold_family_mask": torch.tensor(batch["gold_family_mask"], dtype=torch.bool),
                "family_mask": torch.tensor(batch["family_mask"], dtype=torch.bool),
                "family_target_is_null": torch.tensor(batch["family_target_is_null"], dtype=torch.bool),
                "null_token_id": torch.tensor(batch["null_token_id"], dtype=torch.long),
            }

    class JointTrainer(Trainer):
        def __init__(self, *trainer_args: Any, **trainer_kwargs: Any):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model: Any, inputs: dict[str, torch.Tensor], return_outputs: bool = False, **_: Any):
            local = dict(inputs)
            labels = local.pop("labels")
            task_id = local.pop("task_id")
            candidate_token_ids = local.pop("candidate_token_ids")
            candidate_mask = local.pop("candidate_mask")
            candidate_family_indices = local.pop("candidate_family_indices")
            exact_positive_mask = local.pop("exact_positive_mask")
            target_is_null = local.pop("target_is_null")
            gold_family_mask = local.pop("gold_family_mask")
            family_mask = local.pop("family_mask")
            family_target_is_null = local.pop("family_target_is_null")
            null_token_id = local.pop("null_token_id")
            outputs = model(**local)
            logits = outputs.logits.float()

            generation_rows = task_id == 0
            if generation_rows.any():
                generation_logits = logits[generation_rows, :-1, :].contiguous()
                generation_labels = labels[generation_rows, 1:].contiguous()
                generation_loss = F.cross_entropy(
                    generation_logits.view(-1, generation_logits.size(-1)),
                    generation_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                generation_loss = logits.sum() * 0.0

            rank_losses: list[torch.Tensor] = []
            for row_index in torch.nonzero(task_id == 1, as_tuple=False).view(-1).tolist():
                last_position = int(local["attention_mask"][row_index].sum().item()) - 1
                mask = candidate_mask[row_index]
                token_ids = candidate_token_ids[row_index][mask]
                scores = logits[row_index, last_position, token_ids]
                null_score = logits[row_index, last_position, null_token_id[row_index]].view(1)
                all_scores = torch.cat([scores, null_score], dim=0)

                positives = exact_positive_mask[row_index][mask]
                exact_target = torch.cat(
                    [positives, target_is_null[row_index].view(1)],
                    dim=0,
                )
                exact_distribution = exact_target.float() / exact_target.sum().clamp_min(1)
                exact_listwise = -(exact_distribution * F.log_softmax(all_scores, dim=0)).sum()
                positive_scores = all_scores[exact_target]
                negative_scores = all_scores[~exact_target]
                pairwise = F.softplus(-(positive_scores[:, None] - negative_scores[None, :])).mean()

                family_ids = candidate_family_indices[row_index][mask]
                valid_families = family_mask[row_index]
                family_scores = []
                for family_index in torch.nonzero(valid_families, as_tuple=False).view(-1).tolist():
                    family_scores.append(torch.logsumexp(scores[family_ids == family_index], dim=0))
                stacked_family_scores = torch.stack(family_scores) if family_scores else scores.new_empty((0,))
                all_family_scores = torch.cat([stacked_family_scores, null_score], dim=0)
                family_target = torch.cat(
                    [gold_family_mask[row_index][valid_families], family_target_is_null[row_index].view(1)],
                    dim=0,
                )
                family_distribution = family_target.float() / family_target.sum().clamp_min(1)
                hierarchy = -(family_distribution * F.log_softmax(all_family_scores, dim=0)).sum()
                rank_losses.append(
                    exact_listwise
                    + args.pairwise_loss_weight * pairwise
                    + args.hierarchy_loss_weight * hierarchy
                )

            ranking_loss = torch.stack(rank_losses).mean() if rank_losses else logits.sum() * 0.0
            loss = generation_loss + args.ranking_loss_weight * ranking_loss
            return (loss, outputs) if return_outputs else loss

    class MonitorCallback(TrainerCallback):
        def __init__(self, monitor: ResourceMonitor):
            self.monitor = monitor

        def on_train_begin(self, args_, state, control, **kwargs):
            self.monitor.log("train_begin", step=int(state.global_step))

        def on_step_end(self, args_, state, control, **kwargs):
            step = int(state.global_step)
            if args.monitor_steps > 0 and step > 0 and step % args.monitor_steps == 0:
                self.monitor.log("train_step", step=step)

        def on_train_end(self, args_, state, control, **kwargs):
            self.monitor.log("train_end", step=int(state.global_step))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        group_by_length=args.group_by_length,
        dataloader_num_workers=args.dataloader_num_workers,
        optim=args.optim,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False if is_distributed() else None,
        report_to=[],
        remove_unused_columns=False,
    )
    callbacks: list[Any] = []
    if args.monitor_steps > 0 and should_write_artifacts():
        callbacks.append(
            MonitorCallback(
                ResourceMonitor(
                    args.monitor_log or str(Path(args.output_dir) / "resource_monitor.jsonl"),
                    max_system_mem_percent=args.max_system_mem_percent,
                    max_gpu_mem_percent=args.max_gpu_mem_percent,
                    gpu_indices=parse_gpu_indices(args.monitor_gpu_indices),
                )
            )
        )

    trainer = JointTrainer(
        model=model,
        args=training_args,
        train_dataset=JointDataset(),
        data_collator=JointCollator(),
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    if should_write_artifacts():
        tokenizer.save_pretrained(args.output_dir)
        manifest = {
            "architecture": "Qwen3ForCausalLM+existing_Flat_LoRA",
            "objective": "joint_grounding_CE+first_token_listwise_rank+pairwise_margin+article_auxiliary",
            "generation_records": len(generation_rows),
            "ranking_records": len(ranking_rows),
            "marker_token_ids": marker_ids,
            "arguments": vars(args),
        }
        (Path(args.output_dir) / "joint_training_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
