#!/usr/bin/env python3
"""CAGUI bridge SFT for Qwen3.5-VL GUI action prediction.

This supervised bridge uses the exact CAGUI prompt/target format consumed by the
GRPO script, so the policy learns multi-step CAGUI actions before RL.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from gui_agent.data.ui_r1_warmup import UiR1WarmupDataset
from gui_agent.training.grpo import (
    CaguiEpisodeDataset,
    build_prompt,
    load_resized_image,
    normalize_pred_action,
    parse_action_text,
    reward_completion_detail,
    target_to_sft_dict,
)

try:
    from transformers.integrations import flash_attention as transformers_flash_attention
except Exception:  # pragma: no cover - flash-attn is optional unless requested.
    transformers_flash_attention = None

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-0.8B")
DEFAULT_DATASET_DIR = Path(os.environ.get("CAGUI_ROOT", REPO_ROOT / "data" / "CAGUI"))
DEFAULT_INIT_ADAPTER = ""
DEFAULT_OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", REPO_ROOT / "artifacts" / "train" / "sft-cagui"))


def patch_flash_attention_none_s_aux() -> None:
    """Work around Transformers 5.6.0 passing None through s_aux.to(...)."""
    if transformers_flash_attention is None:
        return

    def flash_attention_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        sliding_window: int | None = None,
        softcap: float | None = None,
        is_causal: bool | None = None,
        s_aux: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        if kwargs.get("output_attentions", False):
            transformers_flash_attention.logger.warning_once(
                "Flash Attention does not support `output_attentions=True`."
                " Please set your attention to `eager` if you want any of these features."
            )

        seq_len = query.shape[2]
        if any(dim == 0 for dim in query.shape):
            raise ValueError(
                "Tensor query has shape with a zero dimension.\n"
                "FlashAttention does not support inputs with dim=0.\n"
                "Please check your input shapes or use SDPA instead."
            )

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        target_dtype = transformers_flash_attention.get_target_dtype(query, module)
        is_causal = is_causal if is_causal is not None else module.is_causal

        attn_output = _flash_attention_forward(
            query,
            key,
            value,
            attention_mask,
            query_length=seq_len,
            is_causal=is_causal,
            dropout=dropout,
            softmax_scale=scaling,
            sliding_window=sliding_window,
            softcap=softcap,
            use_top_left_mask=transformers_flash_attention._use_top_left_mask,
            target_dtype=target_dtype,
            attn_implementation=module.config._attn_implementation,
            layer_idx=module.layer_idx if hasattr(module, "layer_idx") else None,
            s_aux=s_aux.to(query.dtype) if s_aux is not None else None,
            **kwargs,
        )

        return attn_output, None

    transformers_flash_attention.flash_attention_forward = flash_attention_forward
    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward


def resolve_attn_implementation(attn_implementation: str | None) -> str | None:
    if attn_implementation != "flash_attention_2":
        return attn_implementation

    if not torch.cuda.is_available():
        print("flash_attention_2 requested but CUDA is unavailable; falling back to sdpa", flush=True)
        return "sdpa"

    device_idx = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device_idx)
    if major < 8:
        device_name = torch.cuda.get_device_name(device_idx)
        print(
            "flash_attention_2 requested but this GPU is "
            f"{device_name} with compute capability {major}.{minor}; "
            "FlashAttention requires Ampere (8.0) or newer, falling back to sdpa",
            flush=True,
        )
        return "sdpa"

    patch_flash_attention_none_s_aux()
    return attn_implementation


@dataclass
class CaguiSftCollator:
    processor: Any
    max_ui_boxes: int
    max_image_side: int

    def __call__(self, samples: list[Any]) -> dict[str, torch.Tensor]:
        full_texts: list[str] = []
        images = []
        answer_texts: list[str] = []

        for sample in samples:
            prompt = build_prompt(self.processor, sample, self.max_ui_boxes)
            answer = target_to_sft_dict(sample.target)
            full_texts.append(prompt + answer)
            answer_texts.append(answer)
            images.append(load_resized_image(sample.image_path, self.max_image_side))

        batch = self.processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        for row_idx, answer_text in enumerate(answer_texts):
            answer_ids = self.processor.tokenizer(answer_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            start = find_subsequence(batch["input_ids"][row_idx], answer_ids)
            if start is None:
                raise ValueError(f"Could not find answer tokens in sample {row_idx}: {answer_text}")
            labels[row_idx, :start] = -100

        batch["labels"] = labels
        return batch


class CaguiListDataset(Dataset):
    def __init__(self, samples: list[Any]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Any:
        return self.samples[idx]


class JsonlLoggingCallback(TrainerCallback):
    def __init__(self, output_dir: str | Path):
        self.path = Path(output_dir) / "train_log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args: TrainingArguments, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not logs or not state.is_world_process_zero:
            return
        payload = {"step": state.global_step, "epoch": state.epoch}
        payload.update(logs)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def find_subsequence(sequence: torch.Tensor, pattern: torch.Tensor) -> int | None:
    if pattern.numel() == 0 or sequence.numel() < pattern.numel():
        return None
    max_start = sequence.numel() - pattern.numel()
    for start in range(max_start, -1, -1):
        if torch.equal(sequence[start : start + pattern.numel()], pattern):
            return start
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset_format", choices=("cagui", "ui_r1_warmup"), default="cagui")
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--data_file", default="", help="UI-R1 warm-up JSON file when --dataset_format=ui_r1_warmup.")
    parser.add_argument("--image_dir", default="", help="UI-R1 warm-up image directory when --dataset_format=ui_r1_warmup.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--init_adapter", default=str(DEFAULT_INIT_ADAPTER), help="Existing SFT adapter to continue from. Use empty string to initialize a new LoRA.")
    parser.add_argument("--split", default="domestic")
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--history_window", type=int, default=4)
    parser.add_argument("--max_ui_boxes", type=int, default=30)
    parser.add_argument("--max_image_side", type=int, default=672)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument(
        "--split_manifest",
        default=os.environ.get("SPLIT_MANIFEST", ""),
        help="Versioned episode split manifest. When set, it overrides val/test ratios and validates dataset membership.",
    )
    parser.add_argument("--max_eval_samples", type=int, default=256)
    parser.add_argument("--max_test_samples", type=int, default=256)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--test_generate_samples", type=int, default=None)
    parser.add_argument(
        "--skip_final_eval",
        action="store_true",
        help="Reuse the latest in-training eval metrics instead of repeating trainer.evaluate() after training.",
    )
    parser.add_argument("--eval_max_new_tokens", type=int, default=48)
    parser.add_argument("--preview_samples", type=int, default=5, help="Print this many samples before training.")

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--attn_implementation", default=None)

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", default="q_proj,v_proj")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def save_run_config(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = vars(args).copy()
    payload["argv"] = sys.argv
    payload["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    payload["world_size"] = int(os.environ.get("WORLD_SIZE", "1"))
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def split_by_episode(
    dataset: CaguiEpisodeDataset,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[CaguiListDataset, CaguiListDataset, CaguiListDataset]:
    samples = list(dataset.samples)
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("--val_ratio and --test_ratio must be non-negative and sum to less than 1.0")
    if (val_ratio == 0 and test_ratio == 0) or len(samples) < 2:
        return CaguiListDataset(samples), CaguiListDataset([]), CaguiListDataset([])

    groups: dict[str, list[Any]] = collections.defaultdict(list)
    for sample in samples:
        groups[str(sample.episode_id)].append(sample)

    episode_ids = list(groups)
    random.Random(seed).shuffle(episode_ids)
    val_count = max(1, int(round(len(episode_ids) * val_ratio))) if val_ratio > 0 else 0
    test_count = max(1, int(round(len(episode_ids) * test_ratio))) if test_ratio > 0 else 0
    if val_count + test_count >= len(episode_ids):
        available = max(0, len(episode_ids) - 1)
        val_count = min(val_count, available)
        test_count = min(test_count, max(0, available - val_count))
    val_ids = set(episode_ids[:val_count])
    test_ids = set(episode_ids[val_count : val_count + test_count])

    train_samples: list[Any] = []
    val_samples: list[Any] = []
    test_samples: list[Any] = []
    for episode_id in episode_ids:
        if episode_id in val_ids:
            val_samples.extend(groups[episode_id])
        elif episode_id in test_ids:
            test_samples.extend(groups[episode_id])
        else:
            train_samples.extend(groups[episode_id])
    return CaguiListDataset(train_samples), CaguiListDataset(val_samples), CaguiListDataset(test_samples)


def split_from_manifest(
    dataset: CaguiEpisodeDataset,
    manifest_path: str | Path,
    expected_dataset_split: str,
) -> tuple[CaguiListDataset, CaguiListDataset, CaguiListDataset, dict[str, Any]]:
    path = Path(manifest_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest.get("manifest_version") != 1:
        raise ValueError(f"Unsupported split manifest version in {path}: {manifest.get('manifest_version')!r}")
    if manifest.get("dataset_split") != expected_dataset_split:
        raise ValueError(
            f"Split manifest targets {manifest.get('dataset_split')!r}, but --split is {expected_dataset_split!r}"
        )

    partition_names = ("train", "validation", "test")
    partition_ids: dict[str, list[str]] = {}
    for name in partition_names:
        partition = manifest.get(name)
        if not isinstance(partition, dict) or not isinstance(partition.get("episode_ids"), list):
            raise ValueError(f"Split manifest {path} has no valid {name}.episode_ids list")
        ids = [str(episode_id) for episode_id in partition["episode_ids"]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Split manifest {path} contains duplicate episode IDs in {name}")
        if partition.get("episode_count") != len(ids):
            raise ValueError(f"Split manifest {path} has an incorrect {name}.episode_count")
        partition_ids[name] = ids

    id_sets = {name: set(ids) for name, ids in partition_ids.items()}
    for left_idx, left in enumerate(partition_names):
        for right in partition_names[left_idx + 1 :]:
            overlap = id_sets[left] & id_sets[right]
            if overlap:
                raise ValueError(f"Split manifest {path} overlaps {left}/{right}: {sorted(overlap)[:5]}")

    samples = list(dataset.samples)
    actual_ids = {str(sample.episode_id) for sample in samples}
    manifest_ids = set().union(*(id_sets[name] for name in partition_names))
    missing = manifest_ids - actual_ids
    unexpected = actual_ids - manifest_ids
    if missing or unexpected:
        raise ValueError(
            f"Dataset does not match split manifest {path}: "
            f"missing={sorted(missing)[:5]}, unexpected={sorted(unexpected)[:5]}"
        )

    full = manifest.get("full", {})
    if full.get("episode_count") != len(actual_ids) or full.get("sample_count") != len(samples):
        raise ValueError(
            f"Dataset counts do not match split manifest {path}: "
            f"episodes={len(actual_ids)}, samples={len(samples)}"
        )

    split_samples: dict[str, list[Any]] = {name: [] for name in partition_names}
    episode_to_partition = {
        episode_id: name for name in partition_names for episode_id in partition_ids[name]
    }
    for sample in samples:
        split_samples[episode_to_partition[str(sample.episode_id)]].append(sample)

    for name in partition_names:
        declared_count = manifest[name].get("sample_count")
        if declared_count != len(split_samples[name]):
            raise ValueError(
                f"Dataset sample count for {name} does not match split manifest {path}: "
                f"expected={declared_count}, actual={len(split_samples[name])}"
            )

    return (
        CaguiListDataset(split_samples["train"]),
        CaguiListDataset(split_samples["validation"]),
        CaguiListDataset(split_samples["test"]),
        manifest,
    )


def save_split_manifest(
    args: argparse.Namespace,
    full_dataset: CaguiEpisodeDataset,
    train_dataset: CaguiListDataset,
    val_dataset: CaguiListDataset,
    test_dataset: CaguiListDataset,
) -> Path:
    def partition(dataset: CaguiListDataset) -> dict[str, Any]:
        episode_ids = sorted({str(sample.episode_id) for sample in dataset.samples})
        return {
            "episode_count": len(episode_ids),
            "sample_count": len(dataset),
            "episode_ids": episode_ids,
        }

    manifest = {
        "dataset_dir": str(Path(args.dataset_dir).expanduser().resolve()),
        "dataset_split": args.split,
        "coordinate_convention": "model_and_api_use_normalized_xy_0_1000; source_cagui_uses_yx",
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "max_episodes": args.max_episodes,
        "full": {
            "episode_count": len({str(sample.episode_id) for sample in full_dataset.samples}),
            "sample_count": len(full_dataset),
        },
        "train": partition(train_dataset),
        "validation": partition(val_dataset),
        "test": partition(test_dataset),
    }
    manifest_path = Path(args.output_dir) / "split_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=True)
    return manifest_path


def save_fixed_split_manifest(args: argparse.Namespace, manifest: dict[str, Any]) -> Path:
    manifest_path = Path(args.output_dir) / "split_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=True)
    return manifest_path


def save_ui_r1_manifest(args: argparse.Namespace, dataset: UiR1WarmupDataset) -> Path:
    manifest_path = Path(args.output_dir) / "dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(dataset.summary, file, ensure_ascii=False, indent=2, sort_keys=True)
    return manifest_path


def limit_dataset(dataset: CaguiListDataset, limit: int) -> CaguiListDataset:
    if limit <= 0 or len(dataset) <= limit:
        return dataset
    return CaguiListDataset(dataset.samples[:limit])


def action_label_from_target(target: dict[str, Any]) -> str:
    kind = str(target.get("kind", "UNKNOWN"))
    if kind == "POINT":
        return "click"
    if kind == "SWIPE":
        return "scroll"
    if kind == "TYPE":
        return "input_text"
    if kind == "PRESS":
        return f"press_{str(target.get('press', '')).lower()}"
    if kind == "STATUS":
        status = str(target.get("status", "finish"))
        return "stop" if status == "finish" else "impossible"
    if kind == "WAIT":
        return "wait"
    return "unknown"


def action_label_from_pred(obj: dict[str, Any] | None) -> str:
    if not obj:
        return "parse_failed"
    if "TYPE" in obj:
        return "input_text"
    if "PRESS" in obj:
        return f"press_{str(obj.get('PRESS', '')).lower()}"
    if "STATUS" in obj:
        status = str(obj.get("STATUS", ""))
        return "stop" if status == "finish" else "impossible"
    if "POINT" in obj:
        return "scroll" if "to" in obj else "click"
    if "duration" in obj:
        return "wait"
    return str(obj.get("action", "unknown")).lower()


def extract_point(obj: dict[str, Any] | None) -> list[float] | None:
    if not obj:
        return None
    point = obj.get("POINT")
    if isinstance(point, list) and len(point) == 2:
        try:
            return [float(point[0]), float(point[1])]
        except Exception:
            return None
    return None


def point_distance(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b:
        return None
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)


def point_hits_target_box(pred_point: list[float] | None, gold_point: list[float] | None, boxes: list[list[float]]) -> bool:
    if pred_point is None or gold_point is None:
        return False
    px, py = pred_point[0] / 1000.0, pred_point[1] / 1000.0
    gx, gy = gold_point[0] / 1000.0, gold_point[1] / 1000.0
    for box in boxes:
        y, x, h, w = box
        if x <= gx <= x + w and y <= gy <= y + h:
            return x <= px <= x + w and y <= py <= y + h
    dist = point_distance(pred_point, gold_point)
    return dist is not None and dist <= 50.0


def generate_completion(model: Any, processor: Any, prompt: str, image: Any, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        )
    completion_ids = generated[0][input_len:]
    return processor.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def evaluate_generation_metrics(
    model: Any,
    processor: Any,
    dataset: CaguiListDataset,
    args: argparse.Namespace,
    sample_limit: int,
    prefix: str = "eval",
) -> dict[str, Any]:
    total = min(max(0, sample_limit), len(dataset))
    if total == 0:
        return {}

    model.eval()
    parse_ok = 0
    action_ok = 0
    args_ok = 0
    reward_sum = 0.0
    click_distances: list[float] = []
    click_hits = 0
    click_total = 0
    text_ok = 0
    text_total = 0
    label_totals: collections.Counter[str] = collections.Counter()
    label_correct: collections.Counter[str] = collections.Counter()
    confusion: collections.Counter[str] = collections.Counter()
    examples: list[dict[str, Any]] = []

    for idx in range(total):
        sample = dataset[idx]
        prompt = build_prompt(processor, sample, args.max_ui_boxes)
        image = load_resized_image(sample.image_path, args.max_image_side)
        completion = generate_completion(model, processor, prompt, image, args.eval_max_new_tokens)
        parsed, ok = parse_action_text(completion)
        normalized = normalize_pred_action(parsed) if ok and parsed is not None else None
        reward, detail = reward_completion_detail(completion, sample.target, sample.ui_positions)
        reward_sum += reward
        parse_ok += int(ok and parsed is not None)
        args_ok += int(float(detail.get("args_reward", 0.0)) >= 1.0)

        gold_label = action_label_from_target(sample.target)
        pred_label = action_label_from_pred(normalized)
        label_totals[gold_label] += 1
        confusion[f"{gold_label}->{pred_label}"] += 1
        if pred_label == gold_label:
            action_ok += 1
            label_correct[gold_label] += 1

        if gold_label == "click":
            click_total += 1
            pred_point = extract_point(normalized)
            gold_point = sample.target.get("point")
            distance = point_distance(pred_point, gold_point)
            if distance is not None:
                click_distances.append(distance)
            click_hits += int(point_hits_target_box(pred_point, gold_point, sample.ui_positions))

        if gold_label == "input_text":
            text_total += 1
            pred_text = str((normalized or {}).get("TYPE", ""))
            text_ok += int(pred_text == str(sample.target.get("text", "")))

        if len(examples) < 8 and (pred_label != gold_label or not ok):
            examples.append(
                {
                    "idx": idx,
                    "episode_id": sample.episode_id,
                    "step_id": sample.step_id,
                    "target": target_to_sft_dict(sample.target),
                    "completion": completion,
                    "reward": round(reward, 4),
                    "detail": detail,
                }
            )

    per_action = {
        label: {
            "accuracy": round(label_correct[label] / max(1, count), 4),
            "total": count,
            "correct": label_correct[label],
        }
        for label, count in sorted(label_totals.items())
    }
    metrics: dict[str, Any] = {
        f"{prefix}_generate_samples": total,
        "parse_success_rate": round(parse_ok / total, 4),
        "format_valid_rate": round(parse_ok / total, 4),
        "action_type_accuracy": round(action_ok / total, 4),
        "argument_accuracy": round(args_ok / total, 4),
        "reward_mean": round(reward_sum / total, 4),
        "per_action_type": per_action,
        "confusion": dict(sorted(confusion.items())),
        "bad_examples": examples,
    }
    if click_total:
        metrics["click"] = {
            "total": click_total,
            "hit_rate": round(click_hits / click_total, 4),
            "mean_distance": round(sum(click_distances) / len(click_distances), 4) if click_distances else None,
        }
    if text_total:
        metrics["input_text"] = {"total": text_total, "exact_match": round(text_ok / text_total, 4)}
    model.train()
    return metrics


def dataset_distribution(dataset: CaguiListDataset) -> dict[str, Any]:
    counts = collections.Counter(action_label_from_target(sample.target) for sample in dataset.samples)
    episodes = {sample.episode_id for sample in dataset.samples}
    return {
        "samples": len(dataset),
        "episodes": len(episodes),
        "action_counts": dict(sorted(counts.items())),
    }


def latest_logged_eval_metrics(log_history: list[dict[str, Any]]) -> dict[str, Any]:
    for entry in reversed(log_history):
        metrics = {key: value for key, value in entry.items() if key.startswith("eval_")}
        if metrics:
            return metrics
    return {}


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    is_primary_process = rank == 0
    if is_primary_process:
        save_run_config(args)
    torch.manual_seed(args.seed)

    fixed_manifest: dict[str, Any] | None = None
    if args.dataset_format == "ui_r1_warmup":
        if not args.data_file or not args.image_dir:
            raise ValueError("--data_file and --image_dir are required for --dataset_format=ui_r1_warmup")
        if args.split_manifest:
            raise ValueError("--split_manifest is only valid for --dataset_format=cagui")
        full_dataset = UiR1WarmupDataset(args.data_file, args.image_dir)
        train_dataset = CaguiListDataset(list(full_dataset.samples))
        raw_val_dataset = CaguiListDataset([])
        raw_test_dataset = CaguiListDataset([])
        split_manifest_path = Path(args.output_dir) / "dataset_manifest.json"
        if is_primary_process:
            split_manifest_path = save_ui_r1_manifest(args, full_dataset)
    else:
        full_dataset = CaguiEpisodeDataset(args.dataset_dir, args.split, args.history_window, args.max_episodes)
        if args.split_manifest:
            train_dataset, raw_val_dataset, raw_test_dataset, fixed_manifest = split_from_manifest(
                full_dataset,
                args.split_manifest,
                args.split,
            )
        else:
            train_dataset, raw_val_dataset, raw_test_dataset = split_by_episode(
                full_dataset,
                args.val_ratio,
                args.test_ratio,
                args.seed,
            )
        split_manifest_path = Path(args.output_dir) / "split_manifest.json"
        if is_primary_process:
            if fixed_manifest is not None:
                split_manifest_path = save_fixed_split_manifest(args, fixed_manifest)
            else:
                split_manifest_path = save_split_manifest(
                    args,
                    full_dataset,
                    train_dataset,
                    raw_val_dataset,
                    raw_test_dataset,
                )
    eval_dataset = limit_dataset(raw_val_dataset, args.max_eval_samples)
    test_dataset = limit_dataset(raw_test_dataset, args.max_test_samples)
    print(f"loaded {len(full_dataset)} {args.dataset_format} SFT samples", flush=True)
    print(
        json.dumps(
            {
                "dataset": {
                    "train": dataset_distribution(train_dataset),
                    "val": dataset_distribution(eval_dataset),
                    "test": dataset_distribution(test_dataset),
                    "raw_val": dataset_distribution(raw_val_dataset),
                    "raw_test": dataset_distribution(raw_test_dataset),
                    "val_ratio": args.val_ratio,
                    "test_ratio": args.test_ratio,
                    "split_manifest": str(split_manifest_path),
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    preview_count = min(max(0, args.preview_samples), len(train_dataset))
    for idx in range(preview_count):
        sample = train_dataset[idx]
        print(
            json.dumps(
                {
                    "preview_idx": idx,
                    "episode_id": sample.episode_id,
                    "step_id": sample.step_id,
                    "instruction": sample.instruction,
                    "target": target_to_sft_dict(sample.target),
                    "history_len": len(sample.history),
                    "image_path": sample.image_path,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if args.dry_run:
        return

    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    attn_implementation = resolve_attn_implementation(args.attn_implementation)
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if args.load_in_4bit:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        model_kwargs["device_map"] = {"": local_rank} if local_rank >= 0 else "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype if dtype in (torch.float16, torch.bfloat16) else torch.bfloat16,
        )

    model = AutoModelForImageTextToText.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    init_adapter = Path(args.init_adapter) if args.init_adapter else None
    if init_adapter and (init_adapter / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)
        print(f"loaded trainable init adapter from {init_adapter}", flush=True)
    else:
        if init_adapter:
            print(f"init adapter not found at {init_adapter}; initializing new LoRA", flush=True)
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[x.strip() for x in args.target_modules.split(",") if x.strip()],
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if len(eval_dataset) > 0 else "no",
        eval_steps=args.eval_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        dataloader_prefetch_factor=2 if args.dataloader_num_workers > 0 else None,
        ddp_find_unused_parameters=False,
        disable_tqdm=False,
        logging_first_step=True,
        seed=args.seed,
    )

    print(
        json.dumps(
            {
                "train_config": {
                    "output_dir": args.output_dir,
                    "dataset_format": args.dataset_format,
                    "init_adapter": args.init_adapter,
                    "max_steps": args.max_steps,
                    "num_train_epochs": args.num_train_epochs,
                    "per_device_train_batch_size": args.per_device_train_batch_size,
                    "per_device_eval_batch_size": args.per_device_eval_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "learning_rate": args.learning_rate,
                    "max_image_side": args.max_image_side,
                    "history_window": args.history_window,
                    "max_ui_boxes": args.max_ui_boxes,
                    "load_in_4bit": args.load_in_4bit,
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "eval_steps": args.eval_steps,
                    "max_eval_samples": args.max_eval_samples,
                    "max_test_samples": args.max_test_samples,
                    "test_generate_samples": args.test_generate_samples,
                    "log_file": str(Path(args.output_dir) / "train_log.jsonl"),
                    "metrics_file": str(Path(args.output_dir) / "metrics.json"),
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        data_collator=CaguiSftCollator(
            processor=processor,
            max_ui_boxes=args.max_ui_boxes,
            max_image_side=args.max_image_side,
        ),
        callbacks=[JsonlLoggingCallback(args.output_dir)],
    )
    trainer.train()
    if len(eval_dataset) == 0:
        eval_loss_metrics = {}
    elif args.skip_final_eval:
        eval_loss_metrics = latest_logged_eval_metrics(trainer.state.log_history)
    else:
        eval_loss_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    if trainer.is_world_process_zero():
        test_generate_samples = args.test_generate_samples
        if test_generate_samples is None:
            test_generate_samples = len(test_dataset)
        test_metrics = (
            evaluate_generation_metrics(
                model,
                processor,
                test_dataset,
                args,
                sample_limit=test_generate_samples,
                prefix="test",
            )
            if len(test_dataset) > 0
            else {}
        )
        metrics = {
            "train": dataset_distribution(train_dataset),
            "val": dataset_distribution(eval_dataset),
            "test": dataset_distribution(test_dataset),
            "eval_loss": eval_loss_metrics,
            "test_generation": test_metrics,
        }
        metrics_path = Path(args.output_dir) / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(
            json.dumps(
                {
                    "metrics_file": str(metrics_path),
                    "test_generation": test_metrics,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        processor.save_pretrained(args.output_dir)
        print(f"done. {args.dataset_format} SFT adapter saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
