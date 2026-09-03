#!/usr/bin/env python3
"""Evaluate a base model or SFT adapter on a fixed CAGUI test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from gui_agent.training.grpo import CaguiEpisodeDataset
from gui_agent.training.sft import (
    REPO_ROOT,
    dataset_distribution,
    evaluate_generation_metrics,
    resolve_attn_implementation,
    split_from_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument(
        "--split_manifest",
        default=str(REPO_ROOT / "configs" / "splits" / "cagui-domestic-seed42-v1.json"),
    )
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--split", default="domestic")
    parser.add_argument("--history_window", type=int, default=4)
    parser.add_argument("--max_ui_boxes", type=int, default=30)
    parser.add_argument("--max_image_side", type=int, default=448)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--max_samples", type=int, default=0, help="0 evaluates the complete fixed test split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "device_map": "auto",
    }
    attn_implementation = resolve_attn_implementation(args.attn_implementation)
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForImageTextToText.from_pretrained(args.model_name_or_path, **model_kwargs)
    if args.adapter_path:
        adapter_path = Path(args.adapter_path).expanduser().resolve()
        if not (adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.config.use_cache = True
    model.eval()
    return model, processor


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    manifest_path = Path(args.split_manifest).expanduser().resolve()
    output_path = Path(args.output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    full_dataset = CaguiEpisodeDataset(args.dataset_dir, args.split, args.history_window, 0)
    _, _, test_dataset, manifest = split_from_manifest(full_dataset, manifest_path, args.split)
    sample_limit = len(test_dataset) if args.max_samples <= 0 else min(args.max_samples, len(test_dataset))
    metadata = {
        "label": args.label,
        "base_model": str(Path(args.model_name_or_path).expanduser().resolve()),
        "adapter_path": str(Path(args.adapter_path).expanduser().resolve()) if args.adapter_path else None,
        "dataset_dir": str(Path(args.dataset_dir).expanduser().resolve()),
        "dataset_split": args.split,
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": sha256_file(manifest_path),
        "git_commit": git_commit(),
        "seed": args.seed,
        "max_image_side": args.max_image_side,
        "max_ui_boxes": args.max_ui_boxes,
        "max_new_tokens": args.max_new_tokens,
        "test": dataset_distribution(test_dataset),
        "evaluated_samples": sample_limit,
        "manifest_version": manifest["manifest_version"],
    }
    print(json.dumps({"evaluation": metadata}, ensure_ascii=False), flush=True)
    if args.dry_run:
        output_path.write_text(json.dumps({"metadata": metadata}, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    model, processor = load_model(args)
    metric_args = SimpleNamespace(
        max_ui_boxes=args.max_ui_boxes,
        max_image_side=args.max_image_side,
        eval_max_new_tokens=args.max_new_tokens,
    )
    started = time.monotonic()
    generation_metrics = evaluate_generation_metrics(
        model,
        processor,
        test_dataset,
        metric_args,
        sample_limit=sample_limit,
        prefix="test",
    )
    elapsed_seconds = round(time.monotonic() - started, 3)
    result = {
        "metadata": metadata,
        "elapsed_seconds": elapsed_seconds,
        "test_generation": generation_metrics,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_file": str(output_path), "elapsed_seconds": elapsed_seconds}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
