#!/usr/bin/env python3
"""Evaluate a GUI action model on ScreenSpot or ScreenSpot-v2."""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from gui_agent.evaluation.sft import git_commit, load_model, sha256_file
from gui_agent.training.grpo import SYSTEM_PROMPT, normalize_pred_action, parse_action_text, truncate_at_balanced_dict
from gui_agent.training.sft import action_label_from_pred, extract_point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--data_json", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_image_side", type=int, default=448)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_samples(data_json: Path, image_dir: Path, max_samples: int) -> list[dict[str, Any]]:
    payload = json.loads(data_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list: {data_json}")
    samples = payload if max_samples <= 0 else payload[:max_samples]
    required = {"img_filename", "bbox", "instruction", "data_type", "data_source"}
    for idx, sample in enumerate(samples):
        missing = required - set(sample)
        if missing:
            raise ValueError(f"Sample {idx} is missing fields: {sorted(missing)}")
        bbox = sample["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Sample {idx} has invalid bbox: {bbox!r}")
        image_path = image_dir / sample["img_filename"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Sample {idx} image not found: {image_path}")
    return samples


def build_screenspot_prompt(processor: Any, instruction: str, width: int, height: int) -> str:
    user_text = (
        f"In this UI screenshot, I want to perform the command: {instruction}\n"
        "Return only one Python-style dict. For click actions, use exactly:\n"
        "{'action': 'click', 'coordinate': [x, y]}\n"
        "where x and y are integers in [0, 1000] normalized to the image\n"
        "(x = 0 left edge, x = 1000 right edge; y = 0 top, y = 1000 bottom).\n"
        "Do not include reasoning or extra text.\n"
        "Current step: 0/1\n"
        f"Image size: {width}x{height}\n"
        "Candidate UI boxes [y,x,h,w]: 无\n"
        "Previous actions:\n无"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": user_text}],
        },
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def resize_image(path: Path, max_image_side: int) -> tuple[Image.Image, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    original_size = image.size
    if max_image_side and max(image.size) > max_image_side:
        image.thumbnail((max_image_side, max_image_side), Image.Resampling.BICUBIC)
    return image, original_size


def generate_batch(
    model: Any,
    processor: Any,
    prompts: list[str],
    images: list[Image.Image],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    input_width = inputs["input_ids"].shape[1]
    eos_token_id = processor.tokenizer.eos_token_id
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id or eos_token_id,
            eos_token_id=eos_token_id,
        )
    completions = []
    for row in generated:
        completion_ids = row[input_width:]
        raw_text = processor.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        eos_terminated = bool(
            eos_token_id is not None
            and any(int(token_id) == int(eos_token_id) for token_id in completion_ids.tolist())
        )
        completions.append(
            {
                "raw_text": raw_text,
                "action_text": truncate_at_balanced_dict(raw_text),
                "eos_terminated": eos_terminated,
            }
        )
    return completions


def coordinate_to_pixel(point: list[float] | None, width: int, height: int) -> list[float] | None:
    if point is None or len(point) != 2:
        return None
    if not all(math.isfinite(value) for value in point):
        return None
    if not all(0.0 <= value <= 1000.0 for value in point):
        return None
    return [point[0] * width / 1000.0, point[1] * height / 1000.0]


def point_in_bbox(point: list[float] | None, bbox: list[float]) -> bool:
    if point is None:
        return False
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return x1 <= point[0] <= x2 and y1 <= point[1] <= y2


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    if not total:
        return {}
    keys = (
        "raw_strict_format",
        "parse_success",
        "eos_terminated",
        "extra_content",
        "action_correct",
        "coordinate_valid",
        "bbox_hit",
    )
    counts = {key: sum(bool(record[key]) for record in records) for key in keys}
    distances = [float(record["center_distance_px"]) for record in records if record["center_distance_px"] is not None]
    valid_coordinates = max(1, counts["coordinate_valid"])
    return {
        "total": total,
        "raw_strict_format_rate": round(counts["raw_strict_format"] / total, 6),
        "parse_success_rate": round(counts["parse_success"] / total, 6),
        "eos_termination_rate": round(counts["eos_terminated"] / total, 6),
        "extra_content_rate": round(counts["extra_content"] / total, 6),
        "action_type_accuracy": round(counts["action_correct"] / total, 6),
        "coordinate_valid_rate": round(counts["coordinate_valid"] / total, 6),
        "parameter_accuracy": round(counts["bbox_hit"] / total, 6),
        "grounding_accuracy": round(counts["bbox_hit"] / total, 6),
        "grounding_accuracy_given_valid_coordinate": round(counts["bbox_hit"] / valid_coordinates, 6),
        "mean_center_distance_px": round(sum(distances) / len(distances), 3) if distances else None,
        "counts": counts,
    }


def grouped_metrics(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        groups[str(record[field])].append(record)
    return {key: summarize_records(value) for key, value in sorted(groups.items())}


def evaluate_record(
    sample: dict[str, Any],
    completion: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    raw_text = completion["raw_text"]
    action_text = completion["action_text"]
    parsed, parse_ok = parse_action_text(action_text)
    raw_parsed, raw_parse_ok = parse_action_text(raw_text)
    normalized = normalize_pred_action(parsed) if parse_ok and parsed is not None else None
    predicted_action = action_label_from_pred(normalized)
    normalized_point = extract_point(normalized)
    pixel_point = coordinate_to_pixel(normalized_point, width, height)
    bbox = [float(value) for value in sample["bbox"]]
    center = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
    distance = None
    if pixel_point is not None:
        distance = math.dist(pixel_point, center)
    return {
        "img_filename": sample["img_filename"],
        "instruction": sample["instruction"],
        "data_type": sample["data_type"],
        "data_source": sample["data_source"],
        "image_size": [width, height],
        "bbox": bbox,
        "raw_text": raw_text,
        "action_text": action_text,
        "parsed_action": parsed,
        "predicted_action": predicted_action,
        "normalized_coordinate": normalized_point,
        "pixel_coordinate": pixel_point,
        "raw_strict_format": bool(raw_text == action_text and raw_parse_ok and raw_parsed is not None),
        "parse_success": bool(parse_ok and parsed is not None),
        "eos_terminated": bool(completion["eos_terminated"]),
        "extra_content": raw_text != action_text,
        "action_correct": predicted_action == "click",
        "coordinate_valid": pixel_point is not None,
        "bbox_hit": point_in_bbox(pixel_point, bbox),
        "center_distance_px": round(distance, 3) if distance is not None else None,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_index < num_shards")
    torch.manual_seed(args.seed)
    data_json = Path(args.data_json).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    all_samples = load_samples(data_json, image_dir, args.max_samples)
    samples = all_samples[args.shard_index :: args.num_shards]
    metadata = {
        "label": args.label,
        "base_model": str(Path(args.model_name_or_path).expanduser().resolve()),
        "adapter_path": str(Path(args.adapter_path).expanduser().resolve()) if args.adapter_path else None,
        "data_json": str(data_json),
        "data_json_sha256": sha256_file(data_json),
        "image_dir": str(image_dir),
        "dataset_samples": len(all_samples),
        "evaluated_samples": len(samples),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "batch_size": args.batch_size,
        "max_image_side": args.max_image_side,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "git_commit": git_commit(),
        "metric_note": "All samples are click grounding tasks; parameter_accuracy equals bbox-hit grounding accuracy.",
    }
    print(json.dumps({"evaluation": metadata}, ensure_ascii=False), flush=True)
    if args.dry_run:
        output_file.write_text(json.dumps({"metadata": metadata}, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    model, processor = load_model(args)
    processor.tokenizer.padding_side = "left"
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for start in range(0, len(samples), args.batch_size):
        batch_samples = samples[start : start + args.batch_size]
        images: list[Image.Image] = []
        sizes: list[tuple[int, int]] = []
        prompts: list[str] = []
        for sample in batch_samples:
            image, size = resize_image(image_dir / sample["img_filename"], args.max_image_side)
            images.append(image)
            sizes.append(size)
            prompts.append(build_screenspot_prompt(processor, sample["instruction"], *size))
        completions = generate_batch(model, processor, prompts, images, args.max_new_tokens)
        for sample, completion, size in zip(batch_samples, completions, sizes):
            records.append(evaluate_record(sample, completion, *size))
        print(
            json.dumps(
                {
                    "progress": len(records),
                    "total": len(samples),
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                }
            ),
            flush=True,
        )

    elapsed_seconds = round(time.monotonic() - started, 3)
    result = {
        "metadata": metadata,
        "elapsed_seconds": elapsed_seconds,
        "metrics": summarize_records(records),
        "by_data_type": grouped_metrics(records, "data_type"),
        "by_data_source": grouped_metrics(records, "data_source"),
        "records": records,
    }
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_file": str(output_file), "metrics": result["metrics"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
