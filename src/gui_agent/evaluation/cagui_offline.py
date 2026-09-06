#!/usr/bin/env python3
"""AgentCPM-compatible offline trajectory evaluation for a fixed CAGUI partition."""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from gui_agent.evaluation.screenspot import generate_batch
from gui_agent.evaluation.sft import git_commit, load_model, sha256_file
from gui_agent.training.grpo import (
    CaguiEpisodeDataset,
    build_prompt,
    load_resized_image,
    normalize_pred_action,
    parse_action_text,
    truncate_at_balanced_dict,
)
from gui_agent.training.sft import (
    action_label_from_pred,
    action_label_from_target,
    extract_point,
    point_hits_target_box,
    split_from_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--split", default="domestic")
    parser.add_argument("--partition", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--history_window", type=int, default=4)
    parser.add_argument("--max_ui_boxes", type=int, default=30)
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


def official_action_label(label: str) -> str:
    if label.startswith("press_"):
        return "press"
    if label in {"stop", "impossible", "wait"}:
        return "stop"
    return label


def enlarge_box(box: list[float], factor: float = 1.2) -> list[float]:
    y, x, height, width = (float(value) for value in box)
    height_delta = (factor - 1.0) * height
    width_delta = (factor - 1.0) * width
    return [
        max(0.0, y - height_delta / 2.0),
        max(0.0, x - width_delta / 2.0),
        min(1.0, height + height_delta),
        min(1.0, width + width_delta),
    ]


def xy_in_yxhw(point_xy: list[float], box_yxhw: list[float]) -> bool:
    x, y = point_xy
    box_y, box_x, box_h, box_w = box_yxhw
    return box_x <= x <= box_x + box_w and box_y <= y <= box_y + box_h


def official_click_match(
    pred_point: list[float] | None,
    gold_point: list[float] | None,
    ui_positions: list[list[float]],
) -> tuple[bool, str, float | None]:
    if pred_point is None or gold_point is None:
        return False, "missing_coordinate", None
    pred_xy = [float(pred_point[0]) / 1000.0, float(pred_point[1]) / 1000.0]
    gold_xy = [float(gold_point[0]) / 1000.0, float(gold_point[1]) / 1000.0]
    distance = math.dist(pred_xy, gold_xy)
    candidate_boxes = [enlarge_box(box) for box in ui_positions]
    target_boxes = [box for box in candidate_boxes if xy_in_yxhw(gold_xy, box)]
    if target_boxes and any(xy_in_yxhw(pred_xy, box) for box in target_boxes):
        return True, "expanded_bbox", distance
    # AgentCPM falls back to the normalized 0.14 distance threshold both when
    # the gold point has no candidate box and when the prediction misses it.
    return distance <= 0.14, "distance_0.14_fallback", distance


def official_step_match(sample: Any, normalized: dict[str, Any] | None) -> dict[str, Any]:
    strict_gold_label = action_label_from_target(sample.target)
    strict_pred_label = action_label_from_pred(normalized)
    gold_label = official_action_label(strict_gold_label)
    pred_label = official_action_label(strict_pred_label)
    type_match = pred_label == gold_label
    exact_match = False
    parameter_detail: dict[str, Any] = {}

    if type_match and gold_label == "click":
        pred_point = extract_point(normalized)
        gold_point = sample.target.get("point")
        exact_match, rule, distance = official_click_match(pred_point, gold_point, sample.ui_positions)
        parameter_detail = {
            "rule": rule,
            "normalized_distance": round(distance, 6) if distance is not None else None,
            "strict_bbox_hit": point_hits_target_box(pred_point, gold_point, sample.ui_positions),
        }
    elif type_match and gold_label == "scroll":
        pred_direction = str((normalized or {}).get("to", ""))
        gold_direction = str(sample.target.get("to", ""))
        exact_match = pred_direction == gold_direction
        parameter_detail = {"pred_direction": pred_direction, "gold_direction": gold_direction}
    elif type_match and gold_label == "input_text":
        pred_text = str((normalized or {}).get("TYPE", ""))
        gold_text = str(sample.target.get("text", ""))
        pred_normalized = pred_text.lower().strip()
        gold_normalized = gold_text.lower().strip()
        exact_match = pred_normalized in gold_normalized or gold_normalized in pred_normalized
        parameter_detail = {
            "pred_text": pred_text,
            "gold_text": gold_text,
            "strict_text_exact": pred_text == gold_text,
        }
    elif type_match and gold_label == "press":
        exact_match = strict_pred_label == strict_gold_label
        parameter_detail = {"pred_button": strict_pred_label, "gold_button": strict_gold_label}
    elif type_match and gold_label == "stop":
        exact_match = True

    return {
        "gold_action": gold_label,
        "pred_action": pred_label,
        "strict_gold_action": strict_gold_label,
        "strict_pred_action": strict_pred_label,
        "type_match": type_match,
        "exact_match": bool(exact_match),
        "parameter_detail": parameter_detail,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    total = len(records)
    episodes: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        episodes[str(record["episode_id"])].append(record)

    episode_successes = 0
    goal_progress_values = []
    episode_breakdown = []
    for episode_id, episode_records in sorted(episodes.items()):
        ordered = sorted(episode_records, key=lambda item: int(item["step_id"]))
        correct_prefix = 0
        for record in ordered:
            if not record["exact_match"]:
                break
            correct_prefix += 1
        success = correct_prefix == len(ordered)
        episode_successes += int(success)
        progress = correct_prefix / len(ordered)
        goal_progress_values.append(progress)
        episode_breakdown.append(
            {
                "episode_id": episode_id,
                "steps": len(ordered),
                "correct_prefix_steps": correct_prefix,
                "success": success,
                "goal_progress": round(progress, 6),
            }
        )

    per_action = {}
    for label in sorted({record["gold_action"] for record in records}):
        subset = [record for record in records if record["gold_action"] == label]
        per_action[label] = {
            "count": len(subset),
            "type_match": sum(record["type_match"] for record in subset),
            "exact_match": sum(record["exact_match"] for record in subset),
            "type_acc": round(sum(record["type_match"] for record in subset) / len(subset), 6),
            "exact_acc": round(sum(record["exact_match"] for record in subset) / len(subset), 6),
        }

    click_records = [record for record in records if record["gold_action"] == "click"]
    type_records = [record for record in records if record["gold_action"] == "input_text"]
    return {
        "total_steps": total,
        "total_episodes": len(episodes),
        "format_hit_rate": round(sum(record["format_hit"] for record in records) / total, 6),
        "raw_strict_format_rate": round(sum(record["raw_strict_format"] for record in records) / total, 6),
        "eos_termination_rate": round(sum(record["eos_terminated"] for record in records) / total, 6),
        "type_match": sum(record["type_match"] for record in records),
        "exact_match": sum(record["exact_match"] for record in records),
        "type_match_rate": round(sum(record["type_match"] for record in records) / total, 6),
        "exact_match_rate": round(sum(record["exact_match"] for record in records) / total, 6),
        "episode_successes": episode_successes,
        "episode_success_rate": round(episode_successes / len(episodes), 6),
        "goal_progress": round(sum(goal_progress_values) / len(goal_progress_values), 6),
        "per_action": per_action,
        "strict_click_bbox_hit_rate": (
            round(
                sum(bool(record["parameter_detail"].get("strict_bbox_hit")) for record in click_records)
                / len(click_records),
                6,
            )
            if click_records
            else None
        ),
        "strict_type_exact_rate": (
            round(
                sum(bool(record["parameter_detail"].get("strict_text_exact")) for record in type_records)
                / len(type_records),
                6,
            )
            if type_records
            else None
        ),
        "episode_breakdown": episode_breakdown,
    }


def evaluate_record(sample: Any, completion: dict[str, Any], source_index: int) -> dict[str, Any]:
    raw_text = completion["raw_text"]
    action_text = truncate_at_balanced_dict(raw_text)
    parsed, parse_ok = parse_action_text(action_text)
    raw_parsed, raw_parse_ok = parse_action_text(raw_text)
    normalized = normalize_pred_action(parsed) if parse_ok and parsed is not None else None
    match = official_step_match(sample, normalized)
    return {
        "source_index": source_index,
        "episode_id": sample.episode_id,
        "step_id": sample.step_id,
        "episode_length": sample.episode_length,
        "instruction": sample.instruction,
        "raw_text": raw_text,
        "action_text": action_text,
        "parsed_action": parsed,
        "normalized_action": normalized,
        "format_hit": bool(parse_ok and parsed is not None),
        "raw_strict_format": bool(raw_text == action_text and raw_parse_ok and raw_parsed is not None),
        "eos_terminated": bool(completion["eos_terminated"]),
        **match,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_index < num_shards")
    torch.manual_seed(args.seed)
    manifest_path = Path(args.split_manifest).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    full_dataset = CaguiEpisodeDataset(args.dataset_dir, args.split, args.history_window, 0)
    train_dataset, validation_dataset, test_dataset, manifest = split_from_manifest(
        full_dataset, manifest_path, args.split
    )
    partition_dataset = {"train": train_dataset, "validation": validation_dataset, "test": test_dataset}[
        args.partition
    ]
    all_samples = partition_dataset.samples
    if args.max_samples > 0:
        all_samples = all_samples[: args.max_samples]
    indexed_samples = list(enumerate(all_samples))[args.shard_index :: args.num_shards]
    metadata = {
        "label": args.label,
        "base_model": str(Path(args.model_name_or_path).expanduser().resolve()),
        "adapter_path": str(Path(args.adapter_path).expanduser().resolve()) if args.adapter_path else None,
        "dataset_dir": str(Path(args.dataset_dir).expanduser().resolve()),
        "split": args.split,
        "partition": args.partition,
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": sha256_file(manifest_path),
        "manifest_version": manifest["manifest_version"],
        "dataset_samples": len(all_samples),
        "evaluated_samples": len(indexed_samples),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "history_window": args.history_window,
        "max_ui_boxes": args.max_ui_boxes,
        "batch_size": args.batch_size,
        "max_image_side": args.max_image_side,
        "max_new_tokens": args.max_new_tokens,
        "bf16": args.bf16,
        "load_in_4bit": args.load_in_4bit,
        "attn_implementation": args.attn_implementation,
        "git_commit": git_commit(),
        "protocol": "AgentCPM-compatible TM/EM plus strict diagnostics",
    }
    print(json.dumps({"evaluation": metadata}, ensure_ascii=False), flush=True)
    if args.dry_run:
        output_file.write_text(json.dumps({"metadata": metadata}, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    model, processor = load_model(args)
    processor.tokenizer.padding_side = "left"
    records = []
    started = time.monotonic()
    for start in range(0, len(indexed_samples), args.batch_size):
        batch = indexed_samples[start : start + args.batch_size]
        prompts = [build_prompt(processor, sample, args.max_ui_boxes) for _, sample in batch]
        images = [load_resized_image(sample.image_path, args.max_image_side) for _, sample in batch]
        completions = generate_batch(model, processor, prompts, images, args.max_new_tokens)
        records.extend(
            evaluate_record(sample, completion, source_index)
            for (source_index, sample), completion in zip(batch, completions)
        )
        print(
            json.dumps(
                {
                    "progress": len(records),
                    "total": len(indexed_samples),
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                }
            ),
            flush=True,
        )
    elapsed_seconds = round(time.monotonic() - started, 3)
    result = {"metadata": metadata, "elapsed_seconds": elapsed_seconds, "metrics": summarize_records(records), "records": records}
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_file": str(output_file), "metrics": result["metrics"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
