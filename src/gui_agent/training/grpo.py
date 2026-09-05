#!/usr/bin/env python3
"""GRPO training for CAGUI step-wise GUI actions.

This script follows the GRPO stage design:
- step-wise prediction from instruction + history + screenshot
- compact JSON action format
- grouped policy-gradient updates with per-action-type rewards
- single GPU friendly QLoRA / LoRA setup
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import difflib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

try:  # Keep --dry_run usable in shells that have not activated the train env.
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover
    torch = None
    F = None
    DataLoader = None
    Dataset = object


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = Path(os.environ.get("CAGUI_ROOT", REPO_ROOT / "data" / "CAGUI"))
DEFAULT_BASE_MODEL = Path(os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-0.8B"))
DEFAULT_SFT_ADAPTER = Path(os.environ.get("SFT_ADAPTER", REPO_ROOT / "artifacts" / "sft-cagui"))
DEFAULT_OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", REPO_ROOT / "artifacts" / "train" / "grpo-cagui"))

SYSTEM_PROMPT = (
    "You are a GUI assistant. Predict the next GUI action from the screenshot "
    "and the user instruction. Return only one Python-style dict. For click "
    "actions, use exactly {'action': 'click', 'coordinate': [x, y]}. "
    "Coordinates are integers in [0, 1000]. Use "
    "{'action': 'scroll', 'direction': 'up|down|left|right'} for scrolling, "
    "{'action': 'input_text', 'text': '...'} for typing, and "
    "{'action': 'press_back'|'press_home'|'press_enter'|'stop'} for key or stop actions. "
    "Do not include reasoning or extra text. "
)


@dataclass
class GuiSample:
    episode_id: str
    step_id: int
    episode_length: int
    instruction: str
    image_path: str
    width: int
    height: int
    ui_positions: list[list[float]]
    history: list[dict[str, Any]]
    target: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--sft_adapter", default=str(DEFAULT_SFT_ADAPTER), help="Existing SFT/grounding LoRA adapter to continue training. Use empty string to initialize a new LoRA.")
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="domestic")
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_on_error", action="store_true", default=True)
    parser.add_argument("--no_save_on_error", "--no-save-on-error", dest="save_on_error", action="store_false")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--history_window", type=int, default=4)
    parser.add_argument("--max_ui_boxes", type=int, default=30)
    parser.add_argument("--max_image_side", type=int, default=672)
    parser.add_argument("--generation_micro_batch_size", type=int, default=1)
    parser.add_argument("--logprob_micro_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--beta_kl", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.set_defaults(do_sample=True, strict_sft_adapter=True, stop_at_closing_brace=True)
    parser.add_argument("--do_sample", dest="do_sample", action="store_true")
    parser.add_argument("--no_do_sample", "--no-do-sample", dest="do_sample", action="store_false")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--generation_prefix", default="", help="Optional text appended to the prompt during rollout generation and prepended to decoded completions, e.g. \"{'action': \".")
    parser.add_argument("--stop_at_closing_brace", dest="stop_at_closing_brace", action="store_true")
    parser.add_argument("--no_stop_at_closing_brace", "--no-stop-at-closing-brace", dest="stop_at_closing_brace", action="store_false")
    parser.add_argument("--strict_sft_adapter", dest="strict_sft_adapter", action="store_true")
    parser.add_argument("--no_strict_sft_adapter", "--no-strict-sft-adapter", dest="strict_sft_adapter", action="store_false")
    parser.add_argument("--smoke_generate", type=int, default=0, help="Generate this many samples with the loaded adapter, report parse/reward details, then exit.")
    parser.add_argument("--smoke_format_threshold", type=float, default=0.5, help="Minimum parse success rate required by --smoke_generate unless --allow_low_format_rate is set.")
    parser.add_argument("--allow_low_format_rate", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.set_defaults(use_lora=True)
    parser.add_argument("--use_lora", dest="use_lora", action="store_true")
    parser.add_argument("--no_use_lora", "--no-use-lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_list(value: Any, default: list[float]) -> list[float]:
    if value is None:
        return default
    if isinstance(value, list):
        out = value
    elif isinstance(value, str):
        try:
            out = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return default
    else:
        return default
    if not isinstance(out, list) or len(out) != 2:
        return default
    try:
        return [float(out[0]), float(out[1])]
    except Exception:
        return default


def parse_boxes(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if isinstance(value, list):
        boxes = value
    elif isinstance(value, str):
        try:
            boxes = json.loads(value)
        except Exception:
            try:
                boxes = ast.literal_eval(value)
            except Exception:
                return []
    else:
        return []
    out: list[list[float]] = []
    for item in boxes:
        if isinstance(item, list) and len(item) == 4:
            try:
                out.append([float(item[0]), float(item[1]), float(item[2]), float(item[3])])
            except Exception:
                continue
    return out


def normalize_yx_to_xy(yx: list[float]) -> list[int]:
    """Convert CAGUI's normalized [y, x] coordinates to model-facing [x, y]."""
    y, x = yx
    return [
        max(0, min(1000, round(float(x) * 1000))),
        max(0, min(1000, round(float(y) * 1000))),
    ]


def euclidean_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)


def compact_boxes(boxes: list[list[float]], max_boxes: int) -> str:
    trimmed = boxes[:max_boxes] if max_boxes > 0 else boxes
    return json.dumps([[round(v, 4) for v in box] for box in trimmed], ensure_ascii=False)


def target_from_step(step: dict[str, Any]) -> dict[str, Any]:
    action_type = int(step.get("result_action_type", -1))
    text = str(step.get("result_action_text") or "")
    touch_yx = parse_list(step.get("result_touch_yx"), [-1.0, -1.0])
    lift_yx = parse_list(step.get("result_lift_yx"), [-1.0, -1.0])
    duration = step.get("duration", None)
    duration = None if duration in ("", "None") else duration
    target: dict[str, Any] = {
        "action_type": action_type,
        "text": text,
        "touch_yx": touch_yx,
        "lift_yx": lift_yx,
        "duration": duration,
    }

    if action_type == 3:
        target["kind"] = "TYPE"
        return target
    if action_type == 5:
        target["kind"] = "PRESS"
        target["press"] = "BACK"
        return target
    if action_type == 6:
        target["kind"] = "PRESS"
        target["press"] = "HOME"
        return target
    if action_type == 7:
        target["kind"] = "PRESS"
        target["press"] = "ENTER"
        return target
    if action_type == 10:
        target["kind"] = "STATUS"
        target["status"] = "finish"
        return target
    if action_type == 11:
        target["kind"] = "STATUS"
        target["status"] = "impossible"
        return target

    if action_type == 4:
        target["kind"] = "POINT"
        target["point"] = normalize_yx_to_xy(lift_yx if lift_yx != [-1.0, -1.0] else touch_yx)
        dist = euclidean_distance(touch_yx, lift_yx) if all(v >= 0 for v in touch_yx + lift_yx) else 0.0
        if dist > 0.04:
            dy = lift_yx[0] - touch_yx[0]
            dx = lift_yx[1] - touch_yx[1]
            if abs(dy) >= abs(dx):
                target["kind"] = "SWIPE"
                target["to"] = "down" if dy > 0 else "up"
            else:
                target["kind"] = "SWIPE"
                target["to"] = "right" if dx > 0 else "left"
        return target

    if action_type == 0 and duration is not None:
        target["kind"] = "WAIT"
        return target

    target["kind"] = "UNKNOWN"
    return target


def target_to_json(target: dict[str, Any]) -> str:
    kind = target.get("kind", "UNKNOWN")
    if kind == "TYPE":
        return json.dumps({"TYPE": target.get("text", "")}, ensure_ascii=False, separators=(",", ":"))
    if kind == "PRESS":
        return json.dumps({"PRESS": target.get("press", "")}, ensure_ascii=False, separators=(",", ":"))
    if kind == "STATUS":
        return json.dumps({"STATUS": target.get("status", "")}, ensure_ascii=False, separators=(",", ":"))
    if kind == "WAIT":
        duration = int(target.get("duration") or 500)
        return json.dumps({"duration": duration}, ensure_ascii=False, separators=(",", ":"))
    if kind == "POINT":
        point = target.get("point", [0, 0])
        payload: dict[str, Any] = {"POINT": point}
        duration = target.get("duration")
        if duration is not None:
            payload["duration"] = int(duration)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if kind == "SWIPE":
        point = target.get("point", [0, 0])
        payload = {"POINT": point, "to": target.get("to", "down")}
        duration = target.get("duration")
        if duration is not None:
            payload["duration"] = int(duration)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return json.dumps({"STATUS": "continue"}, ensure_ascii=False, separators=(",", ":"))


def target_to_sft_dict(target: dict[str, Any]) -> str:
    kind = target.get("kind", "UNKNOWN")
    if kind == "POINT":
        point = target.get("point", [0, 0])
        return f"{{'action': 'click', 'coordinate': [{int(point[0])}, {int(point[1])}]}}"
    if kind == "SWIPE":
        return f"{{'action': 'scroll', 'direction': '{target.get('to', 'down')}'}}"
    if kind == "TYPE":
        text = str(target.get("text", "")).replace("\\", "\\\\").replace("'", "\\'")
        return f"{{'action': 'input_text', 'text': '{text}'}}"
    if kind == "PRESS":
        return f"{{'action': 'press_{str(target.get('press', '')).lower()}'}}"
    if kind == "STATUS":
        status = str(target.get("status", "finish"))
        return "{'action': 'stop'}" if status == "finish" else "{'action': 'impossible'}"
    if kind == "WAIT":
        return "{'action': 'wait'}"
    return "{'action': 'unknown'}"


def action_kind(target: dict[str, Any]) -> str:
    return str(target.get("kind", "UNKNOWN"))


def format_action_for_history(target: dict[str, Any]) -> str:
    return target_to_sft_dict(target)


class CaguiEpisodeDataset(Dataset):
    def __init__(self, dataset_dir: str, split: str, history_window: int, max_episodes: int = 0):
        self.dataset_dir = Path(dataset_dir)
        self.history_window = history_window
        image_root = self.dataset_dir / "CAGUI_agent"
        episode_root = image_root / split
        episode_files = sorted(episode_root.glob("*/*.json"))
        if max_episodes > 0:
            episode_files = episode_files[:max_episodes]

        self.samples: list[GuiSample] = []
        for ep_path in episode_files:
            try:
                steps = load_json(ep_path)
            except Exception as exc:
                print(f"skip unreadable episode {ep_path}: {exc}")
                continue
            if not isinstance(steps, list):
                continue
            previous: list[dict[str, Any]] = []
            for step in sorted(steps, key=lambda item: int(item.get("step_id", 0))):
                image_path = image_root / str(step.get("image_path", ""))
                if not image_path.exists():
                    continue
                target = target_from_step(step)
                history = previous[-history_window:] if history_window > 0 else []
                ui_positions = parse_boxes(step.get("ui_positions"))
                self.samples.append(
                    GuiSample(
                        episode_id=str(step.get("episode_id", "")),
                        step_id=int(step.get("step_id", len(previous))),
                        episode_length=int(step.get("episode_length", 0) or 0),
                        instruction=str(step.get("instruction", "")),
                        image_path=str(image_path),
                        width=int(step.get("image_width", 0) or 0),
                        height=int(step.get("image_height", 0) or 0),
                        ui_positions=ui_positions,
                        history=list(history),
                        target=target,
                    )
                )
                previous.append(target)

        if not self.samples:
            raise RuntimeError(f"No CAGUI samples found under {episode_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> GuiSample:
        return self.samples[idx]


def collate(samples: list[GuiSample]) -> list[GuiSample]:
    return samples


def parse_action_text(text: str) -> tuple[dict[str, Any] | None, bool]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None, False
    raw = match.group(0)
    try:
        obj = json.loads(raw)
    except Exception:
        try:
            obj = ast.literal_eval(raw)
        except Exception:
            return None, False
    return obj if isinstance(obj, dict) else None, isinstance(obj, dict)


def parse_failure_reward(text: str) -> tuple[float, dict[str, Any]]:
    stripped = text.strip()
    reward = -1.0
    reason = "parse_failed"
    if stripped.startswith("{"):
        reward = -0.8
        reason = "parse_failed_prefix_dict"
    if "action" in stripped:
        reward = max(reward, -0.6)
        reason = "parse_failed_has_action"
    if "{" in stripped and "}" in stripped:
        reward = max(reward, -0.4)
        reason = "parse_failed_malformed_dict"
    return reward, {
        "reason": reason,
        "parsed": None,
        "format_reward": 0.0,
        "parse_shaping_reward": reward,
    }


def truncate_at_balanced_dict(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return text.strip()
    depth = 0
    quote: str | None = None
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if quote is not None:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1].strip()
    return text.strip()


def repair_completion_for_debug(text: str) -> str | None:
    stripped = text.strip()
    if not stripped.startswith("{") or "}" in stripped:
        return None
    if stripped in {"{", "{'", '{"'}:
        return None
    return stripped + "}"


def normalize_pred_action(obj: dict[str, Any]) -> dict[str, Any]:
    if "action" not in obj:
        return obj
    action = str(obj.get("action", "")).lower().strip()
    out: dict[str, Any] = {}
    if action in {"click", "tap"}:
        point = obj.get("coordinate") or obj.get("point") or obj.get("POINT")
        out["POINT"] = point
    elif action in {"long_press", "long press"}:
        point = obj.get("coordinate") or obj.get("point") or obj.get("POINT")
        out["POINT"] = point
        out["duration"] = obj.get("duration", 1000)
    elif action in {"scroll", "swipe"}:
        point = obj.get("coordinate") or obj.get("start") or obj.get("POINT") or [500, 500]
        out["POINT"] = point
        out["to"] = obj.get("direction") or obj.get("to")
    elif action in {"input_text", "type", "input"}:
        out["TYPE"] = obj.get("text", "")
    elif action in {"stop", "finish", "done", "complete"}:
        out["STATUS"] = "finish"
    elif action in {"impossible"}:
        out["STATUS"] = "impossible"
    elif action in {"wait", "no_action"}:
        out["duration"] = obj.get("duration", 500)
    elif action in {"press_back", "back"}:
        out["PRESS"] = "BACK"
    elif action in {"press_home", "home"}:
        out["PRESS"] = "HOME"
    elif action in {"press_enter", "enter"}:
        out["PRESS"] = "ENTER"
    else:
        out["action"] = action
    return out


def point_reward(pred_point: list[float] | None, gold_point: list[float] | None, boxes: list[list[float]]) -> float:
    if not gold_point:
        return 1.0 if not pred_point else 0.0
    if not pred_point:
        return 0.0
    px, py = float(pred_point[0]) / 1000.0, float(pred_point[1]) / 1000.0
    gx, gy = float(gold_point[0]) / 1000.0, float(gold_point[1]) / 1000.0
    dist = euclidean_distance([px, py], [gx, gy])
    distance_reward = max(0.0, 1.0 - dist / 0.25)
    if boxes:
        target_box_reward = 0.0
        for box in boxes:
            y, x, h, w = box
            gold_inside = x <= gx <= x + w and y <= gy <= y + h
            if not gold_inside:
                continue
            pred_inside = x <= px <= x + w and y <= py <= y + h
            if pred_inside:
                cx = x + w / 2.0
                cy = y + h / 2.0
                center_dist = euclidean_distance([px, py], [cx, cy])
                target_box_reward = max(target_box_reward, 1.0 - min(1.0, center_dist / max(0.12, max(w, h))))
            else:
                dx = 0.0
                if px < x:
                    dx = x - px
                elif px > x + w:
                    dx = px - (x + w)
                dy = 0.0
                if py < y:
                    dy = y - py
                elif py > y + h:
                    dy = py - (y + h)
                box_dist = dx + dy
                target_box_reward = max(target_box_reward, max(0.0, 1.0 - box_dist / 0.25))
        if target_box_reward > 0:
            return max(distance_reward, target_box_reward)
    return distance_reward


def direction_from_points(start: list[float], end: list[float]) -> str:
    dy = end[0] - start[0]
    dx = end[1] - start[1]
    if abs(dy) >= abs(dx):
        return "down" if dy > 0 else "up"
    return "right" if dx > 0 else "left"

#奖励函数
def reward_completion_detail(completion: str, target: dict[str, Any], boxes: list[list[float]]) -> tuple[float, dict[str, Any]]:
    obj, ok = parse_action_text(completion)
    if not ok or obj is None:
        reward, detail = parse_failure_reward(completion)
        detail["target_kind"] = action_kind(target)
        debug_repair = repair_completion_for_debug(completion)
        if debug_repair is not None:
            repaired, repaired_ok = parse_action_text(debug_repair)
            detail["debug_repair"] = debug_repair
            detail["debug_repair_ok"] = repaired_ok and repaired is not None
        return reward, detail
    obj = normalize_pred_action(obj)

    kind = action_kind(target)
    reward = 0.0
    format_reward = 1.0 #只要能被解析成 dict，就认为格式分满分。
    action_reward = 0.0
    args_reward = 0.0
#   reward += 0.2 * format_reward + 0.3 * action_reward + 0.5 * args_reward
    #动作规范化
    """
     例如：
    {'action': 'click', 'coordinate': [x, y]}
    会被转成：
    {"POINT": [x, y]}
    {'action': 'input_text', 'text': 'abc'} 会变成：
    {"TYPE": "abc"}
    """
    if kind == "TYPE":
        pred = str(obj.get("TYPE", ""))
        gold = str(target.get("text", ""))
        action_reward = 1.0 if "TYPE" in obj else -0.3
        args_reward = 1.0 if pred == gold else difflib.SequenceMatcher(None, pred, gold).ratio()
    elif kind == "PRESS":
        pred = str(obj.get("PRESS", ""))
        gold = str(target.get("press", ""))
        action_reward = 1.0 if "PRESS" in obj else -0.3
        args_reward = 1.0 if pred == gold else 0.0
    elif kind == "STATUS":
        pred = str(obj.get("STATUS", ""))
        gold = str(target.get("status", ""))
        action_reward = 1.0 if "STATUS" in obj else -0.3
        args_reward = 1.0 if pred == gold else 0.0
    elif kind == "WAIT":
        duration = obj.get("duration")
        action_reward = 1.0 if "duration" in obj and "POINT" not in obj and "TYPE" not in obj else -0.3
        args_reward = 1.0 if isinstance(duration, (int, float)) and 150 <= float(duration) <= 5000 else 0.0
    elif kind in {"POINT", "SWIPE"}:
        point = obj.get("POINT")
        if isinstance(point, list) and len(point) == 2:
            point = [float(point[0]), float(point[1])]
        else:
            point = None
        action_reward = 1.0 if "POINT" in obj else -0.3
        args_reward = point_reward(point, target.get("point"), boxes)
        if kind == "SWIPE":
            pred_to = obj.get("to")
            gold_to = target.get("to")
            if isinstance(gold_to, str):
                args_reward = 0.5 * args_reward + 0.5 * (1.0 if pred_to == gold_to else 0.0)
            duration = obj.get("duration")
            if "duration" in obj:
                args_reward = 0.5 * args_reward + 0.5 * (1.0 if isinstance(duration, (int, float)) and 150 <= float(duration) <= 5000 else 0.0)
        if "duration" in obj and kind == "POINT":
            dur = obj.get("duration")
            args_reward = 0.7 * args_reward + 0.3 * (1.0 if isinstance(dur, (int, float)) and 150 <= float(dur) <= 5000 else 0.0)
    else:
        action_reward = -0.5
        args_reward = 0.0

    if kind != "UNKNOWN" and action_reward > 0:
        reward += 0.2 * format_reward + 0.3 * action_reward + 0.5 * args_reward
    else:
        return -1.0, {
            "reason": "invalid_action_or_kind",
            "parsed": obj,
            "target_kind": kind,
            "action_reward": action_reward,
            "args_reward": args_reward,
            "format_reward": format_reward,
        }

    if isinstance(obj, dict) and len(obj) == 1 and "STATUS" in obj:
        reward += 0.05
    return reward, {
        "reason": "ok",
        "parsed": obj,
        "target_kind": kind,
        "action_reward": action_reward,
        "args_reward": args_reward,
        "format_reward": format_reward,
    }


def reward_completion(completion: str, target: dict[str, Any], boxes: list[list[float]]) -> float:
    reward, _ = reward_completion_detail(completion, target, boxes)
    return reward


def load_resized_image(path: str, max_image_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max_image_side and max(image.size) > max_image_side:
        image.thumbnail((max_image_side, max_image_side), Image.Resampling.BICUBIC)
    return image


def prepare_batch(
    processor: Any,
    samples: list[GuiSample],
    prompts_per_sample: int,
    max_ui_boxes: int,
    max_image_side: int,
) -> tuple[list[str], list[Image.Image], list[dict[str, Any]]]:
    prompts: list[str] = []
    images: list[Image.Image] = []
    targets: list[dict[str, Any]] = []
    for sample in samples:
        prompt = build_prompt(processor, sample, max_ui_boxes)
        image = load_resized_image(sample.image_path, max_image_side)
        for _ in range(prompts_per_sample):
            prompts.append(prompt)
            images.append(image.copy())
            targets.append(sample.target)
    return prompts, images, targets


def build_user_text(sample: GuiSample, max_ui_boxes: int) -> str:
    history_lines = []
    for idx, item in enumerate(sample.history):
        history_lines.append(f"step {idx}: {format_action_for_history(item)}")
    history = "\n".join(history_lines) if history_lines else "无"
    return (
        f"In this UI screenshot, I want to perform the command: {sample.instruction}\n"
        "Return only one Python-style dict. For click actions, use exactly:\n"
        "{'action': 'click', 'coordinate': [x, y]}\n"
        "where x and y are integers in [0, 1000] normalized to the image\n"
        "(x = 0 left edge, x = 1000 right edge; y = 0 top, y = 1000 bottom).\n"
        "For other actions, use exactly:\n"
        "{'action': '<action_name>'}\n"
        "Do not include reasoning or extra text.\n"
        f"Current step: {sample.step_id}/{sample.episode_length or '?'}\n"
        f"Image size: {sample.width}x{sample.height}\n"
        f"Candidate UI boxes [y,x,h,w]: {compact_boxes(sample.ui_positions, max_ui_boxes)}\n"
        f"Previous actions:\n{history}"
    )


def build_prompt(processor: Any, sample: GuiSample, max_ui_boxes: int) -> str:
    user_text = build_user_text(sample, max_ui_boxes)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def response_logprobs_batch(model: Any, processor: Any, prompts: list[str], images: list[Image.Image], completions: list[str]) -> torch.Tensor:
    full_texts = [p + c for p, c in zip(prompts, completions)]
    prompt_inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    full_inputs = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    prompt_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in prompt_inputs.items()}
    full_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in full_inputs.items()}
    labels = full_inputs["input_ids"].clone()
    attention = full_inputs["attention_mask"]
    for i in range(labels.size(0)):
        prompt_len = int(prompt_inputs["attention_mask"][i].sum().item())
        nonpad = torch.nonzero(attention[i], as_tuple=False).flatten()
        labels[i, nonpad[:prompt_len]] = -100
        labels[i, attention[i] == 0] = -100
    outputs = model(**full_inputs)
    logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = torch.gather(F.log_softmax(logits, dim=-1), -1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)


def response_logprobs(
    model: Any,
    processor: Any,
    prompts: list[str],
    images: list[Image.Image],
    completions: list[str],
    micro_batch_size: int,
) -> torch.Tensor:
    chunks = []
    micro_batch_size = max(1, micro_batch_size)
    for start in range(0, len(prompts), micro_batch_size):
        end = start + micro_batch_size
        chunks.append(response_logprobs_batch(model, processor, prompts[start:end], images[start:end], completions[start:end]))
    return torch.cat(chunks, dim=0)


def generate_completion(
    model: Any,
    processor: Any,
    prompt: str,
    image: Image.Image,
    device: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    repetition_penalty: float,
    stop_at_closing_brace: bool,
    generation_prefix: str = "",
) -> str:
    generation_prompt = prompt + generation_prefix
    inputs = processor(text=[generation_prompt], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    generation_kwargs: dict[str, Any] = {
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    generated = model.generate(**inputs, **generation_kwargs)
    row = generated[0]
    completion_ids = row[input_len:]
    completion = processor.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    completion = generation_prefix + completion
    if stop_at_closing_brace:
        completion = truncate_at_balanced_dict(completion)
    return completion


def validate_sft_adapter(path: str, strict: bool) -> Path | None:
    if not path:
        if strict:
            raise ValueError("--sft_adapter is required when --strict_sft_adapter is enabled")
        return None
    adapter = Path(path)
    config = adapter / "adapter_config.json"
    has_weights = (adapter / "adapter_model.safetensors").exists() or (adapter / "adapter_model.bin").exists()
    if not config.exists() or not has_weights:
        message = (
            f"SFT adapter is incomplete: {adapter}. Expected adapter_config.json and "
            "adapter_model.safetensors or adapter_model.bin. Set SFT_ADAPTER to the SFT/bridge adapter, "
            "or pass --no_strict_sft_adapter only for intentional fresh LoRA debugging."
        )
        if strict:
            raise FileNotFoundError(message)
        print(message, flush=True)
        return None
    print(f"validated SFT adapter: {adapter}", flush=True)
    try:
        cfg = load_json(config)
        print(
            json.dumps(
                {
                    "adapter_r": cfg.get("r"),
                    "adapter_lora_alpha": cfg.get("lora_alpha"),
                    "adapter_target_modules": cfg.get("target_modules"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    except Exception as exc:
        print(f"warning: could not read adapter config {config}: {exc}", flush=True)
    return adapter


def run_smoke_generate(model: Any, processor: Any, dataset: CaguiEpisodeDataset, args: argparse.Namespace, device: Any) -> float:
    model.eval()
    total = min(max(0, args.smoke_generate), len(dataset))
    if total == 0:
        return 1.0
    ok_count = 0
    rewards: list[float] = []
    with torch.no_grad():
        for idx in range(total):
            sample = dataset[idx]
            prompt = build_prompt(processor, sample, args.max_ui_boxes)
            image = load_resized_image(sample.image_path, args.max_image_side)
            completion = generate_completion(
                model,
                processor,
                prompt,
                image,
                device,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                args.do_sample,
                args.repetition_penalty,
                args.stop_at_closing_brace,
                args.generation_prefix,
            )
            parsed, ok = parse_action_text(completion)
            reward, detail = reward_completion_detail(completion, sample.target, sample.ui_positions)
            ok_count += int(ok and parsed is not None)
            rewards.append(reward)
            print(
                json.dumps(
                    {
                        "smoke_idx": idx,
                        "target": target_to_sft_dict(sample.target),
                        "completion": completion,
                        "parsed": parsed,
                        "reward": reward,
                        "detail": detail,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    rate = ok_count / total
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(
        json.dumps(
            {
                "smoke_total": total,
                "format_success_rate": round(rate, 4),
                "reward_mean": round(mean_reward, 4),
                "reward_max": round(max(rewards), 4) if rewards else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    model.train()
    if rate < args.smoke_format_threshold and not args.allow_low_format_rate:
        raise RuntimeError(
            f"Smoke generation format success rate {rate:.3f} is below threshold "
            f"{args.smoke_format_threshold:.3f}. Run a CAGUI SFT bridge or pass --allow_low_format_rate for debugging."
        )
    return rate


def save_checkpoint(model: Any, processor: Any, output_dir: str, step: int) -> None:
    ckpt = Path(output_dir) / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt))
    processor.save_pretrained(str(ckpt))
    print(f"saved {ckpt}", flush=True)


def main() -> None:
    args = parse_args()
    if args.num_generations < 2:
        raise ValueError("--num_generations must be >= 2 for GRPO")
    random.seed(args.seed)
    if torch is not None:
        torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = CaguiEpisodeDataset(args.dataset_dir, args.split, args.history_window, args.max_episodes)
    print(f"loaded {len(dataset)} CAGUI steps", flush=True)
    if args.dry_run:
        sample = dataset[0]
        print(
            json.dumps(
                {
                    "episode_id": sample.episode_id,
                    "step_id": sample.step_id,
                    "instruction": sample.instruction,
                    "target": target_to_sft_dict(sample.target),
                    "compact_json_target": target_to_json(sample.target),
                    "reward_self": reward_completion(target_to_sft_dict(sample.target), sample.target, sample.ui_positions),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return

    if torch is None or DataLoader is None or F is None:
        raise RuntimeError("torch is required for training. Activate the training environment before running without --dry_run.")
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
        get_cosine_schedule_with_warmup,
    )
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except Exception as exc:
        raise RuntimeError("peft is required for LoRA training. Activate/install the training environment.") from exc

    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype if dtype in (torch.float16, torch.bfloat16) else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quantization_config,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.use_lora:
        if LoraConfig is None or get_peft_model is None:
            raise RuntimeError("peft is required for --use_lora")
        sft_adapter = validate_sft_adapter(args.sft_adapter, args.strict_sft_adapter)
        if sft_adapter is not None:
            model = PeftModel.from_pretrained(model, str(sft_adapter), is_trainable=True)
            print(f"loaded trainable SFT adapter from {sft_adapter}", flush=True)
        else:
            print("initializing a new LoRA adapter because --no_strict_sft_adapter was set", flush=True)
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
        ref_context = model.disable_adapter
    else:
        ref_model = AutoModelForImageTextToText.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
            quantization_config=quantization_config,
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

        @contextlib.contextmanager
        def ref_context():
            yield ref_model

    if args.smoke_generate > 0:
        run_smoke_generate(model, processor, dataset, args, next(model.parameters()).device)
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup_steps = max(1, int(args.max_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, args.max_steps)

    model.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    print(
        f"start training: max_steps={args.max_steps}, batch_size={args.batch_size}, "
        f"num_generations={args.num_generations}, max_new_tokens={args.max_new_tokens}, "
        f"load_in_4bit={args.load_in_4bit}",
        flush=True,
    )
    try:
        while global_step < args.max_steps:
            for samples in loader:
                prompts, images, targets = prepare_batch(
                    processor,
                    samples,
                    args.num_generations,
                    args.max_ui_boxes,
                    args.max_image_side,
                )
                device = next(model.parameters()).device
                with torch.no_grad():
                    was_training = model.training
                    old_use_cache = getattr(model.config, "use_cache", None)
                    model.eval()
                    if old_use_cache is not None:
                        model.config.use_cache = True
                    completions = []
                    for prompt, image in zip(prompts, images):
                        completions.append(
                            generate_completion(
                                model,
                                processor,
                                prompt,
                                image,
                                device,
                                args.max_new_tokens,
                                args.temperature,
                                args.top_p,
                                args.do_sample,
                                args.repetition_penalty,
                                args.stop_at_closing_brace,
                                args.generation_prefix,
                            )
                        )
                    if old_use_cache is not None:
                        model.config.use_cache = old_use_cache
                    if was_training:
                        model.train()

                    if len(completions) != len(prompts):
                        raise RuntimeError(f"generated {len(completions)} completions for {len(prompts)} prompts")
                    for i, completion in enumerate(completions[: min(len(completions), 3)]):
                        print(f"debug_step={global_step} sample_idx={i} completion={completion!r}", flush=True)

                    old_logps = response_logprobs(model, processor, prompts, images, completions, args.logprob_micro_batch_size).detach()
                    if args.use_lora:
                        with ref_context():
                            ref_logps = response_logprobs(model, processor, prompts, images, completions, args.logprob_micro_batch_size).detach()
                    else:
                        with ref_context() as ref_model:
                            ref_logps = response_logprobs(ref_model, processor, prompts, images, completions, args.logprob_micro_batch_size).detach()

                reward_values = []
                reward_details: list[dict[str, Any]] = []
                for idx, (completion, target) in enumerate(zip(completions, targets)):
                    sample = samples[idx // args.num_generations]
                    reward, detail = reward_completion_detail(completion, target, sample.ui_positions)
                    reward_values.append(reward)
                    reward_details.append(detail)
                rewards = torch.tensor(reward_values, device=old_logps.device)
                expected_rollouts = len(samples) * args.num_generations
                if len(completions) != expected_rollouts or len(targets) != expected_rollouts:
                    raise RuntimeError(
                        f"rollout count mismatch: prompts={len(prompts)}, completions={len(completions)}, "
                        f"targets={len(targets)}, expected={expected_rollouts}"
                    )
                if len(rewards) != expected_rollouts:
                    raise RuntimeError(
                        f"reward/completion count mismatch: rewards={len(rewards)}, expected={expected_rollouts}, "
                        f"samples={len(samples)}, num_generations={args.num_generations}, completions={len(completions)}"
                    )
                grouped_rewards = rewards.view(len(samples), args.num_generations)
                mean = grouped_rewards.mean(dim=1, keepdim=True)
                std = grouped_rewards.std(dim=1, keepdim=True).clamp_min(1e-4)
                advantages = ((grouped_rewards - mean) / std).view(-1).detach()

                total_rollouts = len(prompts)
                micro_bs = max(1, args.logprob_micro_batch_size)
                policy_loss_sum = 0.0
                kl_sum = 0.0
                for start in range(0, total_rollouts, micro_bs):
                    end = min(start + micro_bs, total_rollouts)
                    mb_weight = (end - start) / total_rollouts
                    logps = response_logprobs_batch(model, processor, prompts[start:end], images[start:end], completions[start:end])
                    ratio = torch.exp(logps - old_logps[start:end])
                    clipped = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
                    policy_loss = -torch.minimum(ratio * advantages[start:end], clipped * advantages[start:end]).mean()
                    kl = (logps - ref_logps[start:end]).mean()
                    loss = mb_weight * (policy_loss + args.beta_kl * kl) / args.gradient_accumulation_steps
                    loss.backward()
                    policy_loss_sum += float(policy_loss.detach().item()) * (end - start)
                    kl_sum += float(kl.detach().item()) * (end - start)

                if (global_step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                global_step += 1
                if global_step % args.logging_steps == 0:
                    best = int(torch.argmax(grouped_rewards[0]).item())
                    parse_successes = sum(1 for detail in reward_details if detail.get("parsed") is not None)
                    parse_failures = len(reward_details) - parse_successes
                    action_matches = sum(1 for detail in reward_details if detail.get("action_reward", 0.0) > 0)
                    first_bad = next(({
                        "completion": completions[i],
                        "reward": reward_values[i],
                        **reward_details[i],
                    } for i in range(len(reward_values)) if reward_values[i] <= -0.99), None)
                    print(
                        json.dumps(
                            {
                                "step": global_step,
                                "reward_mean": round(float(rewards.mean().item()), 4),
                                "reward_max": round(float(rewards.max().item()), 4),
                                "format_success_rate": round(parse_successes / max(1, len(reward_details)), 4),
                                "parse_fail_rate": round(parse_failures / max(1, len(reward_details)), 4),
                                "action_match_rate": round(action_matches / max(1, len(reward_details)), 4),
                                "kl": round(float(kl_sum / total_rollouts), 6),
                                "target": target_to_sft_dict(targets[0]),
                                "best_completion": completions[best],
                                "first_bad": first_bad,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                if global_step % args.save_steps == 0:
                    save_checkpoint(model, processor, args.output_dir, global_step)
                if global_step >= args.max_steps:
                    break
    except Exception:
        if args.save_on_error and global_step > 0:
            save_checkpoint(model, processor, args.output_dir, global_step)
            print(f"saved emergency checkpoint at step {global_step}", flush=True)
        raise

    save_checkpoint(model, processor, args.output_dir, global_step)
    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"done. final adapter/model saved to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
