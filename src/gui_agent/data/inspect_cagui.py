#!/usr/bin/env python
"""Count result_action_type values in CAGUI Agent episode JSON files."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAGUI_ROOT = Path(os.environ.get("CAGUI_ROOT", REPO_ROOT / "data" / "CAGUI"))

ACTION_TYPE_NAMES = {
    0: "LONG_POINT",
    1: "NO_ACTION",
    2: "UNUSED_2",
    3: "TYPE",
    4: "DUAL_POINT",
    5: "PRESS_BACK",
    6: "PRESS_HOME",
    7: "PRESS_ENTER",
    8: "UNUSED_8",
    9: "UNUSED_9",
    10: "STATUS_TASK_COMPLETE",
    11: "STATUS_TASK_IMPOSSIBLE",
}


def parse_action_type(value: Any) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def count_episode_file(json_path: Path) -> Counter[int | str]:
    steps = json.loads(json_path.read_text(encoding="utf-8"))
    counts: Counter[int | str] = Counter()
    if not isinstance(steps, list):
        return counts
    for step in steps:
        if not isinstance(step, dict):
            continue
        counts[parse_action_type(step.get("result_action_type", "MISSING"))] += 1
    return counts


def sort_key(item: tuple[int | str, int]) -> tuple[int, int | str]:
    action_type, _ = item
    if isinstance(action_type, int):
        return (0, action_type)
    return (1, action_type)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cagui_root", type=Path, default=DEFAULT_CAGUI_ROOT)
    parser.add_argument("--split", default="domestic")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print counts as JSON.")
    args = parser.parse_args()

    episode_root = args.cagui_root / "CAGUI_agent" / args.split
    episode_files = sorted(episode_root.glob("*/*.json"))
    if args.max_episodes is not None:
        episode_files = episode_files[: args.max_episodes]

    counts: Counter[int | str] = Counter()
    unreadable = 0
    for episode_file in episode_files:
        try:
            counts.update(count_episode_file(episode_file))
        except Exception:
            unreadable += 1

    total_steps = sum(counts.values())
    if args.json:
        output = {
            "cagui_root": str(args.cagui_root),
            "split": args.split,
            "episodes": len(episode_files),
            "unreadable_episodes": unreadable,
            "total_steps": total_steps,
            "counts": [
                {
                    "result_action_type": action_type,
                    "name": ACTION_TYPE_NAMES.get(action_type, "UNKNOWN")
                    if isinstance(action_type, int)
                    else "NON_INTEGER",
                    "count": count,
                }
                for action_type, count in sorted(counts.items(), key=sort_key)
            ],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    print(f"CAGUI root: {args.cagui_root}")
    print(f"Split: {args.split}")
    print(f"Episodes: {len(episode_files)}")
    print(f"Unreadable episodes: {unreadable}")
    print(f"Total steps: {total_steps}")
    print()
    print(f"{'result_action_type':>18}  {'name':<28}  {'count':>8}  {'percent':>8}")
    print("-" * 70)
    for action_type, count in sorted(counts.items(), key=sort_key):
        name = ACTION_TYPE_NAMES.get(action_type, "UNKNOWN") if isinstance(action_type, int) else "NON_INTEGER"
        percent = count / total_steps * 100 if total_steps else 0.0
        print(f"{str(action_type):>18}  {name:<28}  {count:>8}  {percent:>7.2f}%")


if __name__ == "__main__":
    main()
