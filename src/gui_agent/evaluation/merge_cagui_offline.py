#!/usr/bin/env python3
"""Merge CAGUI offline-evaluation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gui_agent.evaluation.cagui_offline import summarize_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--label", required=True)
    return parser.parse_args()


def merge_shards(payloads: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if not payloads:
        raise ValueError("No shard payloads supplied")
    by_index = {int(payload["metadata"]["shard_index"]): payload for payload in payloads}
    num_shards = int(payloads[0]["metadata"]["num_shards"])
    if set(by_index) != set(range(num_shards)):
        raise ValueError(f"Expected shard indexes 0..{num_shards - 1}; got {sorted(by_index)}")
    dataset_samples = int(payloads[0]["metadata"]["dataset_samples"])
    records = sorted(
        [record for payload in payloads for record in payload["records"]],
        key=lambda record: int(record["source_index"]),
    )
    if len(records) != dataset_samples or [record["source_index"] for record in records] != list(range(dataset_samples)):
        raise ValueError("Merged shards do not cover every source index exactly once")
    metadata = dict(payloads[0]["metadata"])
    metadata.update({"label": label, "evaluated_samples": len(records), "shard_index": None})
    return {
        "metadata": metadata,
        "elapsed_seconds_wall_approx": max(float(payload["elapsed_seconds"]) for payload in payloads),
        "elapsed_seconds_process_sum": sum(float(payload["elapsed_seconds"]) for payload in payloads),
        "metrics": summarize_records(records),
        "records": records,
    }


def main() -> None:
    args = parse_args()
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    result = merge_shards(payloads, args.label)
    output_file = Path(args.output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_file": str(output_file), "metrics": result["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
