from gui_agent.evaluation.merge_screenspot import merge_shards


def _record(index: int) -> dict:
    return {
        "index": index,
        "data_type": "icon",
        "data_source": "test",
        "raw_strict_format": True,
        "parse_success": True,
        "eos_terminated": True,
        "extra_content": False,
        "action_correct": True,
        "coordinate_valid": True,
        "bbox_hit": index % 2 == 0,
        "center_distance_px": float(index),
    }


def test_merge_round_robin_shards_restores_source_order() -> None:
    common = {
        "num_shards": 2,
        "dataset_samples": 4,
        "base_model": "base",
        "adapter_path": "adapter",
        "data_json": "data.json",
        "data_json_sha256": "hash",
        "image_dir": "images",
    }
    payloads = [
        {"metadata": dict(common, shard_index=0), "elapsed_seconds": 2, "records": [_record(0), _record(2)]},
        {"metadata": dict(common, shard_index=1), "elapsed_seconds": 3, "records": [_record(1), _record(3)]},
    ]
    merged = merge_shards(payloads, "merged")
    assert [record["index"] for record in merged["records"]] == [0, 1, 2, 3]
    assert merged["metrics"]["grounding_accuracy"] == 0.5
    assert merged["elapsed_seconds_wall_approx"] == 3
