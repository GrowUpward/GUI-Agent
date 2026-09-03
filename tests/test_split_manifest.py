import json
from types import SimpleNamespace

import pytest

from gui_agent.training.sft import split_from_manifest


class FakeDataset:
    def __init__(self, episode_ids):
        self.samples = [SimpleNamespace(episode_id=episode_id) for episode_id in episode_ids]


def write_manifest(path, *, test_ids=None):
    test_ids = test_ids or ["episode-test"]
    manifest = {
        "manifest_version": 1,
        "dataset_name": "CAGUI",
        "dataset_split": "domestic",
        "coordinate_convention": "model_and_api_use_normalized_xy_0_1000; source_cagui_uses_yx",
        "full": {"episode_count": 3, "sample_count": 3},
        "train": {"episode_count": 1, "sample_count": 1, "episode_ids": ["episode-train"]},
        "validation": {"episode_count": 1, "sample_count": 1, "episode_ids": ["episode-val"]},
        "test": {"episode_count": len(test_ids), "sample_count": len(test_ids), "episode_ids": test_ids},
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_split_from_manifest_uses_fixed_episode_membership(tmp_path):
    manifest_path = tmp_path / "split.json"
    write_manifest(manifest_path)
    dataset = FakeDataset(["episode-val", "episode-test", "episode-train"])

    train, validation, test, _ = split_from_manifest(dataset, manifest_path, "domestic")

    assert [sample.episode_id for sample in train.samples] == ["episode-train"]
    assert [sample.episode_id for sample in validation.samples] == ["episode-val"]
    assert [sample.episode_id for sample in test.samples] == ["episode-test"]


def test_split_from_manifest_rejects_dataset_drift(tmp_path):
    manifest_path = tmp_path / "split.json"
    write_manifest(manifest_path)
    dataset = FakeDataset(["episode-val", "episode-new", "episode-train"])

    with pytest.raises(ValueError, match="Dataset does not match split manifest"):
        split_from_manifest(dataset, manifest_path, "domestic")
