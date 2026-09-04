import json

from PIL import Image

from gui_agent.data.ui_r1_warmup import UiR1WarmupDataset


def test_ui_r1_warmup_keeps_only_lossless_actions(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("click.png", "back.png", "scroll.png"):
        Image.new("RGB", (200, 100), color="white").save(image_dir / name)
    records = [
        {
            "img_filename": "click.png",
            "bbox": [20, 10, 60, 30],
            "instruction": "tap target",
            "action": "click",
            "data_source": "ios",
        },
        {
            "img_filename": "back.png",
            "instruction": "go back",
            "action": "navigate_back",
            "data_source": "android",
        },
        {
            "img_filename": "scroll.png",
            "instruction": "scroll",
            "action": "scroll",
            "data_source": "android",
        },
    ]
    data_file = tmp_path / "train_ground.json"
    data_file.write_text(json.dumps(records), encoding="utf-8")

    dataset = UiR1WarmupDataset(data_file, image_dir)

    assert len(dataset) == 2
    assert dataset[0].target["kind"] == "POINT"
    assert dataset[0].target["point"] == [200, 200]
    assert dataset[0].ui_positions == [[0.1, 0.1, 0.2, 0.2]]
    assert dataset[1].target["kind"] == "PRESS"
    assert dataset[1].target["press"] == "BACK"
    assert dataset.summary["raw_samples"] == 3
    assert dataset.summary["usable_samples"] == 2
    assert dataset.summary["skipped_counts"] == {"unsupported_scroll": 1}


def test_ui_r1_warmup_rejects_missing_click_bbox(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (100, 100), color="white").save(image_dir / "bad.png")
    Image.new("RGB", (100, 100), color="white").save(image_dir / "back.png")
    records = [
        {"img_filename": "bad.png", "instruction": "tap", "action": "click"},
        {"img_filename": "back.png", "instruction": "back", "action": "navigate_back"},
    ]
    data_file = tmp_path / "train_ground.json"
    data_file.write_text(json.dumps(records), encoding="utf-8")

    dataset = UiR1WarmupDataset(data_file, image_dir)

    assert len(dataset) == 1
    assert dataset.summary["skipped_counts"] == {"click_missing_bbox": 1}
