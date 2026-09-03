import json

from PIL import Image

from gui_agent.training.grpo import CaguiEpisodeDataset


def test_loads_episode_and_normalizes_click(tmp_path) -> None:
    episode_dir = tmp_path / "CAGUI_agent" / "domestic" / "episode-1"
    episode_dir.mkdir(parents=True)
    image_path = episode_dir / "screen.png"
    Image.new("RGB", (100, 200), color="white").save(image_path)
    steps = [
        {
            "episode_id": "episode-1",
            "step_id": 0,
            "episode_length": 1,
            "instruction": "点击按钮",
            "image_path": "domestic/episode-1/screen.png",
            "image_width": 100,
            "image_height": 200,
            "result_action_type": 4,
            "result_touch_yx": [0.5, 0.25],
            "result_lift_yx": [0.5, 0.25],
            "ui_positions": [[0.45, 0.2, 0.1, 0.1]],
        }
    ]
    (episode_dir / "episode.json").write_text(json.dumps(steps, ensure_ascii=False), encoding="utf-8")

    dataset = CaguiEpisodeDataset(str(tmp_path), "domestic", history_window=4)

    assert len(dataset) == 1
    assert dataset[0].target["kind"] == "POINT"
    assert dataset[0].target["point"] == [250, 500]
