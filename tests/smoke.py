"""Dependency-light smoke tests runnable without pytest."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from PIL import Image

from gui_agent.training.grpo import (
    CaguiEpisodeDataset,
    normalize_pred_action,
    parse_action_text,
    reward_completion,
    truncate_at_balanced_dict,
)


def main() -> None:
    parsed, ok = parse_action_text("{'action': 'click', 'coordinate': [500, 400]}")
    assert ok and parsed == {"action": "click", "coordinate": [500, 400]}
    assert normalize_pred_action({"action": "input_text", "text": "北京"}) == {"TYPE": "北京"}
    assert reward_completion(
        "{'action': 'click', 'coordinate': [500, 500]}",
        {"kind": "POINT", "point": [500, 500]},
        [[0.4, 0.4, 0.2, 0.2]],
    ) == 1.0
    assert truncate_at_balanced_dict("prefix {'action': 'stop'} trailing") == "{'action': 'stop'}"

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        episode_dir = root / "CAGUI_agent" / "domestic" / "episode-1"
        episode_dir.mkdir(parents=True)
        Image.new("RGB", (100, 200), color="white").save(episode_dir / "screen.png")
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
        dataset = CaguiEpisodeDataset(str(root), "domestic", history_window=4)
        assert len(dataset) == 1
        assert dataset[0].target["kind"] == "POINT"
        assert dataset[0].target["point"] == [250, 500]

    print("smoke tests: PASS")


if __name__ == "__main__":
    main()
