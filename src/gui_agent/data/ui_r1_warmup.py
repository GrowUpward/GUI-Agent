"""Load the released UI-R1-low records as clean Stage-0 SFT samples."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from gui_agent.training.grpo import GuiSample


class UiR1WarmupDataset:
    """Convert UI-R1-low records to the action protocol used by this project.

    Only labels that can be mapped without guessing are retained:

    - ``click``: bbox center -> normalized ``[x, y]`` point;
    - ``navigate_back``: ``press_back``.

    ``open_app`` is outside the current executor action space, while the released
    ``scroll`` and ``input_text`` records do not contain the structured arguments
    required by our protocol. They are audited in ``summary`` instead of silently
    becoming noisy targets.
    """

    def __init__(self, data_file: str | Path, image_dir: str | Path):
        self.data_file = Path(data_file).expanduser().resolve()
        self.image_dir = Path(image_dir).expanduser().resolve()
        with self.data_file.open("r", encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list):
            raise ValueError(f"UI-R1 warm-up data must be a JSON list: {self.data_file}")

        self.samples: list[GuiSample] = []
        skipped: collections.Counter[str] = collections.Counter()
        raw_actions: collections.Counter[str] = collections.Counter()
        source_counts: collections.Counter[str] = collections.Counter()

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                skipped["invalid_record"] += 1
                continue

            action = str(record.get("action", "")).strip().lower()
            raw_actions[action or "missing"] += 1
            source_counts[str(record.get("data_source", "unknown"))] += 1
            image_path = self.image_dir / str(record.get("img_filename", ""))
            if not image_path.is_file():
                skipped["missing_image"] += 1
                continue

            target: dict[str, Any]
            ui_positions: list[list[float]] = []
            if action == "click":
                bbox = record.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    skipped["click_missing_bbox"] += 1
                    continue
                try:
                    x1, y1, x2, y2 = (float(value) for value in bbox)
                except (TypeError, ValueError):
                    skipped["click_invalid_bbox"] += 1
                    continue
                with Image.open(image_path) as image:
                    width, height = image.size
                if width <= 0 or height <= 0 or x2 < x1 or y2 < y1:
                    skipped["click_invalid_bbox"] += 1
                    continue
                center_x = max(0, min(1000, round(((x1 + x2) / 2.0) / width * 1000)))
                center_y = max(0, min(1000, round(((y1 + y2) / 2.0) / height * 1000)))
                target = {
                    "kind": "POINT",
                    "action_type": 4,
                    "point": [center_x, center_y],
                    "text": "",
                    "duration": None,
                }
                ui_positions = [[y1 / height, x1 / width, (y2 - y1) / height, (x2 - x1) / width]]
            elif action == "navigate_back":
                with Image.open(image_path) as image:
                    width, height = image.size
                target = {
                    "kind": "PRESS",
                    "action_type": 5,
                    "press": "BACK",
                    "text": "",
                    "duration": None,
                }
            else:
                skipped[f"unsupported_{action or 'missing'}"] += 1
                continue

            self.samples.append(
                GuiSample(
                    episode_id=f"ui-r1-warmup-{index:04d}",
                    step_id=0,
                    episode_length=1,
                    instruction=str(record.get("instruction", "")).strip(),
                    image_path=str(image_path),
                    width=width,
                    height=height,
                    ui_positions=ui_positions,
                    history=[],
                    target=target,
                )
            )

        if not self.samples:
            raise RuntimeError(
                f"No compatible UI-R1 warm-up samples found in {self.data_file} with images under {self.image_dir}"
            )

        usable_actions = collections.Counter(sample.target["kind"] for sample in self.samples)
        self.summary: dict[str, Any] = {
            "manifest_version": 1,
            "dataset": "UI-R1-low",
            "purpose": "stage0_auxiliary_sft_warmup",
            "source_file": str(self.data_file),
            "source_sha256": hashlib.sha256(self.data_file.read_bytes()).hexdigest(),
            "image_dir": str(self.image_dir),
            "coordinate_convention": "source_bbox_xyxy_pixels_to_model_xy_normalized_0_1000",
            "raw_samples": len(records),
            "usable_samples": len(self.samples),
            "raw_action_counts": dict(sorted(raw_actions.items())),
            "usable_target_counts": dict(sorted(usable_actions.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "skipped_counts": dict(sorted(skipped.items())),
            "policy": {
                "click": "use_bbox_center",
                "navigate_back": "map_to_press_back",
                "open_app": "skip_not_in_executor_action_space",
                "scroll": "skip_missing_direction_parameter",
                "input_text": "skip_missing_structured_text_parameter",
            },
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> GuiSample:
        return self.samples[index]
