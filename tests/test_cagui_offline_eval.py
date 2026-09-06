from types import SimpleNamespace

import pytest

from gui_agent.evaluation.cagui_offline import official_click_match, official_step_match, summarize_records


def test_official_click_uses_expanded_target_box() -> None:
    matched, rule, _ = official_click_match([615, 500], [500, 500], [[0.4, 0.4, 0.2, 0.2]])
    assert matched is True
    assert rule == "expanded_bbox"


def test_official_click_falls_back_to_distance_after_box_miss() -> None:
    matched, rule, distance = official_click_match(
        [600, 500],
        [500, 500],
        [[0.45, 0.45, 0.1, 0.1]],
    )
    assert matched is True
    assert rule == "distance_0.14_fallback"
    assert distance == pytest.approx(0.1)


def test_official_type_match_uses_substring_but_preserves_strict_diagnostic() -> None:
    sample = SimpleNamespace(target={"kind": "TYPE", "text": "hello world"}, ui_positions=[])
    result = official_step_match(sample, {"TYPE": "hello"})
    assert result["type_match"] is True
    assert result["exact_match"] is True
    assert result["parameter_detail"]["strict_text_exact"] is False


def test_episode_success_and_prefix_goal_progress() -> None:
    base = {
        "format_hit": True,
        "raw_strict_format": True,
        "eos_terminated": True,
        "type_match": True,
        "gold_action": "click",
        "parameter_detail": {"strict_bbox_hit": True},
    }
    records = [
        dict(base, episode_id="a", step_id=0, exact_match=True),
        dict(base, episode_id="a", step_id=1, exact_match=False),
        dict(base, episode_id="a", step_id=2, exact_match=True),
        dict(base, episode_id="b", step_id=0, exact_match=True),
    ]
    metrics = summarize_records(records)
    assert metrics["episode_success_rate"] == 0.5
    assert metrics["goal_progress"] == round(((1 / 3) + 1) / 2, 6)
