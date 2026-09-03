from gui_agent.training.grpo import (
    normalize_pred_action,
    parse_action_text,
    reward_completion,
    truncate_at_balanced_dict,
)


def test_parse_python_style_click() -> None:
    parsed, ok = parse_action_text("{'action': 'click', 'coordinate': [500, 400]}")
    assert ok
    assert parsed == {"action": "click", "coordinate": [500, 400]}


def test_normalize_input_text() -> None:
    assert normalize_pred_action({"action": "input_text", "text": "北京"}) == {"TYPE": "北京"}


def test_exact_click_gets_full_reward() -> None:
    target = {"kind": "POINT", "point": [500, 500]}
    boxes = [[0.4, 0.4, 0.2, 0.2]]
    reward = reward_completion("{'action': 'click', 'coordinate': [500, 500]}", target, boxes)
    assert reward == 1.0


def test_wrong_action_is_rejected() -> None:
    target = {"kind": "TYPE", "text": "hello"}
    reward = reward_completion("{'action': 'click', 'coordinate': [500, 500]}", target, [])
    assert reward == -1.0


def test_truncate_generation_after_first_dict() -> None:
    text = "prefix {'action': 'stop'} trailing {'action': 'click'}"
    assert truncate_at_balanced_dict(text) == "{'action': 'stop'}"
