from gui_agent.training.grpo import (
    SYSTEM_PROMPT,
    normalize_pred_action,
    parse_action_text,
    reward_completion,
    target_from_step,
    target_to_sft_dict,
    truncate_at_balanced_dict,
)
from gui_agent.training.sft import action_label_from_pred, action_label_from_target


def test_parse_python_style_click() -> None:
    parsed, ok = parse_action_text("{'action': 'click', 'coordinate': [500, 400]}")
    assert ok
    assert parsed == {"action": "click", "coordinate": [500, 400]}


def test_normalize_input_text() -> None:
    assert normalize_pred_action({"action": "input_text", "text": "北京"}) == {"TYPE": "北京"}


def test_training_protocol_uses_native_press_key_action() -> None:
    target = {"kind": "PRESS", "press": "home"}
    assert target_to_sft_dict(target) == "{'action': 'press', 'key': 'HOME'}"
    assert normalize_pred_action({"action": "press", "key": "HOME"}) == {"PRESS": "HOME"}
    assert action_label_from_target(target) == "press"
    assert action_label_from_pred({"PRESS": "HOME"}) == "press"


def test_training_protocol_uses_duration_wait_action() -> None:
    assert target_to_sft_dict({"kind": "WAIT", "duration": 1000}) == "{'action': 'wait', 'duration': 500}"
    assert normalize_pred_action({"action": "wait", "duration": 500}) == {"duration": 500}


def test_no_action_is_wait_and_long_point_is_folded_into_click() -> None:
    wait_target = target_from_step({"result_action_type": 1, "duration": 1000})
    assert wait_target["kind"] == "WAIT"
    assert wait_target["duration"] == 500

    long_point_target = target_from_step(
        {
            "result_action_type": 0,
            "duration": 1000,
            "result_touch_yx": [0.4, 0.3],
            "result_lift_yx": [0.4, 0.3],
        }
    )
    assert long_point_target["kind"] == "POINT"
    assert long_point_target["point"] == [300, 400]
    assert long_point_target["source_action"] == "long_press"
    assert target_to_sft_dict(long_point_target) == "{'action': 'click', 'coordinate': [300, 400]}"


def test_impossible_source_label_is_folded_into_stop() -> None:
    target = target_from_step({"result_action_type": 11})
    assert target["kind"] == "STATUS"
    assert target["status"] == "finish"
    assert target["source_status"] == "impossible"
    assert target_to_sft_dict(target) == "{'action': 'stop'}"


def test_system_prompt_declares_the_complete_training_action_space() -> None:
    assert "{'action': 'press', 'key': 'BACK'|'HOME'|'ENTER'}" in SYSTEM_PROMPT
    assert "{'action': 'wait', 'duration': 500}" in SYSTEM_PROMPT
    assert "impossible" not in SYSTEM_PROMPT
    assert "press_back" not in SYSTEM_PROMPT


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


def test_truncate_repeated_assistant_turn_to_first_action() -> None:
    text = (
        "{'action': 'click', 'coordinate': [512, 612]}\n"
        "assistant\n<think>\n\n</think>\n\n"
        "{'action': 'click', 'coordinate': [512, 630]}"
    )
    action_text = truncate_at_balanced_dict(text)
    parsed, ok = parse_action_text(action_text)
    assert ok
    assert parsed == {"action": "click", "coordinate": [512, 612]}


def test_truncate_keeps_brace_inside_quoted_text() -> None:
    text = "{'action': 'input_text', 'text': '集合 {A}'} trailing"
    assert truncate_at_balanced_dict(text) == "{'action': 'input_text', 'text': '集合 {A}'}"
