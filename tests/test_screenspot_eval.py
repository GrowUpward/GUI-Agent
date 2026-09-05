from gui_agent.evaluation.screenspot import coordinate_to_pixel, evaluate_record, point_in_bbox, summarize_records


def test_coordinate_conversion_and_bbox_hit() -> None:
    point = coordinate_to_pixel([500.0, 250.0], width=200, height=400)
    assert point == [100.0, 100.0]
    assert point_in_bbox(point, [90, 90, 110, 110])
    assert coordinate_to_pixel([1001.0, 0.0], 200, 400) is None


def test_evaluate_record_tracks_format_action_and_parameter() -> None:
    sample = {
        "img_filename": "example.png",
        "instruction": "click it",
        "data_type": "icon",
        "data_source": "android",
        "bbox": [90, 90, 110, 110],
    }
    completion = {
        "raw_text": "{'action': 'click', 'coordinate': [500, 250]}",
        "action_text": "{'action': 'click', 'coordinate': [500, 250]}",
        "eos_terminated": True,
    }
    record = evaluate_record(sample, completion, width=200, height=400)
    assert record["raw_strict_format"] is True
    assert record["action_correct"] is True
    assert record["coordinate_valid"] is True
    assert record["bbox_hit"] is True


def test_summary_uses_all_samples_as_denominator() -> None:
    records = [
        {
            "raw_strict_format": True,
            "parse_success": True,
            "eos_terminated": True,
            "extra_content": False,
            "action_correct": True,
            "coordinate_valid": True,
            "bbox_hit": True,
            "center_distance_px": 1.0,
        },
        {
            "raw_strict_format": False,
            "parse_success": False,
            "eos_terminated": False,
            "extra_content": False,
            "action_correct": False,
            "coordinate_valid": False,
            "bbox_hit": False,
            "center_distance_px": None,
        },
    ]
    metrics = summarize_records(records)
    assert metrics["action_type_accuracy"] == 0.5
    assert metrics["parameter_accuracy"] == 0.5
    assert metrics["grounding_accuracy_given_valid_coordinate"] == 1.0
