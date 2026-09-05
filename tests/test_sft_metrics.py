from gui_agent.training.sft import latest_logged_eval_metrics


def test_latest_logged_eval_metrics_uses_last_evaluation() -> None:
    history = [
        {"loss": 1.0, "step": 1},
        {"eval_loss": 0.8, "eval_runtime": 2.0, "step": 100},
        {"loss": 0.7, "step": 101},
        {"eval_loss": 0.6, "eval_runtime": 2.1, "step": 200},
        {"train_runtime": 10.0, "step": 200},
    ]

    assert latest_logged_eval_metrics(history) == {"eval_loss": 0.6, "eval_runtime": 2.1}


def test_latest_logged_eval_metrics_handles_no_evaluation() -> None:
    assert latest_logged_eval_metrics([{"loss": 1.0}]) == {}
