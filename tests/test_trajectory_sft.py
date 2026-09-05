from types import SimpleNamespace

import torch

from gui_agent.training.sft import CaguiListDataset, CaguiTrajectoryDataset, find_subsequence_from


def _sample(episode_id: str, step_id: int) -> SimpleNamespace:
    return SimpleNamespace(episode_id=episode_id, step_id=step_id)


def test_trajectory_windows_preserve_steps_without_crossing_episodes() -> None:
    steps = CaguiListDataset(
        [
            _sample("b", 1),
            _sample("a", 2),
            _sample("a", 0),
            _sample("b", 0),
            _sample("a", 1),
        ]
    )
    trajectories = CaguiTrajectoryDataset(steps, window_steps=2)

    assert sum(len(window.steps) for window in trajectories.windows) == len(steps)
    assert all(len(window.steps) <= 2 for window in trajectories.windows)
    assert all(all(step.episode_id == window.episode_id for step in window.steps) for window in trajectories.windows)
    assert [[step.step_id for step in window.steps] for window in trajectories.windows] == [[0, 1], [2], [0, 1]]


def test_find_subsequence_from_selects_each_repeated_answer_in_order() -> None:
    sequence = torch.tensor([1, 7, 8, 2, 7, 8, 3])
    pattern = torch.tensor([7, 8])
    first = find_subsequence_from(sequence, pattern, 0)
    second = find_subsequence_from(sequence, pattern, first + len(pattern))
    assert first == 1
    assert second == 4

