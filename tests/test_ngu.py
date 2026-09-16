import random

import pytest

from episodic.eval import redgreen
from episodic.trainers import rewards, sao


def test_ngu_classify_all_at_max_is_filter():
    assert sao.ngu_classify([1.0, 1.0, 1.0]) == "filter"


def test_ngu_classify_mixed_is_train():
    assert sao.ngu_classify([1.0, 0.0, 0.5]) == "train"


def test_ngu_classify_all_zero_is_retry():
    assert sao.ngu_classify([0.0, 0.0, 0.0]) == "retry"


def test_ngu_classify_all_equal_below_max_is_retry():
    assert sao.ngu_classify([0.3, 0.3, 0.3]) == "retry"


def _samples(rewards):
    return [(f"tok{i}", f"lp{i}") for i in range(len(rewards))]


def test_scheduler_retries_all_failed_group_under_seeded_rng():
    scheduler = sao.NGUScheduler(k=2, p=1.0, max_rounds=5, rng=random.Random(0))
    result = scheduler.submit("p1", _samples([0.0, 0.0]), [0.0, 0.0])
    assert result["action"] == "retry"
    assert result["rounds"] == 1
    assert scheduler.pending("p1")


def test_scheduler_accumulates_and_emits_trainable_group_after_a_later_solve():
    scheduler = sao.NGUScheduler(k=2, p=1.0, max_rounds=5, rng=random.Random(0))
    scheduler.submit("p1", _samples([0.0, 0.0]), [0.0, 0.0])
    result = scheduler.submit("p1", _samples([1.0, 0.0]), [1.0, 0.0])
    assert result["action"] == "train"
    assert result["rounds"] == 2
    assert result["rewards"] == [0.0, 0.0, 1.0, 0.0]
    assert len(result["samples"]) == 4
    assert not scheduler.pending("p1")


def test_scheduler_filters_all_solved_group_immediately():
    scheduler = sao.NGUScheduler(k=3, p=1.0, max_rounds=5, rng=random.Random(0))
    result = scheduler.submit("p2", _samples([1.0, 1.0, 1.0]), [1.0, 1.0, 1.0])
    assert result["action"] == "filter"
    assert result["samples"] is None
    assert not scheduler.pending("p2")


def test_scheduler_drops_after_max_rounds_of_all_failed():
    scheduler = sao.NGUScheduler(k=1, p=1.0, max_rounds=2, rng=random.Random(0))
    first = scheduler.submit("p3", _samples([0.0]), [0.0])
    assert first["action"] == "retry"
    second = scheduler.submit("p3", _samples([0.0]), [0.0])
    assert second["action"] == "drop"
    assert second["rounds"] == 2
    assert not scheduler.pending("p3")


def test_scheduler_drops_when_retry_coin_fails():
    scheduler = sao.NGUScheduler(k=1, p=0.0, max_rounds=5, rng=random.Random(0))
    result = scheduler.submit("p4", _samples([0.0]), [0.0])
    assert result["action"] == "drop"
    assert result["rounds"] == 1
    assert not scheduler.pending("p4")


def test_scheduler_is_deterministic_under_a_fixed_seed():
    def run():
        scheduler = sao.NGUScheduler(k=1, p=0.5, max_rounds=6, rng=random.Random(123))
        actions = []
        for _ in range(6):
            action = scheduler.submit("p5", _samples([0.0]), [0.0])["action"]
            actions.append(action)
            if action != "retry":
                break
        return actions

    assert run() == run()


@pytest.fixture(scope="module")
def solution_episode(tmp_path_factory):
    repos_dir = str(tmp_path_factory.mktemp("ngu_repos"))
    buggy = "def solve(a, b):\n    return a - b\n"
    fixed = "def solve(a, b):\n    return a + b\n"
    imports, visible, hidden = [], redgreen._check("solve(2, 3) == 5"), redgreen._check("solve(4, 5) == 9")
    test_src = redgreen._module(imports, visible, hidden)
    return redgreen.build_task(repos_dir, "ep_ngu_operator", buggy, fixed, test_src, bug_class="operator")


def _numbered_edit(episode, body):
    path = episode["diffs"][0]["file"]
    return f"EDIT {path} 1-100000\n{body.rstrip(chr(10))}\nENDEDIT"


def test_gate_all_tests_reward_is_binary_on_full_pass(solution_episode):
    gold = _numbered_edit(solution_episode, "def solve(a, b):\n    return a + b\n")
    broken = _numbered_edit(solution_episode, "def solve(a, b):\n    return a - b\n")
    scores = rewards.gate_all_tests_reward(
        completions=[gold, broken, "no edit here"],
        meta=[solution_episode, solution_episode, None])
    assert scores == [1.0, 0.0, 0.0]
