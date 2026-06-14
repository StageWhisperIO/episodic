import os
import sys
import math
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from episodic import new_episode
from episodic.core.reward import reward_vector

import stages


def _make_step(index, tool, input_data, observation):
    return {
        "index": index,
        "ts": "2024-01-01T00:00:00Z",
        "type": "tool_call",
        "tool": tool,
        "intent": "do something",
        "input": input_data,
        "observation": observation,
    }


def _make_episodes():
    ep1 = new_episode("ep1", intent="fix the login bug")
    ep1["steps"] = [
        _make_step(0, "read_file", {"path": "auth.py"}, "file contents here"),
        _make_step(1, "edit_file", {"path": "auth.py", "content": "fixed"}, "file saved"),
    ]
    ep1["tests"] = [{"ts": "2024-01-01T00:00:00Z", "framework": "pytest", "passed": 5, "failed": 0, "ok": True}]
    ep1["human_feedback"] = [{"ts": "2024-01-01T00:00:00Z", "label": "accepted_as_is"}]
    ep1["outcome"]["status"] = "accepted"
    ep1["stats"]["file_edits"] = 1
    ep1["stats"]["file_reads"] = 2
    ep1["stats"]["shell_commands"] = 0
    ep1["stats"]["tests_run"] = 1
    ep1["reward_vector"] = reward_vector(ep1)

    ep2 = new_episode("ep2", intent="add unit tests for parser")
    ep2["steps"] = [
        _make_step(0, "read_file", {"path": "parser.py"}, "parser code"),
        _make_step(1, "write_file", {"path": "test_parser.py", "content": "tests"}, "tests written"),
        _make_step(2, "shell", {"command": "pytest"}, "5 passed"),
    ]
    ep2["tests"] = [{"ts": "2024-01-01T00:00:00Z", "framework": "pytest", "passed": 5, "failed": 0, "ok": True}]
    ep2["human_feedback"] = [{"ts": "2024-01-01T00:00:00Z", "label": "useful"}]
    ep2["outcome"]["status"] = "merged"
    ep2["stats"]["file_edits"] = 2
    ep2["stats"]["file_reads"] = 1
    ep2["stats"]["shell_commands"] = 1
    ep2["stats"]["tests_run"] = 1
    ep2["reward_vector"] = reward_vector(ep2)

    ep3 = new_episode("ep3", intent="fix the login bug")
    ep3["steps"] = [
        _make_step(0, "read_file", {"path": "wrong.py"}, "wrong file"),
    ]
    ep3["tests"] = [{"ts": "2024-01-01T00:00:00Z", "framework": "pytest", "passed": 0, "failed": 3, "ok": False}]
    ep3["human_feedback"] = [{"ts": "2024-01-01T00:00:00Z", "label": "wrong"}]
    ep3["outcome"]["status"] = "failed"
    ep3["stats"]["file_edits"] = 0
    ep3["stats"]["file_reads"] = 1
    ep3["stats"]["shell_commands"] = 0
    ep3["stats"]["tests_run"] = 1
    ep3["reward_vector"] = reward_vector(ep3)

    ep4 = new_episode("ep4", intent="refactor database layer")
    ep4["steps"] = [
        _make_step(0, "read_file", {"path": "db.py"}, "db code"),
        _make_step(1, "edit_file", {"path": "db.py", "content": "refactored"}, "saved"),
        _make_step(2, "shell", {"command": "pytest"}, "3 passed"),
    ]
    ep4["tests"] = [{"ts": "2024-01-01T00:00:00Z", "framework": "pytest", "passed": 3, "failed": 0, "ok": True}]
    ep4["human_feedback"] = []
    ep4["outcome"]["status"] = "accepted"
    ep4["stats"]["file_edits"] = 3
    ep4["stats"]["file_reads"] = 1
    ep4["stats"]["shell_commands"] = 1
    ep4["stats"]["tests_run"] = 1
    ep4["reward_vector"] = reward_vector(ep4)

    return [ep1, ep2, ep3, ep4]


def main():
    episodes = _make_episodes()
    composites = [ep["reward_vector"]["composite"] for ep in episodes]

    good, stats = stages.quality_filter(episodes, min_composite=0.5)
    assert stats["total"] == 4, f"expected 4 total, got {stats['total']}"
    assert stats["kept"] >= 1, "should keep at least one episode"
    good_ids = {ep["id"] for ep in good}
    assert "ep3" not in good_ids, "ep3 should be filtered (wrong label + failed + low composite)"
    assert stats["kept"] + stats["dropped"] == stats["total"]

    sft_rows = stages.sft_dataset(good)
    assert len(sft_rows) == len(good), "sft rows should match good episodes"
    for row in sft_rows:
        assert "messages" in row
        assert len(row["messages"]) == 2
        assert row["messages"][0]["role"] == "user"
        assert row["messages"][1]["role"] == "assistant"
        assert "meta" in row
        assert "episode_id" in row["meta"]

    pairs = stages.preference_pairs(episodes)
    assert len(pairs) >= 1, f"expected at least 1 pair, got {len(pairs)}"
    for pair in pairs:
        assert "prompt" in pair
        assert "chosen" in pair
        assert "rejected" in pair
        assert "meta" in pair

    model = stages.reward_model_train(episodes)
    assert "weights" in model
    assert "features" in model
    assert "r2" in model
    assert len(model["weights"]) == len(model["features"]), "weights length must match features"
    assert math.isfinite(model["r2"]), f"r2 must be finite, got {model['r2']}"

    transitions = stages.rl_batches(episodes)
    assert len(transitions) > 0, "should have at least one transition"
    terminal_transitions = [t for t in transitions if t["terminal"]]
    assert len(terminal_transitions) > 0, "should have terminal transitions"
    assert len(terminal_transitions) == len(episodes), (
        f"expected {len(episodes)} terminal transitions, got {len(terminal_transitions)}"
    )
    ep2_composite = episodes[1]["reward_vector"]["composite"]
    ep2_terminal = next(
        t for t in terminal_transitions
        if t["state"]["intent"] == episodes[1]["intent"]
        and t["state"]["step_index"] == len(episodes[1]["steps"]) - 1
    )
    assert abs(ep2_terminal["reward"] - ep2_composite) < 1e-9, (
        f"ep2 terminal reward mismatch: {ep2_terminal['reward']} != {ep2_composite}"
    )

    eval_result = stages.evaluate(episodes, model)
    assert "n" in eval_result
    assert eval_result["n"] == len(episodes)
    assert math.isfinite(eval_result["pearson"]), "pearson must be finite"
    assert math.isfinite(eval_result["mean_abs_err"]), "mae must be finite"

    tmp = tempfile.mkdtemp()
    stages.write_jsonl(os.path.join(tmp, "test.jsonl"), sft_rows)
    assert os.path.exists(os.path.join(tmp, "test.jsonl"))

    print("ok")


if __name__ == "__main__":
    main()
