from episodic.trainers.rewards import action_format_reward, _score_action, _text


def test_wellformed_action_scores_full():
    assert _score_action('ACTION Bash({"command": "ls"})') == 1.0
    assert _score_action('ACTION Edit({"file_path": "a.py"})') == 1.0


def test_partial_and_empty_completions():
    assert 0.0 < _score_action("ACTION plus some words") < 1.0
    assert _score_action("just prose, no action") == 0.0
    assert _score_action("") == 0.0


def test_batch_length_matches_and_ranks():
    out = action_format_reward(completions=['ACTION Write({"file_path": "x"})', "nonsense"])
    assert len(out) == 2 and out[0] > out[1]


def test_text_handles_message_shaped_completions():
    assert _text([{"role": "assistant", "content": "ACTION Read({})"}]) == "ACTION Read({})"
    assert _text({"content": "hi"}) == "hi"
