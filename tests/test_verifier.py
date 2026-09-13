from episodic.eval import verifier


def test_softmax_uniform_and_peaked():
    uniform = verifier.softmax({"a": 0.0, "b": 0.0})
    assert abs(uniform["a"] - 0.5) < 1e-9
    peaked = verifier.softmax({"a": 10.0, "b": 0.0})
    assert peaked["a"] > 0.99
    assert verifier.softmax({}) == {}


def test_expected_rating_spans_zero_to_one():
    top = {option: 0.0 for option in verifier.RATING_OPTIONS}
    top["9"] = 20.0
    assert verifier.expected_rating(top) > 0.95
    bottom = {option: 0.0 for option in verifier.RATING_OPTIONS}
    bottom["0"] = 20.0
    assert verifier.expected_rating(bottom) < 0.05
    assert verifier.expected_rating({}) == 0.0


def test_fine_grained_reward_uses_all_criteria():
    def always_high(messages, options):
        return {option: (20.0 if option == "9" else 0.0) for option in options}

    result = verifier.fine_grained_reward("task", "diff", None, always_high)
    assert result["reward"] > 0.95
    assert set(result["criteria"]) == set(verifier.DEFAULT_CRITERIA)


def _content_aware(messages, options):
    content = messages[-1]["content"]
    a_block = content.split("Candidate A (unified diff):\n", 1)[1].split("\n\nCandidate B")[0].strip()
    return {"A": 3.0, "B": 0.0} if a_block == "GOLD" else {"A": 0.0, "B": 3.0}


def test_compare_prefers_the_better_candidate():
    result = verifier.compare("task", "GOLD", "EMPTY", {"Overall": "solves it"}, _content_aware)
    assert result["reward_a"] > 0.5
    assert abs(result["reward_a"] + result["reward_b"] - 1.0) < 1e-9


def test_compare_swap_cancels_position_bias():
    def always_prefers_a(messages, options):
        return {"A": 2.0, "B": 0.0}

    biased = verifier.compare("task", "x", "y", {"Overall": "solves it"}, always_prefers_a, swap=True)
    assert abs(biased["reward_a"] - 0.5) < 1e-9
