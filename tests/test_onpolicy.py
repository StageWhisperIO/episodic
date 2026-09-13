import json

import pytest

from episodic.eval import editfmt, onpolicy, redgreen

TWO_REGION_BUGGY = "def add(a, b):\n    return a - b\n\n\ndef double(x):\n    return x\n"
TWO_REGION_FIXED = "def add(a, b):\n    return a + b\n\n\ndef double(x):\n    return x * 2\n"
TWO_REGION_TEST = ("from solution import add, double\n\n\n"
                   "def test_solve():\n    assert add(2, 3) == 5\n    assert double(4) == 8\n")

WIDE_CORRECT_EDIT = {"path": "solution.py", "start": 1, "end": 2,
                     "body": "def add(a, b):\n    return a + b"}


@pytest.fixture(scope="module")
def two_region_episode(tmp_path_factory):
    base = tmp_path_factory.mktemp("onpolicy")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("EPISODIC_HOME", str(base / "home"))
        yield redgreen.build_task(str(base / "repos"), "ep_onpolicy_two_region",
                                  TWO_REGION_BUGGY, TWO_REGION_FIXED, TWO_REGION_TEST,
                                  bug_class="multi")


def test_parse_edits_round_trips_a_known_block():
    text = "EDIT solution.py 2-2\n    return a + b\nENDEDIT"
    edits = onpolicy.parse_edits(text)
    assert edits == [{"path": "solution.py", "start": 2, "end": 2, "body": "    return a + b"}]
    assert onpolicy.parse_edits(onpolicy.render_edits(edits)) == edits


def test_minimal_correction_preserves_a_correct_edit_and_fixes_only_the_failing_region(two_region_episode):
    files = editfmt._files_of(two_region_episode)
    gold_edits = onpolicy.gold_edits_for(two_region_episode)
    assert len(gold_edits) == 2

    attempt_text = onpolicy.render_edits([WIDE_CORRECT_EDIT])
    localized = onpolicy.localize_failure(two_region_episode, attempt_text)
    assert localized["passes"] is False
    assert len(localized["diverging_regions"]) == 1
    assert localized["diverging_regions"][0]["start"] == 6

    corrected_text = onpolicy.minimal_correction(attempt_text, localized["diverging_regions"], files)
    gold_text = onpolicy.render_edits(gold_edits)
    assert "def add(a, b):" in corrected_text
    assert "return x * 2" in corrected_text
    assert onpolicy.edit_distance(attempt_text, corrected_text) < onpolicy.edit_distance(attempt_text, gold_text)


def test_build_correction_row_has_the_trainer_row_shape():
    row = onpolicy.build_correction_row({"id": "ep_x"}, "prompt text", "target text")
    assert row["messages"][0] == {"role": "user", "content": "prompt text"}
    assert row["messages"][1] == {"role": "assistant", "content": "target text"}
    assert row["meta"] == {"id": "ep_x"}


def test_build_correction_dataset_with_fake_generate_and_gold_oracle(two_region_episode, tmp_path):
    attempt_text = onpolicy.render_edits([WIDE_CORRECT_EDIT])

    def fake_generate(model, messages):
        return attempt_text

    out_path = str(tmp_path / "onpolicy.jsonl")
    summary = onpolicy.build_correction_dataset(
        [two_region_episode], fake_generate, onpolicy.gold_expert_fn, out_path)

    assert summary["total"] == 1
    assert summary["skipped"] == 0
    assert summary["positive"] == 0
    assert summary["corrected"] == 1
    assert summary["mean_minimality_gap"] > 0

    onpolicy_rows = [json.loads(line) for line in open(out_path)]
    offpolicy_rows = [json.loads(line) for line in open(summary["offpolicy_path"])]
    assert len(onpolicy_rows) == 1 and len(offpolicy_rows) == 1
    assert onpolicy_rows[0]["meta"]["id"] == two_region_episode["id"]
    assert "def add(a, b):" in onpolicy_rows[0]["messages"][1]["content"]
    assert onpolicy_rows[0]["messages"][1]["content"] != offpolicy_rows[0]["messages"][1]["content"]
