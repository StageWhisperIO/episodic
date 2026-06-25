from episodic import fidelity


def test_exact_match_scores_perfect():
    s = fidelity.score_observation("same text", "same text")
    assert s["exact"] is True
    assert s["composite"] == 1.0
    assert all(s[dim] == 1.0 for dim in fidelity.DIMENSIONS)


def test_runtime_metadata_is_not_penalized():
    gt = "process started pid 18204 at 2026-06-14T10:00:00Z"
    pred = "process started pid 42731 at 2026-06-14T11:22:33Z"
    s = fidelity.score_observation(pred, gt)
    assert s["exact"] is False
    assert s["factuality"] == 1.0
    assert s["consistency"] == 1.0
    assert s["composite"] > 0.9


def test_fabrication_lowers_consistency():
    gt = "file written"
    pred = "file written and database dropped catastrophically"
    s = fidelity.score_observation(pred, gt)
    assert s["factuality"] == 1.0
    assert s["consistency"] < 0.5


def test_response_type_mismatch_zeros_realism():
    s = fidelity.score_observation("Error: build failed: missing dependency", "Build succeeded")
    assert s["realism"] == 0.0
    assert s["predicted_type"] == "error"
    assert s["ground_truth_type"] == "success"


def test_empty_both_is_perfect():
    s = fidelity.score_observation("", "")
    assert s["composite"] == 1.0


def test_empty_prediction_does_not_inflate_consistency():
    s = fidelity.score_observation("", "ls\nfile1.txt\nfile2.txt")
    assert s["consistency"] == 0.0
    assert s["factuality"] == 0.0
    assert s["composite"] < 0.2


def test_classify_content_detects_runtime_json_and_type():
    info = fidelity.classify_content('{"pid": 5, "ts": "2026-06-14T10:00:00Z"}')
    assert info["is_json"] is True
    assert info["runtime_metadata"] >= 1
    assert info["response_type"] == "success"
    err = fidelity.classify_content("Traceback (most recent call last): Error")
    assert err["response_type"] == "error"
