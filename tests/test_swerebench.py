import subprocess

import pytest

from episodic.eval import gate, swerebench
from episodic.worldmodel import validate as wm

SAMPLE = {
    "instance_id": "acme__widgets-42",
    "repo": "acme/widgets",
    "base_commit": "deadbeef",
    "problem_statement": "add() returns the wrong value.",
    "patch": ("diff --git a/solution.py b/solution.py\n"
              "--- a/solution.py\n+++ b/solution.py\n"
              "@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"),
    "test_patch": ("diff --git a/test_solution.py b/test_solution.py\n"
                   "--- a/test_solution.py\n+++ b/test_solution.py\n"
                   "@@ -1 +1 @@\n-x\n+y\n"),
    "FAIL_TO_PASS": '["test_solution.py::test_add"]',
    "PASS_TO_PASS": "[]",
}


def test_instance_to_episode_maps_metadata_only():
    episode = swerebench.instance_to_episode(SAMPLE)
    repo_state = episode["repo_state"]
    assert repo_state["remote_url"] == "https://github.com/acme/widgets.git"
    assert repo_state["root"] is None
    assert repo_state["base_commit"] == "deadbeef"
    assert repo_state["setup_patch"] == SAMPLE["test_patch"]
    assert [d["file"] for d in episode["diffs"]] == ["solution.py"]
    command = episode["commands"][0]["command"]
    assert '-k "test_add"' in command
    assert "test_solution.py" in command
    assert episode["labels"] == ["swe", "swe-rebench", "certified_by_source"]


def test_source_diffs_exclude_verifier_files():
    instance = dict(SAMPLE, patch=SAMPLE["patch"] +
                    "diff --git a/test_solution.py b/test_solution.py\n"
                    "--- a/test_solution.py\n+++ b/test_solution.py\n@@ -1 +1 @@\n-x\n+y\n")
    episode = swerebench.instance_to_episode(instance)
    assert [d["file"] for d in episode["diffs"]] == ["solution.py"]


def test_instance_without_fail_to_pass_or_source_is_skipped():
    assert swerebench.instance_to_episode(dict(SAMPLE, FAIL_TO_PASS="[]")) is None
    assert swerebench.instance_to_episode(dict(SAMPLE, patch="")) is None


def test_node_parsing_handles_class_and_params():
    ids = ["p/test_a.py::TestC::test_x[case1]", "p/test_a.py::TestC::test_y", "q/test_b.py::test_z"]
    assert swerebench._node_test_names(ids) == ["test_x", "test_y", "test_z"]
    assert swerebench._node_paths(ids) == ["p/test_a.py", "q/test_b.py"]


def test_as_list_handles_json_and_space_and_list():
    assert swerebench._as_list('["a::b", "c::d"]') == ["a::b", "c::d"]
    assert swerebench._as_list("a::b c::d") == ["a::b", "c::d"]
    assert swerebench._as_list(["x"]) == ["x"]
    assert swerebench._as_list(None) == []


def _git(cwd, *args):
    subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True, text=True)


@pytest.fixture
def local_swe_episode(tmp_path, monkeypatch):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path / "home"))
    origin = tmp_path / "origin"
    origin.mkdir()
    (origin / "solution.py").write_text("def add(a, b):\n    return a - b\n")
    _git(str(origin), "init", "-q")
    _git(str(origin), "config", "user.email", "t@t")
    _git(str(origin), "config", "user.name", "t")
    _git(str(origin), "add", "-A")
    _git(str(origin), "commit", "-q", "-m", "base (buggy, no test)")
    base = subprocess.run(["git", "-C", str(origin), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()

    instance = {
        "instance_id": "acme__widgets-1",
        "repo": "acme/widgets",
        "base_commit": base,
        "problem_statement": "add is wrong",
        "patch": ("diff --git a/solution.py b/solution.py\n"
                  "--- a/solution.py\n+++ b/solution.py\n"
                  "@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"),
        "test_patch": ("diff --git a/test_solution.py b/test_solution.py\n"
                       "new file mode 100644\n--- /dev/null\n+++ b/test_solution.py\n"
                       "@@ -0,0 +1,4 @@\n+from solution import add\n+\n"
                       "+def test_add():\n+    assert add(1, 2) == 3\n"),
        "FAIL_TO_PASS": ["test_solution.py::test_add"],
    }
    episode = swerebench.instance_to_episode(instance)
    episode["repo_state"]["remote_url"] = str(origin)
    return episode


def test_setup_patch_reconstructs_red_and_oracle_reaches_green(local_swe_episode):
    empty = gate.graded_score(local_swe_episode, gate.empty_runner)
    assert empty["ok"] is False and empty["pass_fraction"] < 1.0

    gold = wm._unified_diff(local_swe_episode)
    oracle = gate.graded_score(local_swe_episode, wm._oracle_diff_runner(gold))
    assert oracle["ok"] is True and oracle["pass_fraction"] == 1.0


def test_mirror_cache_is_populated_and_reused(local_swe_episode, tmp_path, monkeypatch):
    mirror_dir = tmp_path / "mirrors"
    monkeypatch.setenv("EPISODIC_MIRROR_DIR", str(mirror_dir))

    oracle = gate.graded_score(local_swe_episode, wm._oracle_diff_runner(
        wm._unified_diff(local_swe_episode)))
    assert oracle["ok"] is True
    bare = list(mirror_dir.glob("*.git"))
    assert len(bare) == 1 and (bare[0] / "HEAD").exists()

    again = gate.graded_score(local_swe_episode, gate.empty_runner)
    assert again["ok"] is False
    assert list(mirror_dir.glob("*.git")) == bare
