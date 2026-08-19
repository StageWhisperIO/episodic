import json
import subprocess

import pytest

from episodic.replay import modelrun


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _scratch_repo(tmp_path):
    repo = tmp_path / "scratch_repo"
    repo.mkdir()
    (repo / "mod.py").write_text("x = 1\n")
    _git(str(repo), "init", "-q")
    _git(str(repo), "config", "user.email", "t@t.dev")
    _git(str(repo), "config", "user.name", "t")
    _git(str(repo), "add", "-A")
    _git(str(repo), "commit", "-q", "-m", "base")
    return repo


def test_extract_diff_from_diff_fence():
    text = "Here is my change:\n```diff\ndiff --git a/mod.py b/mod.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n```\nDone."
    diff = modelrun.extract_diff(text)
    assert diff.startswith("diff --git a/mod.py b/mod.py")
    assert "x = 2" in diff


def test_extract_diff_from_patch_fence():
    text = "```patch\ndiff --git a/f b/f\n@@ -1 +1 @@\n-a\n+b\n```"
    diff = modelrun.extract_diff(text)
    assert diff.startswith("diff --git a/f b/f")


def test_extract_diff_raw_diff_without_fence():
    text = "diff --git a/f b/f\n@@ -1 +1 @@\n-a\n+b\n"
    assert modelrun.extract_diff(text) == text


def test_extract_diff_no_diff_present_returns_original_text():
    text = "I made no changes."
    assert modelrun.extract_diff(text) == text


def test_extract_diff_ignores_non_diff_fenced_blocks():
    text = "```python\nprint('hi')\n```"
    assert modelrun.extract_diff(text) == text


def test_extract_diff_handles_empty_and_none():
    assert modelrun.extract_diff("") == ""
    assert modelrun.extract_diff(None) == ""


def test_apply_diff_succeeds_against_a_scratch_repo(tmp_path):
    repo = _scratch_repo(tmp_path)
    diff = (
        "diff --git a/mod.py b/mod.py\n"
        "index 1191247..b326b6b 100644\n"
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
    )

    ok, output = modelrun.apply_diff(diff, repo)

    assert ok is True
    assert (repo / "mod.py").read_text() == "x = 2\n"


def test_apply_diff_fails_on_malformed_patch(tmp_path):
    repo = _scratch_repo(tmp_path)

    ok, output = modelrun.apply_diff("not a real diff at all", repo)

    assert ok is False
    assert output


def test_apply_diff_empty_diff_is_a_clean_failure(tmp_path):
    repo = _scratch_repo(tmp_path)

    ok, output = modelrun.apply_diff("", repo)

    assert ok is False
    assert "empty" in output


def test_resolve_generate_stub_default_returns_empty_string():
    generate = modelrun.resolve_generate({})
    assert generate("candidate", [{"role": "user", "content": "hi"}]) == ""


def test_resolve_generate_stub_dict_keyed_by_model():
    config = {"eval_backend": "stub", "eval_stub": {"candidate": "diff text for candidate"}}
    generate = modelrun.resolve_generate(config)
    assert generate("candidate", []) == "diff text for candidate"
    assert generate("base", []) == ""


def test_resolve_generate_stub_callable():
    calls = []

    def stub(model, messages):
        calls.append((model, messages))
        return f"diff for {model}"

    generate = modelrun.resolve_generate({"eval_backend": "stub", "eval_stub": stub})
    assert generate("candidate", [{"role": "user", "content": "x"}]) == "diff for candidate"
    assert calls == [("candidate", [{"role": "user", "content": "x"}])]


def test_resolve_generate_rejects_unknown_backend():
    with pytest.raises(ValueError):
        modelrun.resolve_generate({"eval_backend": "not-a-real-backend"})


def _fake_opener(json_body):
    class _Response:
        def read(self):
            return json.dumps(json_body).encode("utf-8")

    def opener(request, timeout=None):
        return _Response()

    return opener


def test_resolve_generate_serving_backend_drives_a_fake_chat_completions_call():
    upstream_body = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "diff applied"}}],
    }
    config = {
        "eval_backend": "serving",
        "eval_backend_config": {
            "backend": "openai", "base_url": "http://distilled.local", "opener": _fake_opener(upstream_body),
        },
    }
    generate = modelrun.resolve_generate(config)

    text = generate("candidate-model", [{"role": "user", "content": "fix it"}])

    assert text == "diff applied"


def test_resolve_generate_serving_backend_no_choices_returns_empty_string():
    config = {
        "eval_backend": "serving",
        "eval_backend_config": {
            "backend": "openai", "base_url": "http://distilled.local", "opener": _fake_opener({"choices": []}),
        },
    }
    generate = modelrun.resolve_generate(config)
    assert generate("candidate-model", []) == ""


def test_resolve_generate_serving_backend_refuses_unconfigured_public_openai():
    from episodic import serving

    with pytest.raises(serving.BackendUnavailable):
        modelrun.resolve_generate({"eval_backend": "serving"})

    with pytest.raises(serving.BackendUnavailable):
        modelrun.resolve_generate({"eval_backend": "serving", "eval_backend_config": {"backend": "openai"}})


def test_resolve_generate_serving_backend_allows_openai_with_api_key():
    config = {
        "eval_backend": "serving",
        "eval_backend_config": {
            "backend": "openai", "api_key": "sk-test", "opener": _fake_opener({"choices": []}),
        },
    }
    assert modelrun.resolve_generate(config)("m", []) == ""


def test_resolve_generate_mlx_requires_eval_model_dir():
    with pytest.raises(ValueError):
        modelrun.resolve_generate({"eval_backend": "mlx"})


def test_resolve_generate_tinker_requires_eval_sampler_path():
    with pytest.raises(ValueError):
        modelrun.resolve_generate({"eval_backend": "tinker"})


def test_build_runner_applies_the_generated_diff_and_reports_success(tmp_path):
    repo = _scratch_repo(tmp_path)
    diff_text = (
        "```diff\n"
        "diff --git a/mod.py b/mod.py\n"
        "index 1191247..b326b6b 100644\n"
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
        "```\n"
    )
    calls = []

    def generate(model, messages):
        calls.append((model, messages))
        return diff_text

    runner = modelrun.build_runner(generate)

    output, rc = runner("candidate-model", repo, "fix mod.py")

    assert rc == 0
    assert (repo / "mod.py").read_text() == "x = 2\n"
    assert calls[0][0] == "candidate-model"
    assert "diff" in calls[0][1][0]["content"].lower()


def test_build_runner_reports_failure_when_no_diff_is_produced(tmp_path):
    repo = _scratch_repo(tmp_path)

    runner = modelrun.build_runner(lambda model, messages: "I don't know how to fix this.")

    output, rc = runner("candidate-model", repo, "fix mod.py")

    assert rc == 1
    assert (repo / "mod.py").read_text() == "x = 1\n"
