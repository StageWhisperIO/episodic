import os
import subprocess

from ..replay import modelrun


def _run_test(workspace, test_command, test_cwd):
    cwd = os.path.join(str(workspace), test_cwd) if test_cwd else str(workspace)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    try:
        result = subprocess.run(test_command, cwd=cwd, shell=True, capture_output=True,
                                text=True, timeout=120, env=env)
        return result.stdout + result.stderr, result.returncode
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc), -1


def _revert(workspace):
    subprocess.run(["git", "-C", str(workspace), "checkout", "--", "."], capture_output=True)
    subprocess.run(["git", "-C", str(workspace), "clean", "-fdq"], capture_output=True)


def build_agentic_runner(generate, test_command, max_turns=3, test_cwd=None):
    def runner(model, workspace, prompt_text):
        base_prompt = prompt_text or ""
        history = base_prompt
        last_diff = None
        for turn in range(max_turns):
            text = generate(model, [{"role": "user", "content": history + modelrun._DIFF_INSTRUCTION}])
            diff = modelrun.extract_diff(text)
            applied, log = modelrun.apply_diff(diff, workspace)
            if not applied:
                history = base_prompt + f"\n\nAttempt {turn + 1} did not apply as a unified diff:\n{log[-500:]}"
                continue
            last_diff = diff
            if not test_command:
                return log, 0
            out, rc = _run_test(workspace, test_command, test_cwd)
            if rc == 0:
                return f"agentic solved in {turn + 1} turn(s)", 0
            _revert(workspace)
            history = (base_prompt + f"\n\nAttempt {turn + 1} still fails the tests:\n```diff\n{diff}\n```\n"
                       f"Test output:\n{out[-800:]}\n\nProduce a corrected unified diff.")
        if last_diff:
            modelrun.apply_diff(last_diff, workspace)
        return "agentic: exhausted turns", 1

    return runner
