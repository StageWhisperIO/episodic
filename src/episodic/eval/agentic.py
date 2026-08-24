import os
import re
import subprocess

from ..replay import modelrun

_TOOL_INSTRUCTION = (
    "\n\nYou are fixing the code in a git checkout so the failing tests pass. "
    "Work one step at a time. Emit exactly ONE action per reply, choosing from:\n"
    "  READ <path>        — show a source file (relative to the repo root)\n"
    "  LS <path>          — list a directory\n"
    "  TEST               — run the tests\n"
    "  PATCH              — then a fenced ```diff block with a unified diff; it is applied "
    "and the tests are run automatically\n"
    "Read the relevant files before patching. Reply with the action only."
)
_READ_CHARS = 8000


def _within(workspace, rel):
    root = os.path.normpath(str(workspace))
    target = os.path.normpath(os.path.join(root, rel))
    if target == root or target.startswith(root + os.sep):
        return target
    return None


def _list_dir(workspace, rel):
    target = _within(workspace, rel)
    if target is None or not os.path.isdir(target):
        return f"(not a directory: {rel})"
    return "\n".join(sorted(os.listdir(target))[:200])


def _read_file(workspace, rel):
    target = _within(workspace, rel)
    if target is None or not os.path.isfile(target):
        return f"(not a file: {rel})"
    with open(target, encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()
    body = "\n".join(f"{i + 1}\t{line}" for i, line in enumerate(lines))
    return body[:_READ_CHARS] + ("\n...(truncated)" if len(body) > _READ_CHARS else "")


def _parse_action(text):
    if "```" in text:
        diff = modelrun.extract_diff(text)
        if diff.strip() and diff != text:
            return "PATCH", None, diff
    stripped_text = text.lstrip()
    if stripped_text.startswith("diff --git") or stripped_text.startswith("--- "):
        return "PATCH", None, text
    for line in text.splitlines():
        stripped = line.strip()
        match = re.match(r"(READ|LS)\s+(\S+)", stripped, re.IGNORECASE)
        if match:
            return match.group(1).upper(), match.group(2).strip("`\"'"), None
        if re.match(r"TEST\b", stripped, re.IGNORECASE):
            return "TEST", None, None
    return None, None, None


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


def build_tool_agent(generate, test_command, max_steps=8, test_cwd=None):
    def runner(model, workspace, prompt_text):
        transcript = (prompt_text or "") + _TOOL_INSTRUCTION
        for step in range(max_steps):
            text = generate(model, [{"role": "user", "content": transcript}])
            kind, arg, diff = _parse_action(text)
            if kind == "READ":
                obs = _read_file(workspace, arg)
            elif kind == "LS":
                obs = _list_dir(workspace, arg)
            elif kind in ("PATCH", "TEST"):
                if kind == "PATCH":
                    applied, log = modelrun.apply_diff(diff, workspace)
                    if not applied:
                        obs = f"patch did not apply:\n{log[-500:]}"
                        transcript += f"\n\n[step {step + 1}] PATCH\n{obs}"
                        continue
                if not test_command:
                    return "tool-agent: patched, no test command", 0
                out, rc = _run_test(workspace, test_command, test_cwd)
                if rc == 0:
                    return f"tool-agent solved in {step + 1} step(s)", 0
                obs = f"tests still fail:\n{out[-800:]}"
            else:
                obs = "unrecognized action; use READ, LS, TEST, or PATCH + a ```diff block"
            transcript += f"\n\n[step {step + 1}] {kind or '?'} {arg or ''}\n{obs}"
        return "tool-agent: exhausted steps", 1

    return runner
