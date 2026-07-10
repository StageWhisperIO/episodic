import json
import os
import subprocess
from collections import Counter

from ..schema import FEEDBACK_LABELS, now_iso

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_LABELER_CMD = f"claude -p --model {DEFAULT_MODEL}"
OUTCOME_SUCCESS = ("yes", "partial", "no", "unclear")
RENDER_BUDGET = 6000
PROMPT_CLIP = 400
EVIDENCE_CLIP = 300


def _clamp01(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, number))


def _prompt_text(step):
    return (step.get("input") or {}).get("prompt") or step.get("intent") or ""


def _segments(episode):
    segments = []
    current = None
    for step in episode.get("steps", []):
        if step["type"] == "user_prompt":
            current = {"index": step.get("index"), "prompt": _prompt_text(step), "actions": []}
            segments.append(current)
        elif current is not None:
            current["actions"].append(step)
    return segments


def _summarize(actions):
    counts = Counter(action["type"] for action in actions)
    edits = counts.get("file_edit", 0) + counts.get("file_write", 0) + counts.get("file_delete", 0)
    parts = []
    if edits:
        parts.append(f"{edits} edit(s)")
    if counts.get("shell_command"):
        parts.append(f"{counts['shell_command']} command(s)")
    if counts.get("file_read"):
        parts.append(f"{counts['file_read']} read(s)")
    if counts.get("denial"):
        parts.append(f"{counts['denial']} denial(s)")
    return ", ".join(parts)


def render(episode, budget=RENDER_BUDGET):
    lines = []
    for segment in _segments(episode):
        prompt = " ".join((segment["prompt"] or "").split())[:PROMPT_CLIP]
        lines.append(f"[#{segment['index']}] USER: {prompt}")
        summary = _summarize(segment["actions"])
        if summary:
            lines.append(f"    agent: {summary}")
    return "\n".join(lines)[-budget:]


def build_prompt(episode):
    return (
        "You are grading a coding-agent session transcript to extract the user's feedback signal.\n"
        "Below are the user's turns (with step index) and a summary of what the agent did before each.\n"
        "For every user turn that REACTS to the agent's preceding work, classify it. Ignore turns that "
        "only start a new task or ask an unrelated question.\n"
        f"Allowed labels: {', '.join(FEEDBACK_LABELS)}.\n"
        "Guidance: corrections / 'still broken' / 'that is wrong' -> wrong; the agent had to be told or "
        "rescued -> needed_human_rescue; 'too much / too broad' -> too_broad; 'too slow' -> too_slow; "
        "approvals like 'yes / lgtm / ship it / push / commit' -> accepted_as_is; approval right after "
        "requesting a change -> accepted_after_edits; a genuinely useful result that is praised -> useful.\n"
        "Also judge the OVERALL outcome: success is one of yes|partial|no|unclear.\n"
        "Reply with ONLY a JSON object, no prose:\n"
        '{"feedback":[{"step_index":<int>,"label":"<label>","confidence":<0..1>,"evidence":"<short quote>"}],'
        '"outcome":{"success":"yes|partial|no|unclear","confidence":<0..1>,"rationale":"<short>"}}\n\n'
        f"TRANSCRIPT:\n{render(episode)}"
    )


def _extract_json(text):
    if not text:
        return None
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:index + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None


def mine(episode, generate):
    data = _extract_json(generate(build_prompt(episode))) or {}
    ts_by_index = {step.get("index"): step.get("ts") for step in episode.get("steps", [])}
    fallback_ts = episode.get("created_at") or now_iso()
    env_model = os.environ.get("EPISODIC_LABELER_MODEL")
    model = env_model or getattr(generate, "command", None) or DEFAULT_MODEL

    feedback = []
    for item in data.get("feedback") or []:
        label = item.get("label")
        if label not in FEEDBACK_LABELS:
            continue
        index = item.get("step_index")
        feedback.append({
            "ts": ts_by_index.get(index) or fallback_ts,
            "label": label,
            "note": (item.get("evidence") or "")[:EVIDENCE_CLIP],
            "source": "mined",
            "confidence": _clamp01(item.get("confidence")),
            "evidence_step_index": index if isinstance(index, int) else None,
            "model": model,
        })

    outcome_hint = None
    outcome = data.get("outcome")
    if isinstance(outcome, dict) and outcome.get("success") in OUTCOME_SUCCESS:
        outcome_hint = {
            "success": outcome["success"],
            "confidence": _clamp01(outcome.get("confidence")),
            "rationale": (outcome.get("rationale") or "")[:EVIDENCE_CLIP],
            "source": "mined",
            "model": model,
        }
    return {"feedback": feedback, "outcome_hint": outcome_hint}


def _resolve_command(command):
    return command or os.environ.get("EPISODIC_LABELER_CMD") or DEFAULT_LABELER_CMD


def _run_labeler(resolved, prompt, timeout):
    env = dict(os.environ, EPISODIC_DISABLE="1")
    return subprocess.run(
        resolved, shell=True, input=prompt, text=True,
        capture_output=True, timeout=timeout, env=env,
    )


def command_generate(command=None, timeout=120):
    resolved = _resolve_command(command)

    def generate(prompt):
        proc = _run_labeler(resolved, prompt, timeout)
        if proc.returncode != 0:
            snippet = (proc.stderr or proc.stdout or "").strip()[:200]
            raise RuntimeError(
                f"labeler command {resolved!r} exited {proc.returncode}: {snippet}"
            )
        return proc.stdout

    generate.command = resolved
    return generate


def probe(prompt, command=None, timeout=120):
    resolved = _resolve_command(command)
    proc = _run_labeler(resolved, prompt, timeout)
    return {"command": resolved, "stdout": proc.stdout,
            "stderr": proc.stderr, "code": proc.returncode}
