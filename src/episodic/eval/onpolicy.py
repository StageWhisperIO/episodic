import difflib
import json
import os
import subprocess
import tempfile

from ..worldmodel.validate import _unified_diff
from . import editfmt, gate

_EDIT = editfmt._EDIT


def parse_edits(text):
    edits = []
    for match in _EDIT.finditer(text):
        edits.append({
            "path": match.group("path").strip(),
            "start": int(match.group("start")),
            "end": int(match.group("end")),
            "body": match.group("body"),
        })
    return edits


def render_edits(edits):
    return "\n\n".join(
        f"EDIT {edit['path']} {edit['start']}-{edit['end']}\n{edit['body']}\nENDEDIT"
        for edit in edits)


def fixed_attempt_runner(attempt_text, files):
    def runner(model, workspace, prompt_text):
        applied, log = editfmt.apply_numbered_edits(attempt_text, workspace, files)
        return log, 0 if applied else 1
    return runner


def _gold_file_contents(episode):
    files = editfmt._files_of(episode)
    root = episode["repo_state"]["root"]
    diff = _unified_diff(episode)
    with tempfile.TemporaryDirectory() as work:
        for path in files:
            target = os.path.join(work, path)
            parent = os.path.dirname(target)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(os.path.join(root, path)) as source_handle:
                content = source_handle.read()
            with open(target, "w") as target_handle:
                target_handle.write(content)
        subprocess.run(["git", "init", "-q", work], check=True)
        subprocess.run(["git", "-C", work, "apply", "--recount"],
                       input=diff, text=True, capture_output=True, check=True)
        return {path: open(os.path.join(work, path)).read() for path in files}


def gold_edits_for(episode):
    files = editfmt._files_of(episode)
    root = episode["repo_state"]["root"]
    corrected_by_file = _gold_file_contents(episode)
    edits = []
    for path in files:
        with open(os.path.join(root, path)) as handle:
            original_lines = handle.read().splitlines()
        corrected_lines = corrected_by_file[path].splitlines()
        matcher = difflib.SequenceMatcher(a=original_lines, b=corrected_lines, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            edits.append({"path": path, "start": i1 + 1, "end": i2,
                         "body": "\n".join(corrected_lines[j1:j2])})
    return edits


def minimal_correction(attempt_text, gold_edits, files):
    attempt_edits = [edit for edit in parse_edits(attempt_text) if edit["path"] in files]
    needed_locations = {(edit["path"], edit["start"], edit["end"]) for edit in gold_edits}
    kept = [edit for edit in attempt_edits
           if (edit["path"], edit["start"], edit["end"]) not in needed_locations]
    return render_edits(kept + list(gold_edits))


def _minimal_failing_regions(episode, attempt_text, gold_edits, files):
    needed = list(gold_edits)
    for gold_edit in gold_edits:
        if gold_edit not in needed:
            continue
        trial = [edit for edit in needed if edit != gold_edit]
        candidate = minimal_correction(attempt_text, trial, files)
        if gate.graded_score(episode, fixed_attempt_runner(candidate, files))["ok"]:
            needed = trial
    return needed


def localize_failure(episode, attempt_text):
    files = editfmt._files_of(episode)
    score = gate.graded_score(episode, fixed_attempt_runner(attempt_text, files))
    if score["ok"]:
        return {"passes": True, "score": score, "files": files,
                "gold_edits": None, "diverging_regions": []}
    gold_edits = gold_edits_for(episode)
    diverging = _minimal_failing_regions(episode, attempt_text, gold_edits, files)
    return {"passes": False, "score": score, "files": files,
            "gold_edits": gold_edits, "diverging_regions": diverging}


def gold_expert_fn(episode, attempt_text, feedback):
    feedback = feedback if feedback is not None else localize_failure(episode, attempt_text)
    files = editfmt._files_of(episode)
    needed = feedback.get("diverging_regions")
    if needed is None:
        needed = gold_edits_for(episode)
    return minimal_correction(attempt_text, needed, files)


def edit_distance(a_text, b_text):
    a_lines = a_text.splitlines()
    b_lines = b_text.splitlines()
    previous = list(range(len(b_lines) + 1))
    for i, a_line in enumerate(a_lines, start=1):
        current = [i] + [0] * len(b_lines)
        for j, b_line in enumerate(b_lines, start=1):
            if a_line == b_line:
                current[j] = previous[j - 1]
            else:
                current[j] = 1 + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    return previous[-1]


def build_correction_row(episode, prompt, corrected_text):
    return {"messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": corrected_text}],
            "meta": episode}


def _offpolicy_path(out_path):
    if out_path.endswith(".jsonl"):
        return out_path[:-len(".jsonl")] + ".offpolicy.jsonl"
    return out_path + ".offpolicy"


def build_correction_dataset(episodes, generate, expert_fn, out_path):
    off_path = _offpolicy_path(out_path)
    positive = corrected = skipped = 0
    gaps = []
    with open(out_path, "w") as onpolicy_file, open(off_path, "w") as offpolicy_file:
        for episode in episodes:
            diff = _unified_diff(episode)
            if not diff.strip():
                skipped += 1
                continue
            files = editfmt._files_of(episode)
            root = episode["repo_state"]["root"]
            prompt = editfmt.numbered_edit_prompt(episode["intent"], root, files)
            attempt_text = generate("policy", [{"role": "user", "content": prompt}])
            localized = localize_failure(episode, attempt_text)
            gold_edits = localized["gold_edits"] if localized["gold_edits"] is not None \
                else gold_edits_for(episode)
            gold_text = render_edits(gold_edits)
            if localized["passes"]:
                positive += 1
                target = attempt_text
            else:
                corrected += 1
                target = expert_fn(episode, attempt_text, localized)
            gold_distance = edit_distance(attempt_text, gold_text)
            corrected_distance = edit_distance(attempt_text, target)
            gaps.append(gold_distance - corrected_distance)
            onpolicy_file.write(json.dumps(build_correction_row(episode, prompt, target)) + "\n")
            offpolicy_file.write(json.dumps(build_correction_row(episode, prompt, gold_text)) + "\n")
    total = positive + corrected
    return {"total": total, "skipped": skipped, "positive": positive, "corrected": corrected,
            "mean_minimality_gap": (sum(gaps) / len(gaps)) if gaps else 0.0,
            "onpolicy_path": out_path, "offpolicy_path": off_path}
