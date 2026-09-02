import os
import re

_LABELED = re.compile(r"###\s*FILE:\s*(?P<path>\S+)\s*\n+```[\w.+-]*\n(?P<body>.*?)```", re.DOTALL)
_ANY_BLOCK = re.compile(r"```[\w.+-]*\n(?P<body>.*?)```", re.DOTALL)
_EDIT = re.compile(r"EDIT\s+(?P<path>\S+)\s+(?P<start>\d+)\s*-\s*(?P<end>\d+)\s*\n(?P<body>.*?)\n?ENDEDIT",
                   re.DOTALL)


def number_lines(content):
    return "\n".join(f"{index + 1}\t{line}" for index, line in enumerate(content.splitlines()))


def _read(workspace, path):
    with open(os.path.join(workspace, path)) as handle:
        return handle.read()


def _write(workspace, path, content):
    if not content.endswith("\n"):
        content += "\n"
    with open(os.path.join(workspace, path), "w") as handle:
        handle.write(content)


def whole_file_prompt(task, workspace, files):
    parts = [task, "", "Fix the source so the failing tests pass. Current files:"]
    for path in files:
        parts.append(f"\n### FILE: {path}\n```python\n{_read(workspace, path)}```")
    parts.append("\nReturn every file you change COMPLETELY, each in this exact format:\n"
                 "### FILE: <path>\n```python\n<full corrected file>\n```")
    return "\n".join(parts)


def apply_whole_file(text, workspace, files):
    labeled = list(_LABELED.finditer(text))
    if labeled:
        applied = False
        for match in labeled:
            path = match.group("path").strip()
            if path in files:
                _write(workspace, path, match.group("body"))
                applied = True
        return applied, "wrote labeled files"
    blocks = list(_ANY_BLOCK.finditer(text))
    if len(files) == 1 and len(blocks) == 1:
        _write(workspace, files[0], blocks[0].group("body"))
        return True, "wrote single file"
    return False, "no usable file block"


def numbered_edit_prompt(task, workspace, files):
    parts = [task, "", "Fix the source so the failing tests pass. Files with line numbers:"]
    for path in files:
        parts.append(f"\n### FILE: {path}\n{number_lines(_read(workspace, path))}")
    parts.append("\nReply with one or more edits. Each replaces an INCLUSIVE 1-indexed line range with "
                 "new code (raw lines, no line-number prefixes), in EXACTLY this format:\n"
                 "EDIT <path> <start>-<end>\n<new lines>\nENDEDIT")
    return "\n".join(parts)


def apply_numbered_edits(text, workspace, files):
    edits = {}
    for match in _EDIT.finditer(text):
        path = match.group("path").strip()
        if path not in files:
            continue
        edits.setdefault(path, []).append(
            (int(match.group("start")), int(match.group("end")), match.group("body")))
    if not edits:
        return False, "no edits"
    for path, file_edits in edits.items():
        lines = _read(workspace, path).splitlines()
        for start, end, body in sorted(file_edits, key=lambda edit: edit[0], reverse=True):
            start = max(1, min(start, len(lines) + 1))
            end = max(start - 1, min(end, len(lines)))
            lines[start - 1:end] = body.split("\n")
        _write(workspace, path, "\n".join(lines))
    return True, "applied edits"


def _files_of(episode):
    return [diff["file"] for diff in episode.get("diffs", [])]


def build_edit_runner(generate, files, fmt="wholefile"):
    prompt_for = numbered_edit_prompt if fmt == "numbered" else whole_file_prompt
    apply = apply_numbered_edits if fmt == "numbered" else apply_whole_file

    def runner(model, workspace, prompt_text):
        prompt = prompt_for(prompt_text, workspace, files)
        text = generate(model, [{"role": "user", "content": prompt}])
        applied, log = apply(text, workspace, files)
        return log, 0 if applied else 1

    return runner
