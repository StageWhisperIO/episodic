import re

SOURCE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb",
    ".php", ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".swift", ".kt",
    ".scala", ".sh", ".m", ".mm",
}

TEST_PATH_MARKERS = ("test", "spec", "__tests__", ".test.", ".spec.", "_test.")

RISKY_PATTERNS = (
    (re.compile(r"(auth|login|password|secret|token|credential|oauth|jwt)", re.I), "touches authentication or secrets"),
    (re.compile(r"(^|/)\.env"), "modifies environment / secret config"),
    (re.compile(r"(migration|migrate)", re.I), "changes database migrations"),
    (re.compile(r"(dockerfile|docker-compose)", re.I), "changes container build"),
    (re.compile(r"\.github/workflows|\.gitlab-ci|circleci|jenkinsfile", re.I), "changes CI configuration"),
    (re.compile(r"(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|cargo\.lock|go\.sum)", re.I), "changes dependency lockfile"),
    (re.compile(r"\.(tf|tfvars)$|kubernetes|helm|k8s", re.I), "changes infrastructure definitions"),
)

TODO_PATTERN = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b[:\s](.*)", re.I)
LARGE_DIFF_THRESHOLD = 250


def _ext(path):
    dot = path.rfind(".")
    return path[dot:].lower() if dot != -1 else ""


def is_test_file(path):
    lowered = path.lower()
    return any(marker in lowered for marker in TEST_PATH_MARKERS)


def is_source_file(path):
    return _ext(path) in SOURCE_EXTENSIONS and not is_test_file(path)


def _what_changed(diffs):
    lines = []
    for entry in diffs:
        churn = f"+{entry['additions']}/-{entry['deletions']}" if entry["unified"] else ""
        lines.append(f"{entry['status']}: {entry['file']} {churn}".strip())
    return lines


def _risky_edits(diffs):
    flagged = []
    for entry in diffs:
        reasons = []
        for pattern, label in RISKY_PATTERNS:
            if pattern.search(entry["file"]):
                reasons.append(label)
        if entry["status"] == "deleted":
            reasons.append("deletes a file")
        if entry["additions"] + entry["deletions"] >= LARGE_DIFF_THRESHOLD:
            reasons.append(f"large change ({entry['additions'] + entry['deletions']} lines)")
        if reasons:
            flagged.append(f"{entry['file']} — {', '.join(reasons)}")
    return flagged


def _tests_run(tests):
    lines = []
    for test in tests:
        status = "ok" if test["ok"] else "FAILED"
        lines.append(
            f"{test['framework']}: {test['passed']} passed / {test['failed']} failed [{status}] ({test['command']})"
        )
    return lines


def _tests_missing(episode):
    diffs = episode["diffs"]
    edited_sources = [entry["file"] for entry in diffs if is_source_file(entry["file"]) and entry["status"] != "deleted"]
    touched_tests = [entry["file"] for entry in diffs if is_test_file(entry["file"])]
    missing = []
    if edited_sources and not touched_tests:
        preview = ", ".join(edited_sources[:5])
        suffix = "..." if len(edited_sources) > 5 else ""
        missing.append(f"No test files changed for {len(edited_sources)} edited source file(s): {preview}{suffix}")
    if edited_sources and not episode["tests"]:
        missing.append("No tests were executed during this session.")
    failing = [test for test in episode["tests"] if not test["ok"]]
    if failing:
        missing.append(f"{len(failing)} test command(s) reported failures.")
    return missing


def _added_todos(diffs):
    todos = []
    for entry in diffs:
        if not entry["unified"]:
            continue
        for line in entry["unified"].splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                match = TODO_PATTERN.search(line[1:])
                if match:
                    todos.append(f"{entry['file']}: {match.group(1).upper()} {match.group(2).strip()}".strip())
    return todos


def _pr_title(intent, diffs):
    first = ""
    for line in (intent or "").splitlines():
        if line.strip():
            first = line.strip()
            break
    first = re.sub(r"^(please|can you|could you|i want to|i need to|help me)\s+", "", first, flags=re.I)
    if not first and diffs:
        first = f"Update {diffs[0]['file']}"
    title = first[:72] if first else "Coding session changes"
    return title[0].upper() + title[1:] if title else title


def _follow_ups(episode, tests_missing, risky, todos):
    items = list(todos)
    for line in tests_missing:
        if line.startswith("No test"):
            items.append("Add tests covering the changed source files.")
            break
    if any(not test["ok"] for test in episode["tests"]):
        items.append("Fix the failing tests before merging.")
    if risky:
        items.append("Have a reviewer check the risky edits flagged above.")
    seen = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


def summarize(episode):
    diffs = episode["diffs"]
    what_changed = _what_changed(diffs)
    risky = _risky_edits(diffs)
    tests_missing = _tests_missing(episode)
    todos = _added_todos(diffs)
    follow_ups = _follow_ups(episode, tests_missing, risky, todos)
    files_touched = [entry["file"] for entry in diffs]
    title = _pr_title(episode["intent"], diffs)
    description = _render_pr_description(episode, title, what_changed, episode["tests"], risky, follow_ups)
    return {
        "episode_id": episode["id"],
        "intent": episode["intent"],
        "what_changed": what_changed,
        "why_changed": (episode["intent"] or "No recorded intent.").strip(),
        "files_touched": files_touched,
        "tests_run": _tests_run(episode["tests"]),
        "tests_missing": tests_missing,
        "risky_edits": risky,
        "suggested_pr_title": title,
        "suggested_pr_description": description,
        "follow_up_todos": follow_ups,
    }


def _bullets(items, empty="_None._"):
    if not items:
        return empty
    return "\n".join(f"- {item}" for item in items)


def _render_pr_description(episode, title, what_changed, tests, risky, follow_ups):
    test_lines = _tests_run(tests) or ["No tests were run."]
    checklist = "\n".join(f"- [ ] {item}" for item in follow_ups) or "- [ ] _None._"
    return (
        f"## Summary\n{episode['intent'] or 'Changes produced during a coding session.'}\n\n"
        f"## Changes\n{_bullets(what_changed)}\n\n"
        f"## Tests\n{_bullets(test_lines)}\n\n"
        f"## Risks\n{_bullets(risky)}\n\n"
        f"## Follow-ups\n{checklist}\n"
    )


def render_markdown(report):
    sections = [
        f"# Session summary — {report['episode_id']}",
        f"\n**Intent:** {report['intent'] or '(none recorded)'}",
        f"\n## What changed\n{_bullets(report['what_changed'])}",
        f"\n## Why it changed\n{report['why_changed']}",
        f"\n## Files touched\n{_bullets(report['files_touched'])}",
        f"\n## Tests run\n{_bullets(report['tests_run'], '_No tests detected._')}",
        f"\n## Tests missing\n{_bullets(report['tests_missing'], '_Nothing flagged._')}",
        f"\n## Risky edits\n{_bullets(report['risky_edits'], '_None detected._')}",
        f"\n## Suggested PR title\n{report['suggested_pr_title']}",
        f"\n## Suggested PR description\n{report['suggested_pr_description']}",
        f"\n## Follow-up TODOs\n{_bullets(report['follow_up_todos'], '_None._')}",
    ]
    return "\n".join(sections)
