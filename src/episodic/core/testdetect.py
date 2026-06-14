import re

TEST_COMMAND_PATTERNS = (
    ("pytest", re.compile(r"\bpytest\b|\bpy\.test\b")),
    ("python-unittest", re.compile(r"python[0-9.]*\s+-m\s+unittest")),
    ("jest", re.compile(r"\bjest\b")),
    ("vitest", re.compile(r"\bvitest\b")),
    ("mocha", re.compile(r"\bmocha\b")),
    ("go-test", re.compile(r"\bgo\s+test\b")),
    ("cargo-test", re.compile(r"\bcargo\s+test\b")),
    ("rspec", re.compile(r"\brspec\b")),
    ("phpunit", re.compile(r"\bphpunit\b")),
    ("gradle-test", re.compile(r"\bgradlew?\b.*\btest\b")),
    ("maven-test", re.compile(r"\bmvn\b.*\btest\b")),
    ("npm-test", re.compile(r"\b(npm|pnpm|yarn|bun)\s+(run\s+)?test\b")),
    ("make-test", re.compile(r"\bmake\s+test\b")),
)

_PARSERS = []


def _parser(func):
    _PARSERS.append(func)
    return func


@_parser
def _pytest(output):
    match = re.search(
        r"(?:(\d+) failed.*?)?(?:(\d+) passed)?(?:.*?(\d+) skipped)?",
        output,
    )
    passed = re.search(r"(\d+) passed", output)
    failed = re.search(r"(\d+) failed", output)
    skipped = re.search(r"(\d+) skipped", output)
    errors = re.search(r"(\d+) error", output)
    if not (passed or failed or errors):
        return None
    return {
        "passed": int(passed.group(1)) if passed else 0,
        "failed": (int(failed.group(1)) if failed else 0) + (int(errors.group(1)) if errors else 0),
        "skipped": int(skipped.group(1)) if skipped else 0,
    }


@_parser
def _jest_vitest(output):
    match = re.search(r"Tests:\s+(?:(\d+) failed,\s+)?(?:(\d+) skipped,\s+)?(\d+) passed", output)
    if not match:
        return None
    return {
        "failed": int(match.group(1) or 0),
        "skipped": int(match.group(2) or 0),
        "passed": int(match.group(3) or 0),
    }


@_parser
def _go(output):
    if not re.search(r"^(ok|FAIL|PASS)\b", output, re.MULTILINE):
        return None
    failed = len(re.findall(r"^--- FAIL", output, re.MULTILINE))
    passed = len(re.findall(r"^--- PASS", output, re.MULTILINE))
    if failed == 0 and passed == 0:
        passed = len(re.findall(r"^ok\s", output, re.MULTILINE))
        failed = len(re.findall(r"^FAIL\s", output, re.MULTILINE))
    return {"passed": passed, "failed": failed, "skipped": 0}


@_parser
def _generic(output):
    match = re.search(r"(\d+)\s+passing(?:.*?(\d+)\s+failing)?", output)
    if not match:
        return None
    return {
        "passed": int(match.group(1)),
        "failed": int(match.group(2) or 0),
        "skipped": 0,
    }


def classify_command(command):
    for framework, pattern in TEST_COMMAND_PATTERNS:
        if pattern.search(command):
            return framework
    return None


def parse_output(output):
    if not output:
        return None
    for parser in _PARSERS:
        result = parser(output)
        if result is not None:
            return result
    return None


def detect_test_run(command, output, ts, exit_code=None):
    framework = classify_command(command or "")
    if framework is None:
        return None
    counts = parse_output(output or "") or {"passed": 0, "failed": 0, "skipped": 0}
    total = counts["passed"] + counts["failed"] + counts["skipped"]
    if exit_code is not None and exit_code != 0:
        ok = False
    elif total > 0 and counts["failed"] == 0:
        ok = True
    else:
        ok = False
    return {
        "ts": ts,
        "framework": framework,
        "command": command,
        "passed": counts["passed"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "total": total,
        "ok": ok,
    }
