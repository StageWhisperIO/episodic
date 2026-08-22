import hashlib
import os
import subprocess

from .. import store
from ..core import diffparse
from ..schema import new_episode

_TS = "2026-08-20T10:00:00+00:00"


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _pytest(repo):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(["python", "-m", "pytest", "-q"], cwd=repo, capture_output=True,
                          text=True, env=env)


def _write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def _salt(*parts):
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:6]


def _test_file(body):
    return f"from solution import solve\n\n\ndef test_solve():\n    {body}\n"


def _raises_file(call):
    return ("import pytest\nfrom solution import solve\n\n\n"
            f"def test_solve():\n    with pytest.raises(ValueError):\n        {call}\n")


def task_specs(variant):
    n = variant
    a, b, k = 2 + n, 3 + n, 3 + n
    xs = list(range(1, 5 + n))
    dedup = [3 + n, 1, 1, 2]
    pair = ["a", chr(98 + (n % 20))]
    num, coef = 100 + n, 2 + n
    truthy = 1 + n
    word = f"hi{n}"
    return [
        ("operator",
         "def solve(a, b):\n    return a - b\n",
         "def solve(a, b):\n    return a + b\n",
         _test_file(f"assert solve({a}, {b}) == {a + b}")),
        ("operator",
         "def solve(a, b):\n    return a + b\n",
         "def solve(a, b):\n    return a * b\n",
         _test_file(f"assert solve({a}, {b}) == {a * b}")),
        ("bound",
         "def solve(n):\n    return list(range(n - 1))\n",
         "def solve(n):\n    return list(range(n))\n",
         _test_file(f"assert solve({k}) == {list(range(k))}")),
        ("bound",
         "def solve(xs, k):\n    return xs[:k - 1]\n",
         "def solve(xs, k):\n    return xs[:k]\n",
         _test_file(f"assert solve({xs}, 2) == {xs[:2]}")),
        ("ds",
         "def solve(xs):\n    return sorted(xs)\n",
         "def solve(xs):\n    return sorted(set(xs))\n",
         _test_file(f"assert solve({dedup}) == {sorted(set(dedup))}")),
        ("ds",
         "def solve(xs):\n    return {i: v for i, v in enumerate(xs, 1)}\n",
         "def solve(xs):\n    return {i: v for i, v in enumerate(xs)}\n",
         _test_file(f"assert solve({pair}) == {{0: {pair[0]!r}, 1: {pair[1]!r}}}")),
        ("exc",
         f"def solve(n):\n    return {num} // n\n",
         f"def solve(n):\n    if n == 0:\n        raise ValueError('zero')\n    return {num} // n\n",
         _raises_file("solve(0)")),
        ("exc",
         f"def solve(n):\n    return {coef} * n\n",
         f"def solve(n):\n    if n < 0:\n        raise ValueError('negative')\n    return {coef} * n\n",
         _raises_file("solve(-1)")),
        ("logic",
         "def solve(a, b):\n    return a or b\n",
         "def solve(a, b):\n    return a and b\n",
         _test_file(f"assert solve({truthy}, 0) == 0")),
        ("logic",
         "def solve(x):\n    return x\n",
         "def solve(x):\n    return not x\n",
         _test_file(f"assert solve({truthy}) is False")),
        ("str",
         "def solve(s):\n    return s\n",
         "def solve(s):\n    return s.strip()\n",
         _test_file(f"assert solve({'  ' + word + ' '!r}) == {word!r}")),
        ("str",
         "def solve(s):\n    return s.lower()\n",
         "def solve(s):\n    return s.upper()\n",
         _test_file(f"assert solve({word!r}) == {word.upper()!r}")),
    ]


def build_task(repos_dir, task_id, buggy_src, fixed_src, test_src, bug_class="misc"):
    repo = os.path.join(repos_dir, task_id)
    subprocess.run(["rm", "-rf", repo], check=True)
    os.makedirs(repo)
    _write(os.path.join(repo, "solution.py"), buggy_src)
    _write(os.path.join(repo, "test_solution.py"), test_src)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "eval@episodic.dev")
    _git(repo, "config", "user.name", "episodic-eval")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "buggy base (red)")
    base_commit = _git(repo, "rev-parse", "HEAD").stdout.strip()

    red = _pytest(repo)
    if red.returncode == 0:
        raise AssertionError(f"{task_id}: base is not RED (test already passes)")
    _write(os.path.join(repo, "solution.py"), fixed_src)
    green = _pytest(repo)
    if green.returncode != 0:
        raise AssertionError(f"{task_id}: fix is not GREEN\n{green.stdout[-400:]}")
    gold_diff = _git(repo, "diff").stdout
    _git(repo, "checkout", "--", "solution.py")

    parsed = diffparse.parse_unified_diff(gold_diff)
    unified = parsed[0]["unified"] if parsed else None
    additions = sum(p["additions"] for p in parsed)
    deletions = sum(p["deletions"] for p in parsed)

    intent = (f"The test in test_solution.py fails against solution.py:\n\n"
              f"```python\n{buggy_src}```\n\nTest file:\n\n```python\n{test_src}```\n\n"
              f"Fix solution.py so the test passes.")
    ep = new_episode(id=task_id, intent=intent)
    ep["repo_state"].update({"root": repo, "remote_url": None, "base_commit": base_commit, "branch": "main"})
    ep["labels"] = ["swe", bug_class]
    ep["steps"] = [{"index": 0, "ts": _TS, "type": "file_edit", "tool": "Edit",
                    "intent": "fix solution.py", "input": {"file_path": "solution.py"},
                    "observation": "applied", "approved": True, "cwd": repo, "duration_ms": None}]
    ep["diffs"] = [{"file": "solution.py", "status": "modified",
                    "additions": additions, "deletions": deletions, "unified": unified}]
    ep["commands"] = [{"ts": _TS, "command": "pytest -q", "cwd": repo,
                       "exit_code": 0, "output_excerpt": "1 passed", "is_test": True}]
    ep["tests"] = [{"ts": _TS, "framework": "pytest", "command": "pytest -q",
                    "passed": 1, "failed": 0, "skipped": 0, "total": 1, "ok": True}]
    ep["outcome"]["status"] = "merged"
    ep["outcome"]["merged"] = True
    return ep


def generate_corpus(repos_dir, variants=1, save=True):
    os.makedirs(repos_dir, exist_ok=True)
    episodes = []
    for variant in range(variants):
        for index, (bug_class, buggy, fixed, test) in enumerate(task_specs(variant)):
            task_id = f"ep_rg_{bug_class}_{variant:02d}_{index:02d}_{_salt(bug_class, variant, index)}"
            episode = build_task(repos_dir, task_id, buggy, fixed, test, bug_class=bug_class)
            if save:
                store.save_episode(episode)
            episodes.append(episode)
    return episodes
