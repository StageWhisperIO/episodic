import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from episodic import paths
from episodic.schema import now_iso
from episodic.core import testdetect, normalize, diffparse
from episodic.exporters import _captured_verifier

_MIN_REPLAY_FREE_BYTES = 2 * 1024 ** 3


def replay_id_for(episode):
    raw = episode["id"]
    suffix = raw.removeprefix("ep_")
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", suffix)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return "rp_" + (safe or "unknown") + "_" + digest


_VERIFIER_PATTERNS = (
    re.compile(r"(^|/)test_[^/]*\.py$"),
    re.compile(r"(^|/)[^/]*_test\.py$"),
    re.compile(r"(^|/)conftest\.py$"),
    re.compile(r"(^|/)[^/]*\.(test|spec)\.[jt]sx?$"),
    re.compile(r"(^|/)[^/]*_test\.go$"),
    re.compile(r"(^|/)[^/]*_spec\.rb$"),
    re.compile(r"(^|/)(tests?|spec|specs|__tests__)/"),
)


def _is_verifier_file(path):
    return any(pattern.search(path) for pattern in _VERIFIER_PATTERNS)


def _protect_verifier(workspace):
    diff_out, _ = _run_cmd(["git", "-C", str(workspace), "diff", "--name-only"], timeout=30)
    reverted = [p for p in diff_out.splitlines() if p.strip() and _is_verifier_file(p)]
    for path in reverted:
        _run_cmd(["git", "-C", str(workspace), "checkout", "HEAD", "--", path], timeout=30)
    untracked, _ = _run_cmd(
        ["git", "-C", str(workspace), "ls-files", "--others", "--exclude-standard"], timeout=30)
    for path in untracked.splitlines():
        if path.strip() and _is_verifier_file(path):
            target = (Path(workspace) / path).resolve()
            try:
                if os.path.commonpath([str(target), str(Path(workspace).resolve())]) == str(Path(workspace).resolve()):
                    target.unlink()
                    reverted.append(path)
            except (OSError, ValueError):
                pass
    return reverted


def cleanup_replay(replay_id, start=None):
    replays_root = paths.replays_dir(start).resolve()
    replay_dir = (replays_root / replay_id).resolve()
    try:
        if os.path.commonpath([str(replay_dir), str(replays_root)]) != str(replays_root):
            return False
    except ValueError:
        return False
    if replay_dir == replays_root or replay_dir.is_symlink() or not replay_dir.exists():
        return False
    shutil.rmtree(replay_dir, ignore_errors=True)
    return not replay_dir.exists()


def infer_test_command(repo_root, episode):
    for cmd in episode.get("commands", []):
        if cmd.get("is_test"):
            return cmd["command"]
    root = Path(repo_root)
    if (root / "pyproject.toml").exists() or (root / "pytest.ini").exists() or (root / "setup.cfg").exists():
        return "pytest -q"
    if (root / "package.json").exists():
        return "npm test"
    if (root / "go.mod").exists():
        return "go test ./..."
    if (root / "Cargo.toml").exists():
        return "cargo test"
    return None


def collect_lockfiles(repo_root):
    names = [
        "requirements.txt",
        "poetry.lock",
        "Pipfile.lock",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "Cargo.lock",
        "go.sum",
    ]
    result = []
    root = Path(repo_root)
    for name in names:
        p = root / name
        if p.exists():
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            result.append({"path": name, "sha256": digest})
    return result


def _relative_subdir(cwd, repo_root):
    if not cwd or not repo_root:
        return None
    relative = normalize.relativize_command(cwd, repo_root)
    if relative in (cwd, "."):
        return None
    return relative[2:] if relative.startswith("./") else relative


def _matching_command_cwd(command, episode):
    for cmd in episode.get("commands", []):
        if cmd.get("command") == command:
            return cmd.get("cwd")
    return None


def _needs_explicit_cwd(command, subdir):
    return bool(subdir) and subdir not in command


def _resolved_cwd(command, cwd, repo_root):
    subdir = _relative_subdir(cwd, repo_root)
    return subdir if _needs_explicit_cwd(command, subdir) else None


def _resolve_test_command(episode, repo_root):
    verifier = _captured_verifier(episode)
    if verifier is not None:
        command = normalize.relativize_command(verifier["command"], repo_root)
        test_cwd = _resolved_cwd(command, _matching_command_cwd(verifier["command"], episode), repo_root)
        return command, test_cwd

    test_command = infer_test_command(repo_root or "", episode) if repo_root else None
    if test_command is not None:
        return test_command, None

    for cmd in episode.get("commands", []):
        if cmd.get("is_test"):
            command = normalize.relativize_command(cmd["command"], repo_root)
            return command, _resolved_cwd(command, cmd.get("cwd"), repo_root)

    return None, None


def create_replay(episode, start=None):
    replay_id = replay_id_for(episode)
    repo_state = episode.get("repo_state", {})
    repo_root = repo_state.get("root")

    test_command, test_cwd = _resolve_test_command(episode, repo_root)

    lockfiles = collect_lockfiles(repo_root) if repo_root and Path(repo_root).exists() else []

    diffs = episode.get("diffs", [])
    files_changed = [d["file"] for d in diffs]
    total_additions = sum(d.get("additions", 0) for d in diffs)
    total_deletions = sum(d.get("deletions", 0) for d in diffs)

    reward = episode.get("reward_vector", {})
    reward_weights = {k: v for k, v in reward.items() if k not in ("composite", "components") and isinstance(v, (int, float))}

    replay_dir = paths.replays_dir(start) / replay_id
    replay_dir.mkdir(parents=True, exist_ok=True)

    diff_path = str(replay_dir / "expected.diff")

    manifest = {
        "replay_id": replay_id,
        "episode_id": episode["id"],
        "created_at": now_iso(),
        "base_commit": repo_state.get("base_commit"),
        "remote_url": repo_state.get("remote_url"),
        "repo": repo_state.get("repo"),
        "repo_root": repo_root,
        "branch": repo_state.get("branch"),
        "initial_prompt": episode.get("intent", ""),
        "test_command": test_command,
        "test_cwd": test_cwd,
        "lockfiles": lockfiles,
        "expected_outcome": {
            "files_changed": files_changed,
            "additions": total_additions,
            "deletions": total_deletions,
            "diff_path": diff_path,
        },
        "scoring_rules": {
            "tests_pass_weight": 0.6,
            "diff_overlap_weight": 0.4,
            "reward_weights": reward_weights,
        },
    }

    (replay_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (replay_dir / "prompt.txt").write_text(episode.get("intent", ""))

    unified_diffs = diffparse.join_unified(d.get("unified") for d in diffs)
    (replay_dir / "expected.diff").write_text(unified_diffs)

    return manifest


def _run_cmd(args, cwd=None, timeout=60, shell=False):
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=shell,
        )
        return result.stdout + result.stderr, result.returncode
    except Exception:
        return "", -1


def _within(path, root):
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    try:
        return os.path.commonpath([str(resolved), str(root)]) == str(root)
    except ValueError:
        return False


def _reset_workspace(workspace, replays_root):
    if workspace.is_symlink():
        return False, "workspace is a symlink; refusing to operate"
    if workspace.exists():
        if not _within(workspace, replays_root):
            return False, "workspace resolves outside replays root; refusing to delete"
        shutil.rmtree(workspace, ignore_errors=True)
    return True, None


def _init_git_baseline(workspace):
    _run_cmd(["git", "-C", str(workspace), "init", "-q"], timeout=30)
    _run_cmd(["git", "-C", str(workspace), "add", "-A"], timeout=30)
    _run_cmd(["git", "-C", str(workspace),
              "-c", "user.email=replay@episodic.local", "-c", "user.name=episodic",
              "commit", "-q", "-m", "replay base"], timeout=30)


_NOISE_DIRS = ("__pycache__/", ".git/", ".pytest_cache/", "node_modules/", ".mypy_cache/",
               ".ruff_cache/", ".tox/", ".venv/")
_NOISE_SUFFIXES = (".pyc", ".pyo", ".pyd", ".class", ".o")


def _is_noise(path):
    if any(segment in path for segment in _NOISE_DIRS):
        return True
    return path.endswith(_NOISE_SUFFIXES)


def _jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def run_replay(replay_id, model, start=None, runner_cmd=None, execute=False, runner=None):
    replays_root = paths.replays_dir(start).resolve()
    replay_dir = (replays_root / replay_id).resolve()
    if os.path.commonpath([str(replay_dir), str(replays_root)]) != str(replays_root):
        return {"error": f"replay id escapes replays root: {replay_id!r}"}
    manifest_path = replay_dir / "manifest.json"

    if not manifest_path.exists():
        return {"error": f"manifest not found: {manifest_path}"}

    manifest = json.loads(manifest_path.read_text())

    if not execute:
        return {
            "replay_id": replay_id,
            "model": model,
            "ran": False,
            "executed": False,
            "scores": None,
            "note": "not executed: pass execute=true / --execute to clone the repo and run "
                    "the recorded test command and runner.",
            "plan": {
                "remote_url": manifest.get("remote_url"),
                "test_command": manifest.get("test_command"),
                "runner_cmd": runner_cmd or os.environ.get("EPISODIC_REPLAY_CMD"),
            },
        }

    remote_url = manifest.get("remote_url")
    base_commit = manifest.get("base_commit")
    repo = manifest.get("repo")
    test_command = manifest.get("test_command")
    test_cwd = manifest.get("test_cwd")
    expected_files = set(manifest.get("expected_outcome", {}).get("files_changed", []))

    if remote_url or repo:
        try:
            free_bytes = shutil.disk_usage(str(replays_root)).free
        except OSError:
            free_bytes = None
        if free_bytes is not None and free_bytes < _MIN_REPLAY_FREE_BYTES:
            return {"error": f"insufficient disk for replay workspace: {free_bytes} bytes free "
                             f"(need >= {_MIN_REPLAY_FREE_BYTES})", "replay_id": replay_id,
                    "model": model, "executed": True, "scores": None}

    workspace = replay_dir / "workspace"
    workspace_created = False
    ok, reason = _reset_workspace(workspace, replays_root)
    if not ok:
        return {"error": reason, "replay_id": replay_id, "model": model,
                "executed": True, "scores": None}

    if remote_url:
        out, code = _run_cmd(["git", "clone", remote_url, str(workspace)], timeout=120)
        if code != 0 or not workspace.exists():
            if workspace.exists() and not workspace.is_symlink():
                shutil.rmtree(workspace, ignore_errors=True)
            return {"error": f"git clone failed (rc={code})", "replay_id": replay_id,
                    "model": model, "executed": True, "scores": None}
        if base_commit:
            _, checkout_rc = _run_cmd(["git", "-C", str(workspace), "checkout", base_commit], timeout=30)
            if checkout_rc != 0:
                shutil.rmtree(workspace, ignore_errors=True)
                return {"error": f"git checkout {base_commit!r} failed", "replay_id": replay_id,
                        "model": model, "executed": True, "scores": None}
        workspace_created = True
    elif repo:
        candidate = manifest.get("repo_root")
        if candidate and (Path(candidate) / ".git").exists():
            try:
                shutil.copytree(
                    candidate,
                    str(workspace),
                    ignore=shutil.ignore_patterns(".git", ".episodic", "node_modules"),
                    symlinks=True,
                )
                _init_git_baseline(workspace)
                workspace_created = True
            except Exception:
                pass

    if not workspace_created and workspace.exists() and not workspace.is_symlink():
        shutil.rmtree(workspace, ignore_errors=True)

    runner_template = runner_cmd or os.environ.get("EPISODIC_REPLAY_CMD")
    ran = False
    dry_run = False
    runner_output = None
    runner_rc = None

    if runner is not None and workspace_created:
        try:
            runner_output, runner_rc = runner(model, workspace, manifest.get("initial_prompt", ""))
        except Exception as exc:
            runner_output, runner_rc = f"runner raised: {exc}", -1
        ran = True
    elif runner_template and workspace_created:
        try:
            cmd_str = runner_template.format(
                model=shlex.quote(model),
                prompt_file=shlex.quote(str(replay_dir / "prompt.txt")),
                workspace=shlex.quote(str(workspace)),
            )
        except (KeyError, IndexError, ValueError) as exc:
            return {"error": f"invalid runner template: {exc}", "replay_id": replay_id,
                    "model": model, "executed": True, "scores": None}
        runner_output, runner_rc = _run_cmd(cmd_str, cwd=str(workspace), timeout=300, shell=True)
        ran = True
    else:
        dry_run = True

    verifier_reverted = []
    if workspace_created and ran:
        verifier_reverted = _protect_verifier(workspace)

    tests_result = None
    test_rc = None
    produced_files = []
    diff_overlap = 0.0

    if workspace_created and test_command:
        try:
            test_cwd_dir = str(workspace / test_cwd) if test_cwd else str(workspace)
            out, rc = _run_cmd(test_command, cwd=test_cwd_dir, timeout=120, shell=True)
            test_rc = rc
            ts = now_iso()
            tests_result = testdetect.detect_test_run(test_command, out, ts, exit_code=rc)
        except Exception:
            pass

    if tests_result is not None:
        passed = tests_result.get("passed", 0)
        failed = tests_result.get("failed", 0)
        errors = tests_result.get("errors", 0)
        tests_result["ok"] = bool(test_rc == 0 and passed > 0 and failed == 0 and errors == 0)

    if workspace_created:
        try:
            diff_out, _ = _run_cmd(["git", "-C", str(workspace), "diff"], timeout=30)
            produced_set = set()
            for line in diff_out.splitlines():
                if line.startswith("+++ b/"):
                    produced_set.add(line[6:])
            untracked, _ = _run_cmd(
                ["git", "-C", str(workspace), "ls-files", "--others", "--exclude-standard"], timeout=30)
            produced_set.update(line for line in untracked.splitlines() if line.strip())
            produced_set = {path for path in produced_set if not _is_noise(path)}
            produced_files = list(produced_set)
            diff_overlap = _jaccard(produced_set, expected_files)
        except Exception:
            pass

    tests_pass_score = 0.0
    if tests_result is not None:
        total = tests_result.get("total", 0)
        passed = tests_result.get("passed", 0)
        if total > 0:
            tests_pass_score = passed / total
        elif tests_result.get("ok"):
            tests_pass_score = 1.0

    scoring_rules = manifest.get("scoring_rules", {})
    w_tests = scoring_rules.get("tests_pass_weight", 0.6)
    w_diff = scoring_rules.get("diff_overlap_weight", 0.4)
    total_score = w_tests * tests_pass_score + w_diff * diff_overlap

    if dry_run:
        scores = None
        note = (
            f"dry run: no workspace created. "
            f"Would clone {remote_url!r}, run {test_command!r} with model {model!r}."
        )
    else:
        scores = {
            "tests_pass": tests_pass_score,
            "diff_overlap": diff_overlap,
            "total": total_score,
        }
        note = "no test command captured for this episode" if not test_command else None

    return {
        "replay_id": replay_id,
        "model": model,
        "ran": ran,
        "dry_run": dry_run,
        "workspace": str(workspace) if workspace_created else None,
        "test_command": test_command,
        "test_cwd": test_cwd,
        "tests": tests_result,
        "produced_files": produced_files,
        "verifier_reverted": verifier_reverted,
        "scores": scores,
        "note": note,
    }
