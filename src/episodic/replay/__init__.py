import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from episodic import paths
from episodic.schema import now_iso
from episodic.core import testdetect


def replay_id_for(episode):
    raw = episode["id"]
    suffix = raw.removeprefix("ep_")
    return "rp_" + suffix


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


def create_replay(episode, start=None):
    replay_id = replay_id_for(episode)
    repo_state = episode.get("repo_state", {})
    repo_root = repo_state.get("root")

    test_command = infer_test_command(repo_root or "", episode) if repo_root else None
    if test_command is None:
        for cmd in episode.get("commands", []):
            if cmd.get("is_test"):
                test_command = cmd["command"]
                break

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
        "branch": repo_state.get("branch"),
        "initial_prompt": episode.get("intent", ""),
        "test_command": test_command,
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

    unified_diffs = "\n".join(d.get("unified", "") or "" for d in diffs)
    (replay_dir / "expected.diff").write_text(unified_diffs)

    return manifest


def _run_cmd(args, cwd=None, timeout=60):
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout + result.stderr, result.returncode
    except Exception:
        return "", -1


def _jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def run_replay(replay_id, model, start=None):
    replay_dir = paths.replays_dir(start) / replay_id
    manifest_path = replay_dir / "manifest.json"

    if not manifest_path.exists():
        return {"error": f"manifest not found: {manifest_path}"}

    manifest = json.loads(manifest_path.read_text())

    remote_url = manifest.get("remote_url")
    base_commit = manifest.get("base_commit")
    repo = manifest.get("repo")
    test_command = manifest.get("test_command")
    expected_files = set(manifest.get("expected_outcome", {}).get("files_changed", []))

    workspace = replay_dir / "workspace"
    workspace_created = False

    if remote_url:
        out, code = _run_cmd(["git", "clone", remote_url, str(workspace)], timeout=120)
        if code == 0 and base_commit:
            _run_cmd(["git", "-C", str(workspace), "checkout", base_commit], timeout=30)
        if workspace.exists():
            workspace_created = True
    elif repo:
        candidate = manifest.get("repo_state", {}).get("root") if "repo_state" in manifest else None
        if candidate and Path(candidate).exists():
            try:
                shutil.copytree(
                    candidate,
                    str(workspace),
                    ignore=shutil.ignore_patterns(".git", ".episodic", "node_modules"),
                )
                workspace_created = True
            except Exception:
                pass

    runner_template = os.environ.get("EPISODIC_REPLAY_CMD")
    ran = False
    dry_run = False
    runner_output = None
    runner_rc = None

    if runner_template and workspace_created:
        cmd_str = runner_template.format(
            model=model,
            prompt_file=str(replay_dir / "prompt.txt"),
            workspace=str(workspace),
        )
        runner_output, runner_rc = _run_cmd(cmd_str, cwd=str(workspace), timeout=300)
        ran = True
    else:
        dry_run = True

    tests_result = None
    produced_files = []
    diff_overlap = 0.0

    if workspace_created and test_command:
        try:
            out, rc = _run_cmd(test_command.split(), cwd=str(workspace), timeout=120)
            ts = now_iso()
            tests_result = testdetect.detect_test_run(test_command, out, ts)
        except Exception:
            pass

    if workspace_created:
        try:
            diff_out, _ = _run_cmd(["git", "-C", str(workspace), "diff"], timeout=30)
            produced_set = set()
            for line in diff_out.splitlines():
                if line.startswith("+++ b/"):
                    produced_set.add(line[6:])
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
        note = None

    return {
        "replay_id": replay_id,
        "model": model,
        "ran": ran,
        "dry_run": dry_run,
        "workspace": str(workspace) if workspace_created else None,
        "test_command": test_command,
        "tests": tests_result,
        "produced_files": produced_files,
        "scores": scores,
        "note": note,
    }
