import json
import time
import urllib.request

from .. import store
from ..core import diffparse
from ..replay import _is_verifier_file
from ..schema import new_episode

_TS = "2026-08-27T00:00:00+00:00"
_DATASET = "nebius/SWE-rebench"
_ROWS_API = "https://datasets-server.huggingface.co/rows"
_PAGE = 100


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [str(parsed)]
        except json.JSONDecodeError:
            return value.split()
    return list(value)


def _node_test_names(node_ids):
    names = []
    for node in node_ids:
        tail = node.split("::")[-1].split("[")[0].strip()
        if tail and tail not in names:
            names.append(tail)
    return names


def _node_paths(node_ids):
    paths = []
    for node in node_ids:
        path = node.split("::")[0].strip()
        if path and path not in paths:
            paths.append(path)
    return paths


def _source_diffs(patch):
    blocks = [b for b in diffparse.parse_unified_diff(patch or "") if b.get("unified")]
    return [{"file": b["file"], "status": "modified", "additions": b.get("additions", 0),
             "deletions": b.get("deletions", 0), "unified": b["unified"]}
            for b in blocks if not _is_verifier_file(b["file"])]


def _test_command(select, test_paths):
    select_expr = f'-k "{select}" ' if select else ""
    return f"python -m pytest -q {select_expr}" + " ".join(test_paths)


def instance_to_episode(instance):
    repo = instance.get("repo") or ""
    base_commit = instance.get("base_commit")
    fail_to_pass = _as_list(instance.get("FAIL_TO_PASS"))
    source_diffs = _source_diffs(instance.get("patch"))
    if not (repo and base_commit and fail_to_pass and source_diffs):
        return None

    instance_id = instance.get("instance_id") or f"{repo}_{str(base_commit)[:8]}"
    select = " or ".join(_node_test_names(fail_to_pass)) or None
    test_paths = _node_paths(fail_to_pass)
    test_command = _test_command(select, test_paths)

    problem = (instance.get("problem_statement") or "").strip()
    intent = (problem + "\n\n" if problem else "")
    intent += ("The following tests fail at the current commit and must pass:\n"
               + "\n".join(f"  {node}" for node in fail_to_pass))

    episode = new_episode(id="swerebench_" + instance_id.replace("/", "_"), intent=intent)
    episode["repo_state"].update({
        "root": None,
        "repo": repo,
        "remote_url": f"https://github.com/{repo}.git",
        "base_commit": base_commit,
        "branch": None,
        "setup_patch": (instance.get("test_patch") or None),
    })
    episode["labels"] = ["swe", "swe-rebench", "certified_by_source"]
    episode["diffs"] = source_diffs
    episode["commands"] = [{"ts": _TS, "command": test_command, "cwd": None, "exit_code": 0,
                            "output_excerpt": "", "is_test": True}]
    episode["tests"] = [{"ts": _TS, "framework": "pytest", "command": test_command,
                         "passed": None, "failed": 0, "skipped": 0, "total": None, "ok": True}]
    episode["outcome"]["status"] = "merged"
    episode["outcome"]["merged"] = True
    episode["diff_source"] = "swe-rebench"
    return episode


def _fetch_page(dataset, config, split, offset, length, retries=3):
    url = (f"{_ROWS_API}?dataset={urllib.request.quote(dataset)}&config={config}"
           f"&split={split}&offset={offset}&length={length}")
    last = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "episodic-ingest"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except Exception as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise last


def stream_instances(split="test", limit=None, dataset=_DATASET, config="default"):
    offset = 0
    seen = 0
    total = None
    while total is None or offset < total:
        page = _fetch_page(dataset, config, split, offset, _PAGE)
        total = page.get("num_rows_total", total)
        rows = page.get("rows", [])
        if not rows:
            break
        for entry in rows:
            yield entry.get("row", {})
            seen += 1
            if limit and seen >= limit:
                return
        offset += len(rows)


def stream_episodes(split="test", limit=None, dataset=_DATASET, repos=None, config="default"):
    kept = 0
    for instance in stream_instances(split, None, dataset, config):
        if repos and (instance.get("repo") or "") not in repos:
            continue
        episode = instance_to_episode(instance)
        if episode is None:
            continue
        yield episode
        kept += 1
        if limit and kept >= limit:
            break


def ingest(split="test", limit=None, dataset=_DATASET, repos=None, save=True, config="default"):
    episodes = []
    for episode in stream_episodes(split, limit, dataset, repos, config):
        if save:
            store.save_episode(episode)
        episodes.append(episode)
    return episodes
