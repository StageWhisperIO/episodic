import re

from ..core import gitinfo

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")
_OLD_FILE = re.compile(r"^--- (?:a/)?(.+)$")
_SHA = re.compile(r"^([0-9a-f]{40}) \d+ \d+ (\d+)$")


def _diff(fix_commit, cwd):
    patch = gitinfo._run_diff(
        ["git", "diff", f"{fix_commit}^", fix_commit, "--unified=0", "--no-color"], cwd
    )
    if patch is None:
        patch = gitinfo._run_diff(
            ["git", "show", fix_commit, "--unified=0", "--no-color", "--format="], cwd
        )
    return patch or ""


def old_ranges_by_file(fix_commit, cwd):
    ranges = {}
    current = None
    for line in _diff(fix_commit, cwd).splitlines():
        old_file = _OLD_FILE.match(line)
        if old_file:
            path = old_file.group(1).strip()
            current = None if path == "/dev/null" else path
            continue
        hunk = _HUNK.match(line)
        if hunk and current:
            start = int(hunk.group(1))
            count = int(hunk.group(2)) if hunk.group(2) is not None else 1
            if count > 0:
                ranges.setdefault(current, []).append((start, count))
    return ranges


def _blame(path, spans, fix_commit, cwd):
    counts = {}
    for start, count in spans:
        out = gitinfo._run(
            ["git", "blame", f"{fix_commit}^", "-L", f"{start},+{count}", "--porcelain", "--", path],
            cwd,
        )
        if not out:
            continue
        for line in out.splitlines():
            match = _SHA.match(line)
            if match:
                sha = match.group(1)
                counts[sha] = counts.get(sha, 0) + int(match.group(2))
    return counts


def culprit_commits(fix_commit, cwd):
    files = old_ranges_by_file(fix_commit, cwd)
    culprits = {}
    for path, spans in files.items():
        for sha, lines in _blame(path, spans, fix_commit, cwd).items():
            culprits[sha] = culprits.get(sha, 0) + lines
    return culprits, set(files)


def _episode_files(episode):
    return {diff["file"] for diff in episode.get("diffs", [])}


def map_to_episodes(culprits, regressed_files, episodes):
    by_commit = {}
    for episode in episodes:
        outcome = episode.get("outcome") or {}
        for commit in (outcome.get("commit"), outcome.get("merge_commit")):
            if commit:
                by_commit.setdefault(commit, episode)

    implications = []
    matched_ids = set()
    for sha, lines in sorted(culprits.items(), key=lambda kv: -kv[1]):
        episode = by_commit.get(sha)
        if episode and episode["id"] not in matched_ids:
            matched_ids.add(episode["id"])
            implications.append({
                "episode_id": episode["id"],
                "via": "commit",
                "commit": sha,
                "blamed_lines": lines,
            })

    for episode in episodes:
        if episode["id"] in matched_ids:
            continue
        overlap = sorted(_episode_files(episode) & regressed_files)
        if overlap:
            implications.append({
                "episode_id": episode["id"],
                "via": "file",
                "files": overlap,
                "blamed_lines": 0,
            })
    return implications


def regression_report(fix_commit, cwd, episodes):
    culprits, regressed_files = culprit_commits(fix_commit, cwd)
    return {
        "fix_commit": fix_commit,
        "regressed_files": sorted(regressed_files),
        "culprit_commits": [{"commit": sha, "blamed_lines": lines} for sha, lines in
                            sorted(culprits.items(), key=lambda kv: -kv[1])],
        "implicated": map_to_episodes(culprits, regressed_files, episodes),
    }
