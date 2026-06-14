import re

_OLD_FILE = re.compile(r"^--- (?:a/)?(.+)$")
_NEW_FILE = re.compile(r"^\+\+\+ (?:b/)?(.+)$")
_DIFF_HEADER = re.compile(r"^diff --git a/(.+) b/(.+)$")


def _finalize(block):
    if block is None:
        return None
    if block["file"] is None and block["new_file"]:
        block["file"] = block["new_file"]
    if block["new_file"] and block["new_file"] != "/dev/null":
        block["file"] = block["new_file"]
    elif block["old_file"] and block["old_file"] != "/dev/null":
        block["file"] = block["old_file"]
    status = "modified"
    if block["binary"]:
        status = "binary"
    elif block["old_file"] == "/dev/null":
        status = "added"
    elif block["new_file"] == "/dev/null":
        status = "deleted"
    return {
        "file": block["file"] or block["header_file"] or "unknown",
        "status": status,
        "additions": block["additions"],
        "deletions": block["deletions"],
        "unified": "\n".join(block["lines"]).strip() or None,
    }


def parse_unified_diff(patch):
    if not patch:
        return []
    files = []
    block = None

    def start_block(header_file=None):
        return {
            "header_file": header_file,
            "old_file": None,
            "new_file": None,
            "file": None,
            "additions": 0,
            "deletions": 0,
            "binary": False,
            "lines": [],
        }

    for line in patch.splitlines():
        header = _DIFF_HEADER.match(line)
        if header:
            if block is not None:
                files.append(_finalize(block))
            block = start_block(header.group(2))
            block["lines"].append(line)
            continue
        if block is None:
            old = _OLD_FILE.match(line)
            if old:
                block = start_block()
                block["old_file"] = old.group(1)
                block["lines"].append(line)
            continue
        block["lines"].append(line)
        if line.startswith("Binary files"):
            block["binary"] = True
            continue
        old = _OLD_FILE.match(line)
        if old:
            block["old_file"] = old.group(1)
            continue
        new = _NEW_FILE.match(line)
        if new:
            block["new_file"] = new.group(1)
            continue
        if line.startswith("+") and not line.startswith("+++"):
            block["additions"] += 1
        elif line.startswith("-") and not line.startswith("---"):
            block["deletions"] += 1

    if block is not None:
        files.append(_finalize(block))
    return [entry for entry in files if entry]


def diff_stats(patches):
    return {
        "files": len(patches),
        "additions": sum(entry["additions"] for entry in patches),
        "deletions": sum(entry["deletions"] for entry in patches),
    }
