import re

from . import testdetect

_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
_SEGMENT_SPLIT = testdetect._SEGMENT_SPLIT

DEPLOY_PATTERNS = (
    ("wrangler", re.compile(r"\bwrangler\s+(?:pages\s+)?deploy\b")),
    ("vercel", re.compile(r"\bvercel\s+(?:deploy\b|--prod\b|deploy\s+--prod\b)|\bvercel\s+--prod\b")),
    ("netlify", re.compile(r"\bnetlify\s+deploy\b")),
    ("gh-workflow", re.compile(r"\bgh\s+workflow\s+run\b")),
    ("gh-release", re.compile(r"\bgh\s+release\s+create\b")),
    ("kubectl", re.compile(r"\bkubectl\s+apply\b")),
    ("helm", re.compile(r"\bhelm\s+(?:upgrade|install)\b")),
    ("flyctl", re.compile(r"\bfly(?:ctl)?\s+deploy\b")),
    ("firebase", re.compile(r"\bfirebase\s+deploy\b")),
    ("serverless", re.compile(r"\b(?:serverless|sls)\s+deploy\b")),
    ("eb", re.compile(r"\beb\s+deploy\b")),
    ("cap", re.compile(r"\bcap\s+\w+\s+deploy\b")),
    ("docker-push", re.compile(r"\bdocker\s+push\b")),
    ("npm-deploy", re.compile(r"\b(?:npm|pnpm|yarn|bun)\s+run\s+deploy\b")),
    ("make-deploy", re.compile(r"\bmake\s+deploy\b")),
)

_PROD_FLAGS = re.compile(r"--prod\b|--production\b")
_PROD_WORDS = ("production", "prod", "main", "master")
_STAGING_FLAGS = re.compile(r"--env[= ]stag")
_STAGING_WORDS = ("staging", "stage")
_DEV_FLAGS = re.compile(r"--env[= ]dev")
_DEV_WORDS = ("preview", "dev", "development")

_TOKEN_ADJACENT = set("/.:")


def _bare_word_hit(segment, word):
    for match in re.finditer(r"\b" + re.escape(word) + r"\b", segment):
        start, end = match.span()
        before = segment[start - 1] if start > 0 else " "
        after = segment[end] if end < len(segment) else " "
        if after in _TOKEN_ADJACENT or after == "-":
            continue
        if before in _TOKEN_ADJACENT:
            continue
        if before == "-" and not (start >= 2 and segment[start - 2] == "-"):
            continue
        return True
    return False


def _target_env(segment):
    if _PROD_FLAGS.search(segment) or any(_bare_word_hit(segment, word) for word in _PROD_WORDS):
        return "prod"
    if _STAGING_FLAGS.search(segment) or any(_bare_word_hit(segment, word) for word in _STAGING_WORDS):
        return "staging"
    if _DEV_FLAGS.search(segment) or any(_bare_word_hit(segment, word) for word in _DEV_WORDS):
        return "dev"
    return "unknown"


def classify_deploy(command):
    if not command:
        return None
    stripped = _QUOTED.sub(" ", command)
    for segment in _SEGMENT_SPLIT.split(stripped):
        segment = segment.strip()
        if not segment:
            continue
        for method, pattern in DEPLOY_PATTERNS:
            if pattern.search(segment):
                return {"method": method, "target_env": _target_env(segment)}
    return None
