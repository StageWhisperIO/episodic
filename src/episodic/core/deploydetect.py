import re

_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
_SEGMENT_SPLIT = re.compile(r"\|\||&&|[;\n]")

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

_PROD = re.compile(r"--prod\b|--production\b|\bproduction\b|\bprod\b|\bmain\b|\bmaster\b")
_STAGING = re.compile(r"\bstaging\b|\bstage\b|--env[= ]stag")
_DEV = re.compile(r"\bpreview\b|\bdev\b|\bdevelopment\b|--env[= ]dev")


def _target_env(segment):
    if _PROD.search(segment):
        return "prod"
    if _STAGING.search(segment):
        return "staging"
    if _DEV.search(segment):
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
