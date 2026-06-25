import json
import re

CONTENT_TYPES = ("deterministic", "pre_existing", "runtime_metadata")

DIMENSIONS = ("format", "factuality", "consistency", "realism", "quality")

DEFAULT_WEIGHTS = {
    "format": 0.15,
    "factuality": 0.35,
    "consistency": 0.20,
    "realism": 0.15,
    "quality": 0.15,
}

_RUNTIME_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?",
    r"\b\d{2}:\d{2}:\d{2}\b",
    r"0x[0-9a-fA-F]+",
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    r"\b[0-9a-f]{12,40}\b",
    r"\b\d+(?:\.\d+)?\s*(?:ms|µs|us|ns)\b",
    r"\b\d+(?:\.\d+)?(?:ms|s)\b",
    r"\bpid[=:\s]+\d+\b",
]

_RUNTIME_RE = re.compile("|".join(f"(?:{p})" for p in _RUNTIME_PATTERNS), re.IGNORECASE)
_ERROR_RE = re.compile(
    r"error|traceback|exception|\bfailed\b|fatal|denied|not found|no such|cannot|"
    r"refused|panic|segfault|exit code [1-9]",
    re.IGNORECASE,
)
_PROMPT_RE = re.compile(r"[\w.-]+@[\w.-]+:.*?[$#]\s|^[$#]\s", re.MULTILINE)


def mask_runtime(text):
    return _RUNTIME_RE.sub("§RT§", text or "")


def is_json(text):
    text = (text or "").strip()
    if not text or text[0] not in "{[":
        return False
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def response_type(text):
    if not (text or "").strip():
        return "empty"
    if _ERROR_RE.search(text):
        return "error"
    return "success"


def classify_content(text):
    counts = {key: 0 for key in CONTENT_TYPES}
    for token in (text or "").split():
        if _RUNTIME_RE.search(token):
            counts["runtime_metadata"] += 1
        else:
            counts["deterministic"] += 1
    return {
        **counts,
        "lines": len((text or "").splitlines()),
        "chars": len(text or ""),
        "is_json": is_json(text),
        "response_type": response_type(text),
    }


def _tokens(text):
    return set(mask_runtime(text).split())


def _recall_precision(predicted, ground_truth):
    pred = _tokens(predicted)
    gold = _tokens(ground_truth)
    inter = pred & gold
    recall = 1.0 if not gold else len(inter) / len(gold)
    precision = 1.0 if not pred else len(inter) / len(pred)
    return recall, precision


def _format_score(predicted, ground_truth):
    pred_lines = len((predicted or "").splitlines())
    gold_lines = len((ground_truth or "").splitlines())
    line_sim = 1.0 - min(1.0, abs(pred_lines - gold_lines) / max(1, gold_lines))
    json_match = 1.0 if is_json(predicted) == is_json(ground_truth) else 0.0
    prompt_match = 1.0 if bool(_PROMPT_RE.search(predicted or "")) == bool(_PROMPT_RE.search(ground_truth or "")) else 0.0
    return (line_sim + json_match + prompt_match) / 3.0


def _quality_score(predicted, ground_truth):
    pred_len = len(predicted or "")
    gold_len = len(ground_truth or "")
    if gold_len == 0:
        return 1.0 if pred_len == 0 else 0.0
    return max(0.0, 1.0 - abs(pred_len - gold_len) / gold_len)


def score_observation(predicted, ground_truth, weights=None):
    weights = weights or DEFAULT_WEIGHTS
    exact = (predicted or "") == (ground_truth or "")
    if exact:
        scores = {dim: 1.0 for dim in DIMENSIONS}
    else:
        recall, precision = _recall_precision(predicted, ground_truth)
        scores = {
            "format": _format_score(predicted, ground_truth),
            "factuality": recall,
            "consistency": precision,
            "realism": 1.0 if response_type(predicted) == response_type(ground_truth) else 0.0,
            "quality": _quality_score(predicted, ground_truth),
        }
    composite = sum(weights[dim] * scores[dim] for dim in DIMENSIONS)
    return {
        **{dim: round(scores[dim], 4) for dim in DIMENSIONS},
        "composite": round(composite, 4),
        "exact": exact,
        "classification": classify_content(ground_truth),
        "predicted_type": response_type(predicted),
        "ground_truth_type": response_type(ground_truth),
    }
