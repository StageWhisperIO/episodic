import json
import math
from pathlib import Path

SCHEMA_VERSION = "0.1.0"

FEATURE_NAMES = ("log_length", "keyword_hits", "question_marks", "code_fences", "avg_word_length")

_HARD_KEYWORDS = (
    "error", "traceback", "exception", "bug", "fail", "crash", "regression",
    "race condition", "deadlock", "security", "vulnerab", "migrat", "refactor",
    "concurrency", "performance", "memory leak", "flaky", "timeout",
)

MIN_TRAINING_EXAMPLES = 4


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _sigmoid(z):
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def request_text(payload):
    messages = payload.get("messages") or []
    return "\n".join(str(message.get("content", "")) for message in messages)


def text_features(text):
    text = text or ""
    lowered = text.lower()
    words = text.split()
    avg_word_length = (sum(len(word) for word in words) / len(words)) if words else 0.0
    return {
        "log_length": math.log1p(len(text)),
        "keyword_hits": float(sum(lowered.count(word) for word in _HARD_KEYWORDS)),
        "question_marks": float(text.count("?")),
        "code_fences": float(text.count("```")),
        "avg_word_length": avg_word_length,
    }


def _vector(features):
    return [features[name] for name in FEATURE_NAMES]


def episode_difficulty(episode):
    reward_vector = episode.get("reward_vector") or {}
    composite = reward_vector.get("composite")
    base = 0.5 if composite is None else (1.0 - _clamp(composite))

    validity = episode.get("validity") or {}
    trust = validity.get("trust")
    if trust == "low":
        base += 0.25
    elif trust == "medium":
        base += 0.1

    rubric_summary = (reward_vector.get("components") or {}).get("rubric") or {}
    if rubric_summary.get("hard_violations"):
        base += 0.15

    denials = (episode.get("stats") or {}).get("denials", 0)
    if denials:
        base += min(0.15, 0.05 * denials)

    return _clamp(base)


def _episode_text(episode):
    return episode.get("intent") or ""


def _fit_logistic(rows, labels, epochs=300, lr=0.2, l2=0.001):
    n = len(rows)
    dim = len(rows[0])
    weights = [0.0] * dim
    bias = 0.0
    for _ in range(epochs):
        grad_w = [0.0] * dim
        grad_b = 0.0
        for x, y in zip(rows, labels):
            z = bias + sum(w * xi for w, xi in zip(weights, x))
            error = _sigmoid(z) - y
            for i in range(dim):
                grad_w[i] += error * x[i]
            grad_b += error
        weights = [w - lr * (grad_w[i] / n + l2 * w) for i, w in enumerate(weights)]
        bias -= lr * grad_b / n
    return weights, bias


def _standardize(matrix):
    dim = len(matrix[0])
    n = len(matrix)
    mean = [sum(row[i] for row in matrix) / n for i in range(dim)]
    variance = [sum((row[i] - mean[i]) ** 2 for row in matrix) / n for i in range(dim)]
    std = [math.sqrt(v) or 1.0 for v in variance]
    standardized = [[(row[i] - mean[i]) / std[i] for i in range(dim)] for row in matrix]
    return standardized, mean, std


def learn_router(episodes, hard_threshold=0.5, epochs=300, lr=0.2):
    examples = []
    for episode in episodes:
        text = _episode_text(episode)
        if not text or not episode.get("reward_vector"):
            continue
        difficulty = episode_difficulty(episode)
        examples.append((_vector(text_features(text)), 1.0 if difficulty >= hard_threshold else 0.0))

    trained_on = len(examples)
    positive_rate = (sum(label for _, label in examples) / trained_on) if trained_on else None

    if trained_on < MIN_TRAINING_EXAMPLES or positive_rate in (0.0, 1.0):
        return {
            "schema_version": SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "mean": [0.0] * len(FEATURE_NAMES),
            "std": [1.0] * len(FEATURE_NAMES),
            "weights": [0.0] * len(FEATURE_NAMES),
            "bias": 0.0 if not trained_on else _logit(positive_rate),
            "hard_threshold": hard_threshold,
            "trained_on": trained_on,
            "positive_rate": positive_rate,
            "fallback": True,
        }

    matrix, labels = [row for row, _ in examples], [label for _, label in examples]
    standardized, mean, std = _standardize(matrix)
    weights, bias = _fit_logistic(standardized, labels, epochs=epochs, lr=lr)

    return {
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "mean": mean,
        "std": std,
        "weights": weights,
        "bias": bias,
        "hard_threshold": hard_threshold,
        "trained_on": trained_on,
        "positive_rate": positive_rate,
        "fallback": False,
    }


def _logit(p):
    p = _clamp(p, 0.001, 0.999)
    return math.log(p / (1.0 - p))


def predict_proba(text, model):
    features = _vector(text_features(text))
    mean = model.get("mean") or [0.0] * len(features)
    std = model.get("std") or [1.0] * len(features)
    weights = model.get("weights") or [0.0] * len(features)
    z = model.get("bias", 0.0)
    for xi, mi, si, wi in zip(features, mean, std, weights):
        normalized = (xi - mi) / si if si else 0.0
        z += wi * normalized
    return _sigmoid(z)


def save_router_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    return str(path)


def load_router_model(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


_MODEL_CACHE = {}


def _cached_model_from_path(path):
    try:
        mtime = Path(path).stat().st_mtime
    except OSError:
        return None
    cached = _MODEL_CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        model = load_router_model(path)
    except (OSError, ValueError):
        return None
    _MODEL_CACHE[path] = (mtime, model)
    return model


def resolve_router_model(config):
    inline = config.get("router_model")
    if inline is not None:
        return inline
    path = config.get("router_model_path")
    if not path:
        return None
    return _cached_model_from_path(path)


def learned_escalate(payload, config):
    model = resolve_router_model(config)
    if not model:
        return None
    text = request_text(payload)
    probability = predict_proba(text, model)
    threshold = config.get("router_threshold", model.get("hard_threshold", 0.5))
    return probability >= threshold
