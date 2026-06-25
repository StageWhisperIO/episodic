import hashlib
import os
import shlex
import subprocess
import tempfile

from episodic import fidelity
from episodic.worldmodel import wm_samples, ood_split


def oracle_predictor(sample):
    return sample["target_observation"]


def empty_predictor(sample):
    return ""


def echo_predictor(sample):
    return sample.get("action", "")


def prefix_predictor(sample):
    return sample.get("prev_observation", "")


NAMED_PREDICTORS = {
    "oracle": oracle_predictor,
    "empty": empty_predictor,
    "echo": echo_predictor,
    "prefix": prefix_predictor,
}


def command_predictor(template, timeout=120):
    def predict(sample):
        handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
        handle.write(sample["history"])
        handle.close()
        cmd = template.format(prompt_file=shlex.quote(handle.name))
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            return result.stdout
        except Exception:
            return ""
        finally:
            os.unlink(handle.name)

    return predict


def _mean(values):
    return round(sum(values) / len(values), 4) if values else None


def _aggregate(scored):
    if not scored:
        return {dim: None for dim in fidelity.DIMENSIONS + ("composite",)}
    out = {}
    for dim in fidelity.DIMENSIONS + ("composite",):
        out[dim] = _mean([s["score"][dim] for s in scored])
    out["exact_rate"] = _mean([1.0 if s["score"]["exact"] else 0.0 for s in scored])
    return out


def _group(scored, key):
    groups = {}
    for item in scored:
        groups.setdefault(item[key], []).append(item)
    return {name: _aggregate(items) for name, items in groups.items()}


def _blend(rule, judge_score, weight):
    out = dict(rule)
    present = [dim for dim in fidelity.DIMENSIONS if dim in judge_score]
    for dim in present:
        out[dim] = round((1 - weight) * rule[dim] + weight * float(judge_score[dim]), 4)
    if present:
        out["composite"] = round(
            sum(fidelity.DEFAULT_WEIGHTS[dim] * out[dim] for dim in fidelity.DIMENSIONS), 4)
    elif "composite" in judge_score:
        out["composite"] = round(
            (1 - weight) * rule["composite"] + weight * float(judge_score["composite"]), 4)
    out["judged"] = True
    return out


def run_bench(episodes, predictor="prefix", *, one_per_trajectory=True, seed=0,
              source_holdout=False, holdout_frac=0.3, keep_samples=False,
              judge=None, judge_weight=0.5):
    if isinstance(predictor, str):
        if predictor not in NAMED_PREDICTORS:
            raise ValueError(f"unknown predictor {predictor!r}; choose from {sorted(NAMED_PREDICTORS)}")
        predict = NAMED_PREDICTORS[predictor]
        predictor_name = predictor
    else:
        predict = predictor
        predictor_name = getattr(predictor, "__name__", "callable")

    pool = episodes
    split_info = None
    if source_holdout:
        _, pool, mapping = ood_split(episodes, holdout_frac=holdout_frac, seed=seed)
        split_info = mapping

    samples = wm_samples(pool, one_per_trajectory=one_per_trajectory, seed=seed)
    scored = []
    for sample in samples:
        predicted = predict(sample)
        score = fidelity.score_observation(predicted, sample["target_observation"])
        if judge is not None:
            score = _blend(score, judge(predicted, sample["target_observation"]), judge_weight)
        row = {
            "episode_id": sample["episode_id"],
            "turn_index": sample["turn_index"],
            "domain": sample["domain"],
            "source": sample["source"],
            "score": score,
        }
        if keep_samples:
            row["predicted"] = predicted
            row["target_observation"] = sample["target_observation"]
        scored.append(row)

    report = {
        "predictor": predictor_name,
        "n": len(scored),
        "one_per_trajectory": one_per_trajectory,
        "source_holdout": source_holdout,
        "hybrid": judge is not None,
        "overall": _aggregate(scored),
        "by_domain": _group(scored, "domain"),
        "by_source": _group(scored, "source"),
    }
    if split_info is not None:
        report["split"] = split_info
    if keep_samples:
        report["samples"] = scored
    return report


def _realness(text):
    score = len(fidelity._RUNTIME_RE.findall(text or ""))
    if fidelity._PROMPT_RE.search(text or ""):
        score += 2
    score += min(3, len(text or "") // 100)
    return score


def default_discriminator(candidate_a, candidate_b):
    return 0 if _realness(candidate_a) >= _realness(candidate_b) else 1


def _real_position(sample_id, seed):
    return int(hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()[:8], 16) % 2


def turing_test(episodes, predictor="prefix", *, discriminator=None, one_per_trajectory=True, seed=0):
    discriminator = discriminator or default_discriminator
    predict = NAMED_PREDICTORS[predictor] if isinstance(predictor, str) else predictor
    samples = wm_samples(episodes, one_per_trajectory=one_per_trajectory, seed=seed)
    correct = 0
    for sample in samples:
        real = sample["target_observation"]
        fake = predict(sample)
        real_pos = _real_position(f"{sample['episode_id']}:{sample['turn_index']}", seed)
        pair = [fake, fake]
        pair[real_pos] = real
        pair[1 - real_pos] = fake
        guess = discriminator(pair[0], pair[1])
        if guess == real_pos:
            correct += 1
    n = len(samples)
    return {
        "predictor": predictor if isinstance(predictor, str) else "callable",
        "n": n,
        "discriminator_accuracy": round(correct / n, 4) if n else None,
        "indistinguishability": round(1.0 - abs(correct / n - 0.5) * 2, 4) if n else None,
    }
