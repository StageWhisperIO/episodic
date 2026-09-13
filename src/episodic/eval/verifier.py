import math

DEFAULT_CRITERIA = {
    "Correctness": "Does the change correctly and completely solve the task described?",
    "Verification": "Would the project's tests pass after this change is applied?",
}
RATING_OPTIONS = tuple(str(digit) for digit in range(10))
PAIR_OPTIONS = ("A", "B")


def softmax(logprobs):
    if not logprobs:
        return {}
    highest = max(logprobs.values())
    exps = {key: math.exp(value - highest) for key, value in logprobs.items()}
    total = sum(exps.values())
    if total <= 0:
        uniform = 1.0 / len(exps)
        return {key: uniform for key in exps}
    return {key: value / total for key, value in exps.items()}


def expected_rating(logprobs, options=RATING_OPTIONS):
    present = {option: logprobs[option] for option in options if option in logprobs}
    probs = softmax(present)
    if not probs:
        return 0.0
    span = (len(options) - 1) or 1
    return sum(prob * (int(option) / span) for option, prob in probs.items())


def _pointwise_messages(problem, candidate, criterion):
    system = ("You are a strict code reviewer. Rate how well the candidate change satisfies the "
              "criterion on an integer scale from 0 (fails completely) to 9 (fully satisfies). "
              "Reply with a single digit 0-9 and nothing else.")
    user = (f"Task:\n{problem}\n\nCriterion: {criterion}\n\n"
            f"Candidate change (unified diff):\n{candidate}\n\nRating (0-9):")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def fine_grained_reward(problem, candidate, criteria, option_logprob_fn, options=RATING_OPTIONS):
    criteria = criteria or DEFAULT_CRITERIA
    scores = {}
    for name, description in criteria.items():
        messages = _pointwise_messages(problem, candidate, description)
        scores[name] = expected_rating(option_logprob_fn(messages, list(options)), options)
    reward = sum(scores.values()) / len(scores) if scores else 0.0
    return {"reward": reward, "criteria": scores}


def _pairwise_messages(problem, candidate_a, candidate_b, criterion):
    system = ("You are a strict code reviewer comparing two candidate changes for the same task. "
              "Answer with a single letter: A if the first is better on the criterion, B if the "
              "second is better.")
    user = (f"Task:\n{problem}\n\nCriterion: {criterion}\n\n"
            f"Candidate A (unified diff):\n{candidate_a}\n\n"
            f"Candidate B (unified diff):\n{candidate_b}\n\nBetter candidate (A or B):")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _prob_a(problem, candidate_a, candidate_b, criterion, option_logprob_fn):
    messages = _pairwise_messages(problem, candidate_a, candidate_b, criterion)
    return softmax(option_logprob_fn(messages, list(PAIR_OPTIONS))).get("A", 0.0)


def compare(problem, candidate_a, candidate_b, criteria, option_logprob_fn, swap=True):
    criteria = criteria or DEFAULT_CRITERIA
    per_criterion = {}
    for name, description in criteria.items():
        prob = _prob_a(problem, candidate_a, candidate_b, description, option_logprob_fn)
        if swap:
            swapped = _prob_a(problem, candidate_b, candidate_a, description, option_logprob_fn)
            prob = 0.5 * (prob + (1.0 - swapped))
        per_criterion[name] = prob
    reward_a = sum(per_criterion.values()) / len(per_criterion) if per_criterion else 0.5
    return {"reward_a": reward_a, "reward_b": 1.0 - reward_a, "criteria": per_criterion}
