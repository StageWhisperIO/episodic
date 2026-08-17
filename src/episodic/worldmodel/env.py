from . import WM_SYSTEM, _action_repr

HISTORY_BUDGET = 4000
OBSERVATION_BUDGET = 400


class WorldModelEnv:
    def __init__(self, episode, predictor, policy=None, history_budget=HISTORY_BUDGET, system=None):
        self.episode = episode
        self.predictor = predictor
        self.policy = policy
        self.history_budget = history_budget
        self.system = system or WM_SYSTEM
        self.steps = episode.get("steps", [])
        self.transcript = []
        self.turn_index = 0
        self.done = len(self.steps) == 0
        self.last_predicted = ""

    def reset(self):
        self.transcript = []
        self.turn_index = 0
        self.done = len(self.steps) == 0
        self.last_predicted = ""
        return {"episode_id": self.episode.get("id"), "turn_index": self.turn_index, "done": self.done}

    def _context(self):
        return "\n".join(self.transcript)[-self.history_budget:]

    def _action_text(self, index):
        if self.policy is not None:
            return self.policy(self.episode, index, list(self.transcript))
        return _action_repr(self.steps[index])

    def step(self):
        if self.done:
            raise RuntimeError("WorldModelEnv.step called after the episode ended")
        index = self.turn_index
        action_text = self._action_text(index)
        context = self._context()
        intent_line = f"INTENT: {self.episode.get('intent', '')}"
        history = f"{intent_line}\n{context}" if context else intent_line
        sample = {
            "episode_id": self.episode.get("id"),
            "turn_index": index,
            "action": action_text,
            "history": f"{history}\nACTION: {action_text}\nOBSERVATION:",
            "prev_observation": self.last_predicted,
            "target_observation": self.steps[index].get("observation") or "",
        }
        predicted = self.predictor(sample) or ""
        self.transcript.append(f"ACTION: {action_text}")
        self.transcript.append(f"OBSERVATION: {predicted[:OBSERVATION_BUDGET]}")
        self.last_predicted = predicted
        self.turn_index += 1
        self.done = self.turn_index >= len(self.steps)
        return {
            "turn_index": index,
            "action": action_text,
            "predicted_observation": predicted,
            "target_observation": sample["target_observation"],
            "done": self.done,
        }


def rollout(episode, predictor, policy=None, history_budget=HISTORY_BUDGET, max_turns=None, system=None):
    env = WorldModelEnv(episode, predictor, policy=policy, history_budget=history_budget, system=system)
    env.reset()
    limit = len(env.steps) if max_turns is None else max(0, min(max_turns, len(env.steps)))
    turns = []
    while not env.done and len(turns) < limit:
        turns.append(env.step())
    return {
        "episode_id": episode.get("id"),
        "turns": turns,
        "truncated": len(turns) < len(env.steps),
    }
