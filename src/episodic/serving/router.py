import json
from pathlib import Path

from . import build, difficulty, BackendUnavailable
from .. import paths

TIERS = ("distilled", "frontier")


def resolve_served_ref(start=None, out=None):
    loop_dir = Path(out) if out else (paths.exports_dir(start) / "loop")
    promoted_path = loop_dir / "promoted.json"
    if not promoted_path.exists():
        return None
    try:
        data = json.loads(promoted_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data.get("served_ref") or data.get("model_dir")


def default_escalate(payload, config):
    if payload.get("episodic_escalate"):
        return True
    max_chars = config.get("escalate_chars")
    if max_chars:
        messages = payload.get("messages") or []
        total = sum(len(str(message.get("content", ""))) for message in messages)
        if total > max_chars:
            return True
    return False


def cost_aware_escalate(payload, config):
    if payload.get("episodic_escalate"):
        return True
    decision = difficulty.learned_escalate(payload, config)
    if decision is not None:
        return decision
    return default_escalate(payload, config)


class Router:
    def __init__(self, config=None, start=None):
        self.config = dict(config or {})
        self.start = start
        self.escalate_fn = self.config.get("escalate") or self._default_escalate_fn()

    def _default_escalate_fn(self):
        if self.config.get("router_model") or self.config.get("router_model_path"):
            return cost_aware_escalate
        return default_escalate

    def _tier_config(self, tier):
        tier_config = dict(self.config.get(tier) or {})
        if tier == "distilled" and not tier_config.get("model") and not tier_config.get("sampler_path"):
            served_ref = tier_config.get("served_ref") or resolve_served_ref(self.start, self.config.get("loop_out"))
            if served_ref:
                tier_config["model"] = served_ref
                tier_config["served_ref"] = served_ref
        return tier_config

    def select_tier(self, payload):
        requested = payload.get("model")
        if requested in TIERS:
            return requested
        if self.escalate_fn(payload, self.config):
            return "frontier"
        return "distilled"

    def backend_for(self, tier):
        tier_config = self._tier_config(tier)
        name = tier_config.get("backend", "openai")
        if name == "openai" and not tier_config.get("base_url") and not tier_config.get("api_key"):
            raise BackendUnavailable(
                "openai",
                f"serve tier '{tier}' is not configured; refusing to proxy to the public OpenAI API by "
                f"default. Set a base_url (self-hosted / vLLM / Ollama) or an api_key for this tier.",
            )
        return build(name, tier_config), tier_config

    def route(self, payload):
        tier = self.select_tier(payload)
        backend, tier_config = self.backend_for(tier)
        return tier, backend, tier_config

    def list_models(self):
        entries = []
        for tier in TIERS:
            tier_config = self._tier_config(tier)
            model_id = (
                tier_config.get("served_ref")
                or tier_config.get("model")
                or tier_config.get("sampler_path")
                or tier
            )
            entries.append({"id": model_id, "object": "model", "owned_by": tier, "tier": tier})
        return entries
