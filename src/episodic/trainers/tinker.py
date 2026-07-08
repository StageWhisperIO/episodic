import os

from . import register, TrainerUnavailable
from .trl import _read_rows

HINT = (
    "Tinker backend needs the SDK and an API key: pip install tinker, then "
    "export TINKER_API_KEY. Training runs on Thinking Machines' GPUs "
    "(https://tinker-console.thinkingmachines.ai)."
)
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def _require_tinker(name):
    try:
        import tinker  # noqa: F401
    except ImportError as exc:
        raise TrainerUnavailable(name, HINT) from exc
    if not os.environ.get("TINKER_API_KEY"):
        raise TrainerUnavailable(name, HINT)


def _token_ids(rendered):
    ids = rendered["input_ids"] if hasattr(rendered, "keys") else rendered
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return list(ids)


def _strip_reasoning(text):
    marker = "</think>"
    index = text.rfind(marker)
    return text[index + len(marker):].strip() if index != -1 else text.strip()


def _common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _sft_datum(types, tokenizer, messages):
    prompt_ids = _token_ids(tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True))
    full_ids = _token_ids(tokenizer.apply_chat_template(messages, tokenize=True))
    boundary = _common_prefix_len(prompt_ids, full_ids)
    if len(full_ids) - boundary <= 0 or len(full_ids) < 2:
        return None
    mask = [0.0] * boundary + [1.0] * (len(full_ids) - boundary)
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(full_ids[:-1]),
        loss_fn_inputs={"target_tokens": full_ids[1:], "weights": mask[1:]},
    )
    return datum, sum(mask[1:])


def _mean_loss(metrics, active_tokens):
    return metrics.get("loss:sum", 0.0) / max(active_tokens, 1.0)


class TinkerSFTTrainer:
    name = "tinker-sft"
    consumes = ("sft",)
    version = "1"

    def train(self, dataset_path, out_dir, config):
        _require_tinker(self.name)
        import tinker
        from tinker import types

        model = config.get("model", DEFAULT_MODEL)
        epochs = config.get("epochs", 1)
        batch_size = config.get("batch_size", 4)
        learning_rate = config.get("learning_rate", 1e-4)
        rank = config.get("lora_rank", 32)

        rows = _read_rows(dataset_path)
        if config.get("max_rows"):
            rows = rows[:config["max_rows"]]

        service = tinker.ServiceClient()
        training = service.create_lora_training_client(base_model=model, rank=rank)
        tokenizer = training.get_tokenizer()

        built = []
        for row in rows:
            messages = row.get("messages")
            if not messages or len(messages) < 2:
                continue
            datum = _sft_datum(types, tokenizer, messages)
            if datum is not None:
                built.append(datum)
        if not built:
            raise ValueError("tinker-sft: no trainable rows (need messages with a non-empty assistant turn)")

        losses = []
        step = 0
        for _ in range(epochs):
            for batch in _batches(built, batch_size):
                data = [datum for datum, _ in batch]
                active = sum(count for _, count in batch)
                forward = training.forward_backward(data, "cross_entropy").result()
                training.optim_step(types.AdamParams(learning_rate=learning_rate)).result()
                losses.append(_mean_loss(forward.metrics, active))
                step += 1

        state = training.save_state(name="episodic-sft", overwrite=True).result()
        sampler = training.save_weights_for_sampler(name=f"episodic-sft-{step}").result()
        return {
            "backend": "tinker",
            "base_model": model,
            "method": "lora",
            "lora_rank": rank,
            "examples": len(built),
            "epochs": epochs,
            "steps": step,
            "final_loss": losses[-1] if losses else None,
            "mean_loss": sum(losses) / len(losses) if losses else None,
            "loss_curve": losses,
            "state_path": state.path,
            "sampler_path": sampler.path,
        }


def _grpo_prompts(rows):
    prompts = []
    seen = set()
    for row in rows:
        user = next((m for m in (row.get("messages") or []) if m.get("role") == "user"), None)
        if not user:
            continue
        content = user.get("content", "")
        if not content or content in seen:
            continue
        seen.add(content)
        prompts.append({"user": user, "meta": row.get("meta")})
    return prompts


def _resolve_reward(config):
    import importlib

    funcs = []
    for ref in config.get("reward_funcs", []):
        module_name, _, attr = ref.partition(":")
        funcs.append(getattr(importlib.import_module(module_name), attr))
    if not funcs:
        from .rewards import action_format_reward
        funcs = [action_format_reward]

    def score(prompt_text, completion_text, meta=None):
        total = 0.0
        for func in funcs:
            out = func(prompts=[prompt_text], completions=[completion_text], meta=[meta])
            total += float(out[0]) if out else 0.0
        return total / len(funcs)

    return score


def _group_advantages(rewards):
    n = len(rewards)
    if n == 0:
        return None
    mean = sum(rewards) / n
    std = (sum((r - mean) ** 2 for r in rewards) / n) ** 0.5
    if std < 1e-6:
        return None
    return [(r - mean) / std for r in rewards]


def _grpo_datum(types, prompt_ids, completion_tokens, completion_logprobs, advantage, length_normalize):
    full = prompt_ids + completion_tokens
    prompt_len = len(prompt_ids)
    per_token = advantage / len(completion_tokens) if length_normalize else advantage
    advantages = [0.0] * (prompt_len - 1) + [per_token] * len(completion_tokens)
    logprobs = [0.0] * (prompt_len - 1) + list(completion_logprobs)
    return types.Datum(
        model_input=types.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={"target_tokens": full[1:], "advantages": advantages, "logprobs": logprobs},
    )


class TinkerGRPOTrainer:
    name = "tinker-grpo"
    consumes = ("sft",)
    version = "1"

    def train(self, dataset_path, out_dir, config):
        _require_tinker(self.name)
        import tinker
        from tinker import types

        model = config.get("model", DEFAULT_MODEL)
        rank = config.get("lora_rank", 32)
        group_size = config.get("group_size", 8)
        learning_rate = config.get("learning_rate", 1e-5)
        max_tokens = config.get("max_tokens", 128)
        temperature = config.get("temperature", 1.0)
        prompts_per_step = config.get("prompts_per_step", 2)
        length_normalize = config.get("length_normalize", True)
        init_state = config.get("init_state")

        rows = _read_rows(dataset_path)
        prompts = _grpo_prompts(rows)
        if not prompts:
            raise ValueError("tinker-grpo: no prompts (need SFT rows with a user turn)")
        score = _resolve_reward(config)

        service = tinker.ServiceClient()
        if init_state:
            training = service.create_training_client_from_state(init_state)
        else:
            training = service.create_lora_training_client(base_model=model, rank=rank)
        tokenizer = training.get_tokenizer()

        chunks = list(_batches(prompts, prompts_per_step))
        if config.get("max_steps"):
            chunks = chunks[:config["max_steps"]]

        history = []
        sampler_ttl = config.get("sampler_ttl_seconds", 3600)
        for step, chunk in enumerate(chunks):
            sampler = training.save_weights_for_sampler(name=f"grpo-{step}", ttl_seconds=sampler_ttl).result()
            sampling = service.create_sampling_client(model_path=sampler.path)
            data = []
            step_rewards = []
            active_groups = 0
            for prompt in chunk:
                prompt_ids = _token_ids(tokenizer.apply_chat_template(
                    [prompt["user"]], add_generation_prompt=True, tokenize=True))
                response = sampling.sample(
                    prompt=types.ModelInput.from_ints(prompt_ids),
                    num_samples=group_size,
                    sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=temperature),
                ).result()
                group = []
                for sequence in response.sequences:
                    tokens = list(sequence.tokens)
                    if not tokens:
                        continue
                    action_text = _strip_reasoning(tokenizer.decode(tokens))
                    reward = score(prompt["user"].get("content", ""), action_text, prompt.get("meta"))
                    group.append((tokens, list(sequence.logprobs), reward))
                if not group:
                    continue
                rewards = [reward for _, _, reward in group]
                step_rewards.extend(rewards)
                advantages = _group_advantages(rewards)
                if advantages is None:
                    continue
                active_groups += 1
                for (tokens, logprobs, _), advantage in zip(group, advantages):
                    data.append(_grpo_datum(types, prompt_ids, tokens, logprobs, advantage, length_normalize))

            entry = {
                "step": step,
                "prompts": len(chunk),
                "samples": len(step_rewards),
                "reward_mean": (sum(step_rewards) / len(step_rewards)) if step_rewards else None,
                "reward_max": max(step_rewards) if step_rewards else None,
                "active_groups": active_groups,
            }
            if data:
                forward = training.forward_backward(data, "importance_sampling").result()
                training.optim_step(types.AdamParams(learning_rate=learning_rate)).result()
                entry["loss_sum"] = forward.metrics.get("loss:sum")
                entry["updated"] = True
            else:
                entry["updated"] = False
            history.append(entry)

        state = training.save_state(name="episodic-grpo", overwrite=True).result()
        sampler = training.save_weights_for_sampler(name="episodic-grpo-final").result()
        return {
            "backend": "tinker",
            "base_model": model,
            "method": "grpo",
            "warm_start": bool(init_state),
            "group_size": group_size,
            "prompts": len(prompts),
            "steps": len(history),
            "updates": sum(1 for entry in history if entry.get("updated")),
            "history": history,
            "state_path": state.path,
            "sampler_path": sampler.path,
        }


register(TinkerSFTTrainer())
register(TinkerGRPOTrainer())
