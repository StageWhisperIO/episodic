import os
from collections import deque

from . import register, TrainerUnavailable
from . import sao
from .trl import _read_rows

HINT = (
    "Tinker backend needs the SDK and an API key: pip install tinker, then "
    "export TINKER_API_KEY. Training runs on Thinking Machines' GPUs "
    "(https://tinker-console.thinkingmachines.ai)."
)
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_SAMPLER_TTL_SECONDS = 7 * 24 * 3600
STEP_SAMPLER_TTL_SECONDS = 3600
_MISSING = object()


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


def _open_training(name, config):
    _require_tinker(name)
    import tinker
    from tinker import types

    model = config.get("model", DEFAULT_MODEL)
    rank = config.get("lora_rank", 32)
    init_state = config.get("init_state")

    service = tinker.ServiceClient()
    if init_state:
        training = service.create_training_client_from_state(init_state)
    else:
        training = service.create_lora_training_client(base_model=model, rank=rank)
    return service, training, types, model, rank


def _checkpoint(training, state_name, sampler_name, sampler_ttl_seconds=None):
    state = training.save_state(name=state_name, overwrite=True).result()
    sampler = training.save_weights_for_sampler(name=sampler_name, ttl_seconds=sampler_ttl_seconds).result()
    return {
        "state_path": state.path,
        "sampler_path": sampler.path,
        "sampler_ttl_seconds": sampler_ttl_seconds,
    }


class TinkerSFTTrainer:
    name = "tinker-sft"
    consumes = ("sft",)
    version = "2"

    def train(self, dataset_path, out_dir, config):
        service, training, types, model, rank = _open_training(self.name, config)

        epochs = config.get("epochs", 1)
        batch_size = config.get("batch_size", 4)
        learning_rate = config.get("learning_rate", 1e-4)

        rows = _read_rows(dataset_path)
        if config.get("max_rows"):
            rows = rows[:config["max_rows"]]

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

        result = {
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
        }
        result.update(_checkpoint(
            training, "episodic-sft", f"episodic-sft-{step}",
            config.get("sampler_ttl_seconds", DEFAULT_SAMPLER_TTL_SECONDS)))
        return result


def _sao_datum(types, prompt_ids, completion_tokens, completion_logprobs, completion_advantages):
    full = prompt_ids + completion_tokens
    prompt_len = len(prompt_ids)
    advantages = [0.0] * (prompt_len - 1) + list(completion_advantages)
    logprobs = [0.0] * (prompt_len - 1) + list(completion_logprobs)
    return types.Datum(
        model_input=types.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={"target_tokens": full[1:], "advantages": advantages, "logprobs": logprobs},
    )


def _ce_datum(types, prompt_ids, completion_tokens):
    full = prompt_ids + completion_tokens
    weights = [0.0] * (len(prompt_ids) - 1) + [1.0] * len(completion_tokens)
    return types.Datum(
        model_input=types.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={"target_tokens": full[1:], "weights": weights},
    )


def _tensor_values(tensor):
    values = tensor.tolist() if hasattr(tensor, "tolist") else list(tensor)
    if values and isinstance(values[0], (list, tuple)):
        values = values[0]
    return [float(v) for v in values]


def _completion_logprobs(output, index, count):
    tensors = output.loss_fn_outputs[index]
    for key in tensors:
        if "logprob" in key:
            return _tensor_values(tensors[key])[-count:]
    raise ValueError(f"tinker-sao: forward pass returned no logprobs (keys: {sorted(tensors)})")


class TinkerSAOTrainer:
    name = "tinker-sao"
    consumes = ("sft",)
    version = "1"

    def train(self, dataset_path, out_dir, config):
        service, training, types, model, _rank = _open_training(self.name, config)

        learning_rate = config.get("learning_rate", 1e-5)
        max_tokens = config.get("max_tokens", 128)
        temperature = config.get("temperature", 1.0)
        batch_size = config.get("batch_size", 1)
        epsilon_low = config.get("epsilon_low", sao.DEFAULT_EPSILON_LOW)
        epsilon_high = config.get("epsilon_high", sao.DEFAULT_EPSILON_HIGH)
        baseline_window = config.get("baseline_window", sao.DEFAULT_BASELINE_WINDOW)
        refresh_steps = max(1, config.get("sampler_refresh_steps", sao.DEFAULT_SAMPLER_REFRESH_STEPS))
        length_normalize = config.get("length_normalize", True)
        init_state = config.get("init_state")
        critic_updates = config.get("critic_updates", 2)
        ttl = config.get("sampler_ttl_seconds", _MISSING)
        step_ttl = STEP_SAMPLER_TTL_SECONDS if ttl is _MISSING else ttl
        final_ttl = DEFAULT_SAMPLER_TTL_SECONDS if ttl is _MISSING else ttl

        rows = _read_rows(dataset_path)
        prompts = sao.unique_prompts(rows)
        if not prompts:
            raise ValueError("tinker-sao: no prompts (need SFT rows with a user turn)")
        score = sao.resolve_reward(config)
        from .critic import build_critic
        value_model = build_critic(config, self.name)

        tokenizer = training.get_tokenizer()

        chunks = list(_batches(prompts, batch_size))
        if config.get("max_steps"):
            chunks = chunks[:config["max_steps"]]

        prompt_windows = {}
        global_window = deque(maxlen=baseline_window)
        sampling = None
        sampler_path = None
        history = []
        for step, chunk in enumerate(chunks):
            if sampling is None or step % refresh_steps == 0:
                sampler = training.save_weights_for_sampler(name=f"sao-{step}", ttl_seconds=step_ttl).result()
                sampling = service.create_sampling_client(model_path=sampler.path)
                sampler_path = sampler.path
            rollouts = []
            step_rewards = []
            step_advantages = []
            for prompt in chunk:
                prompt_ids = _token_ids(tokenizer.apply_chat_template(
                    [prompt["user"]], add_generation_prompt=True, tokenize=True))
                response = sampling.sample(
                    prompt=types.ModelInput.from_ints(prompt_ids),
                    num_samples=1,
                    sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=temperature),
                ).result()
                sequence = response.sequences[0] if response.sequences else None
                if sequence is None:
                    continue
                tokens = list(sequence.tokens)
                if not tokens:
                    continue
                action_text = _strip_reasoning(tokenizer.decode(tokens))
                key = prompt["user"].get("content", "")
                reward = score(key, action_text, prompt.get("meta"))
                if value_model is not None:
                    baseline = value_model.value([key])[0]
                else:
                    baseline = sao.running_baseline(key, prompt_windows, global_window)
                    prompt_windows.setdefault(key, deque(maxlen=baseline_window)).append(reward)
                    global_window.append(reward)
                advantage = reward - baseline
                step_rewards.append(reward)
                step_advantages.append(advantage)
                per_token = advantage / len(tokens) if length_normalize else advantage
                rollouts.append((prompt_ids, tokens, list(sequence.logprobs), per_token, key))

            entry = {
                "step": step,
                "prompts": len(chunk),
                "samples": len(rollouts),
                "reward_mean": (sum(step_rewards) / len(step_rewards)) if step_rewards else None,
                "advantage_mean": (sum(step_advantages) / len(step_advantages)) if step_advantages else None,
                "sampler_path": sampler_path,
            }
            if rollouts:
                current = training.forward(
                    [_ce_datum(types, prompt_ids, tokens) for prompt_ids, tokens, _, _, _ in rollouts],
                    "cross_entropy",
                ).result()
                data = []
                masked_tokens = 0
                total_tokens = 0
                for index, (prompt_ids, tokens, rollout_logprobs, per_token, _) in enumerate(rollouts):
                    current_logprobs = _completion_logprobs(current, index, len(tokens))
                    advantages, masked = sao.dis_mask(
                        [per_token] * len(tokens), current_logprobs, rollout_logprobs,
                        epsilon_low, epsilon_high)
                    masked_tokens += masked
                    total_tokens += len(tokens)
                    data.append(_sao_datum(types, prompt_ids, tokens, rollout_logprobs, advantages))
                forward = training.forward_backward(data, "importance_sampling").result()
                training.optim_step(types.AdamParams(learning_rate=learning_rate)).result()
                entry["loss_sum"] = forward.metrics.get("loss:sum")
                entry["clip_fraction"] = masked_tokens / total_tokens if total_tokens else 0.0
                entry["updated"] = True
                if value_model is not None:
                    texts = [key for _, _, _, _, key in rollouts]
                    critic_losses = [value_model.update(texts, step_rewards) for _ in range(critic_updates)]
                    entry["critic_loss"] = critic_losses[-1]
            else:
                entry["updated"] = False
            history.append(entry)

        clip_fractions = [entry["clip_fraction"] for entry in history if "clip_fraction" in entry]
        rewards = [entry["reward_mean"] for entry in history if entry.get("reward_mean") is not None]
        result = {
            "backend": "tinker",
            "base_model": model,
            "method": "sao",
            "warm_start": bool(init_state),
            "prompts": len(prompts),
            "steps": len(history),
            "updates": sum(1 for entry in history if entry.get("updated")),
            "epsilon_low": epsilon_low,
            "epsilon_high": epsilon_high,
            "baseline": "critic" if value_model is not None else "running_mean",
            "critic_model": config.get("critic_model"),
            "baseline_window": baseline_window,
            "sampler_refresh_steps": refresh_steps,
            "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
            "clip_fraction": (sum(clip_fractions) / len(clip_fractions)) if clip_fractions else None,
            "history": history,
        }
        result.update(_checkpoint(training, "episodic-sao", "episodic-sao-final", final_ttl))
        return result


register(TinkerSFTTrainer())
register(TinkerSAOTrainer())
