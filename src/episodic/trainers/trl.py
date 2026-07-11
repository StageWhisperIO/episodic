import json
import math

from . import register, TrainerUnavailable

HINT = "TRL backend needs extras: pip install 'episodic[trl]' (torch, transformers, trl, datasets, accelerate)"
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _require_trl():
    try:
        import datasets  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
    except ImportError as exc:
        raise TrainerUnavailable("trl", HINT) from exc


def _read_rows(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _training_kwargs(config):
    return {
        "num_train_epochs": config.get("epochs", 1),
        "per_device_train_batch_size": config.get("batch_size", 1),
        "gradient_accumulation_steps": config.get("grad_accum", 1),
        "learning_rate": config.get("learning_rate", 2e-5),
        "max_steps": config.get("max_steps", -1),
        "logging_steps": config.get("logging_steps", 1),
        "report_to": [],
    }


class TRLSFTTrainer:
    name = "trl-sft"
    consumes = ("sft",)
    version = "1"

    def train(self, dataset_path, out_dir, config):
        _require_trl()
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        model_name = config.get("model", DEFAULT_MODEL)
        rows = _read_rows(dataset_path)
        dataset = Dataset.from_list([{"messages": row["messages"]} for row in rows])

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)

        args = SFTConfig(output_dir=out_dir, **_training_kwargs(config))
        trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
        outcome = trainer.train()
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        return {
            "model_dir": out_dir,
            "base_model": model_name,
            "examples": len(rows),
            "steps": int(outcome.global_step),
            "train_loss": float(outcome.training_loss),
        }


class TRLDPOTrainer:
    name = "trl-dpo"
    consumes = ("dpo",)
    version = "1"

    def train(self, dataset_path, out_dir, config):
        _require_trl()
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer

        model_name = config.get("model", DEFAULT_MODEL)
        rows = _read_rows(dataset_path)
        dataset = Dataset.from_list([
            {"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]}
            for row in rows
        ])

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)

        args = DPOConfig(output_dir=out_dir, beta=config.get("beta", 0.1), **_training_kwargs(config))
        trainer = DPOTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
        outcome = trainer.train()
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        return {
            "model_dir": out_dir,
            "base_model": model_name,
            "pairs": len(rows),
            "steps": int(outcome.global_step),
            "train_loss": float(outcome.training_loss),
        }


def _prompt_from_messages(messages):
    for message in messages:
        if message.get("role") == "user":
            return message.get("content", "")
    return messages[0].get("content", "") if messages else ""


def _resolve_reward_funcs(config):
    import importlib

    funcs = []
    for ref in config.get("reward_funcs", []):
        module_name, _, attr = ref.partition(":")
        funcs.append(getattr(importlib.import_module(module_name), attr))
    if config.get("reward_model"):
        funcs.append(config["reward_model"])
    if not funcs:
        raise ValueError(
            "trl-grpo requires config.reward_model (path to a trl-reward model) "
            "or config.reward_funcs (list of 'module:attr' callables)"
        )
    return funcs


class TRLRewardTrainer:
    name = "trl-reward"
    consumes = ("dpo",)
    version = "1"

    def train(self, dataset_path, out_dir, config):
        _require_trl()
        from datasets import Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from trl import RewardConfig, RewardTrainer

        model_name = config.get("model", DEFAULT_MODEL)
        rows = _read_rows(dataset_path)
        dataset = Dataset.from_list([
            {"chosen": row["chosen"], "rejected": row["rejected"]} for row in rows
        ])

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

        args = RewardConfig(output_dir=out_dir, **_training_kwargs(config))
        trainer = RewardTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
        outcome = trainer.train()
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        return {
            "model_dir": out_dir,
            "base_model": model_name,
            "pairs": len(rows),
            "steps": int(outcome.global_step),
            "train_loss": float(outcome.training_loss),
        }


def resolve_grpo_generations(config, dataset_rows, num_processes=1):
    configured = config.get("num_generations", 4)
    if isinstance(configured, bool) or not isinstance(configured, int) or configured < 1:
        raise ValueError(f"num_generations must be a positive integer, got {configured!r}")

    per_device_batch = config.get("batch_size", 1)
    grad_accum = config.get("grad_accum", 1)
    effective_batch = max(1, per_device_batch * num_processes * grad_accum)
    cap = min(effective_batch, dataset_rows) if dataset_rows else effective_batch

    num_generations = max(2, min(configured, max(1, cap)))
    base = max(1, per_device_batch * num_processes)
    generation_batch_size = base * num_generations // math.gcd(base, num_generations)
    return num_generations, generation_batch_size


class TRLGRPOTrainer:
    name = "trl-grpo"
    consumes = ("sft",)
    version = "1"

    def train(self, dataset_path, out_dir, config):
        _require_trl()
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer

        model_name = config.get("model", DEFAULT_MODEL)
        rows = _read_rows(dataset_path)
        dataset = Dataset.from_list([{"prompt": _prompt_from_messages(row["messages"])} for row in rows])
        reward_funcs = _resolve_reward_funcs(config)
        num_generations, generation_batch_size = resolve_grpo_generations(config, len(rows))

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)

        args = GRPOConfig(
            output_dir=out_dir,
            num_generations=num_generations,
            generation_batch_size=generation_batch_size,
            **_training_kwargs(config),
        )
        trainer = GRPOTrainer(
            model=model, args=args, train_dataset=dataset,
            reward_funcs=reward_funcs, processing_class=tokenizer,
        )
        outcome = trainer.train()
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        return {
            "model_dir": out_dir,
            "base_model": model_name,
            "prompts": len(rows),
            "num_generations": args.num_generations,
            "steps": int(outcome.global_step),
            "train_loss": float(outcome.training_loss),
        }


def _completion_logprobs_torch(torch, policy, full_ids, completion_len):
    logits = policy(full_ids[:-1].unsqueeze(0)).logits[0]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    targets = full_ids[1:]
    token_logprobs = logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return token_logprobs[-completion_len:]


class TRLSAOTrainer:
    name = "trl-sao"
    consumes = ("sft",)
    version = "1"

    def train(self, dataset_path, out_dir, config):
        from collections import deque

        from . import sao
        from .critic import _pick_device, _require_torch, build_critic

        _require_torch(self.name)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = config.get("model", DEFAULT_MODEL)
        device = config.get("device") or _pick_device(torch)
        learning_rate = config.get("learning_rate", 1e-5)
        max_tokens = config.get("max_tokens", 128)
        temperature = config.get("temperature", 1.0)
        batch_size = config.get("batch_size", 1)
        epsilon_low = config.get("epsilon_low", sao.DEFAULT_EPSILON_LOW)
        epsilon_high = config.get("epsilon_high", sao.DEFAULT_EPSILON_HIGH)
        baseline_window = config.get("baseline_window", sao.DEFAULT_BASELINE_WINDOW)
        refresh_steps = max(1, config.get("sampler_refresh_steps", sao.DEFAULT_SAMPLER_REFRESH_STEPS))
        length_normalize = config.get("length_normalize", True)
        critic_updates = config.get("critic_updates", 2)

        rows = _read_rows(dataset_path)
        prompts = sao.unique_prompts(rows)
        if not prompts:
            raise ValueError("trl-sao: no prompts (need SFT rows with a user turn)")
        score = sao.resolve_reward(config)
        value_model = build_critic(config, self.name)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        policy = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)

        chunks = [prompts[start:start + batch_size] for start in range(0, len(prompts), batch_size)]
        if config.get("max_steps"):
            chunks = chunks[:config["max_steps"]]

        prompt_windows = {}
        global_window = deque(maxlen=baseline_window)
        pending = []
        history = []
        for step, chunk in enumerate(chunks):
            if step % refresh_steps == 0:
                pending = []
                policy.eval()
                with torch.no_grad():
                    for window_chunk in chunks[step:step + refresh_steps]:
                        for prompt in window_chunk:
                            rendered = tokenizer.apply_chat_template(
                                [prompt["user"]], add_generation_prompt=True, tokenize=True)
                            ids = rendered["input_ids"] if hasattr(rendered, "keys") else rendered
                            if ids and isinstance(ids[0], (list, tuple)):
                                ids = ids[0]
                            prompt_ids = torch.tensor([int(i) for i in ids], dtype=torch.long, device=device)
                            generated = policy.generate(
                                prompt_ids.unsqueeze(0), do_sample=True, temperature=temperature,
                                max_new_tokens=max_tokens, min_new_tokens=1,
                                pad_token_id=tokenizer.pad_token_id,
                            )[0]
                            completion = generated[prompt_ids.shape[0]:]
                            if completion.shape[0] == 0:
                                continue
                            rollout_logprobs = _completion_logprobs_torch(
                                torch, policy, generated, completion.shape[0])
                            pending.append({
                                "prompt": prompt,
                                "full_ids": generated,
                                "completion_len": int(completion.shape[0]),
                                "rollout_logprobs": rollout_logprobs.cpu(),
                                "text": tokenizer.decode(completion, skip_special_tokens=True),
                            })

            take = min(len(chunk), len(pending))
            batch, pending = pending[:take], pending[take:]
            step_rewards = []
            step_advantages = []
            scored = []
            for rollout in batch:
                key = rollout["prompt"]["user"].get("content", "")
                reward = score(key, rollout["text"], rollout["prompt"].get("meta"))
                if value_model is not None:
                    baseline = value_model.value([key])[0]
                else:
                    baseline = sao.running_baseline(key, prompt_windows, global_window)
                    prompt_windows.setdefault(key, deque(maxlen=baseline_window)).append(reward)
                    global_window.append(reward)
                advantage = reward - baseline
                step_rewards.append(reward)
                step_advantages.append(advantage)
                per_token = advantage / rollout["completion_len"] if length_normalize else advantage
                scored.append((rollout, key, per_token))

            entry = {
                "step": step,
                "prompts": len(chunk),
                "samples": len(scored),
                "reward_mean": (sum(step_rewards) / len(step_rewards)) if step_rewards else None,
                "advantage_mean": (sum(step_advantages) / len(step_advantages)) if step_advantages else None,
            }
            if scored:
                policy.train()
                losses = []
                masked_tokens = 0
                total_tokens = 0
                optimizer.zero_grad()
                for rollout, key, per_token in scored:
                    current_logprobs = _completion_logprobs_torch(
                        torch, policy, rollout["full_ids"], rollout["completion_len"])
                    advantages, masked = sao.dis_mask(
                        [per_token] * rollout["completion_len"],
                        [float(v) for v in current_logprobs.detach().cpu()],
                        [float(v) for v in rollout["rollout_logprobs"]],
                        epsilon_low, epsilon_high)
                    masked_tokens += masked
                    total_tokens += rollout["completion_len"]
                    ratio = torch.exp(
                        current_logprobs.detach() - rollout["rollout_logprobs"].to(device))
                    coef = ratio * torch.tensor(advantages, dtype=ratio.dtype, device=device)
                    losses.append(-(coef * current_logprobs).sum())
                loss = torch.stack(losses).sum() / len(scored)
                loss.backward()
                optimizer.step()
                entry["loss"] = float(loss.detach())
                entry["clip_fraction"] = masked_tokens / total_tokens if total_tokens else 0.0
                entry["updated"] = True
                if value_model is not None:
                    texts = [key for _, key, _ in scored]
                    critic_losses = [value_model.update(texts, step_rewards) for _ in range(critic_updates)]
                    entry["critic_loss"] = critic_losses[-1]
            else:
                entry["updated"] = False
            history.append(entry)

        policy.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        clip_fractions = [entry["clip_fraction"] for entry in history if "clip_fraction" in entry]
        rewards = [entry["reward_mean"] for entry in history if entry.get("reward_mean") is not None]
        return {
            "model_dir": out_dir,
            "base_model": model_name,
            "method": "sao",
            "device": device,
            "baseline": "critic" if value_model is not None else "running_mean",
            "critic_model": config.get("critic_model"),
            "epsilon_low": epsilon_low,
            "epsilon_high": epsilon_high,
            "sampler_refresh_steps": refresh_steps,
            "prompts": len(prompts),
            "steps": len(history),
            "updates": sum(1 for entry in history if entry.get("updated")),
            "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
            "clip_fraction": (sum(clip_fractions) / len(clip_fractions)) if clip_fractions else None,
            "history": history,
        }


register(TRLSFTTrainer())
register(TRLDPOTrainer())
register(TRLRewardTrainer())
register(TRLGRPOTrainer())
register(TRLSAOTrainer())
