from pathlib import Path

from .. import exporters, trainers
from ..core import rubric
from ..trainers import critic as critic_mod
from ..trainers.trl import _read_rows

DEFAULT_TYPE = "rubric_judge"
DEFAULT_REWARD_MODEL = critic_mod.DEFAULT_CRITIC_MODEL


def _evaluator_config(config):
    return dict(config.get("evaluator") or {})


def _passthrough(judge):
    return {"type": DEFAULT_TYPE, "judge": judge, "critic": None}


def build(config, judge):
    evaluator_config = _evaluator_config(config)
    etype = evaluator_config.get("type", DEFAULT_TYPE)

    if etype == "local_critic":
        critic = critic_mod.build_critic(evaluator_config, "local-critic-evaluator")
        if critic is None:
            return _passthrough(judge)
        return {"type": "local_critic", "judge": rubric.critic_judge(critic), "critic": critic}

    if etype == "trl_reward":
        model_dir = evaluator_config.get("model_dir")
        if not model_dir:
            return _passthrough(judge)
        critic = critic_mod.load_trl_reward_model(model_dir, device=evaluator_config.get("critic_device"))
        return {"type": "trl_reward", "judge": rubric.critic_judge(critic), "critic": critic,
                "model_dir": model_dir}

    return _passthrough(judge)


def _refresh_local_critic(state, evaluator_config, train_episodes, epoch_dir):
    critic = state.get("critic")
    if critic is None:
        critic_config = dict(evaluator_config)
        critic_config.setdefault("critic_model", DEFAULT_REWARD_MODEL)
        critic = critic_mod.build_critic(critic_config, "local-critic-evaluator")
    if critic is None:
        return state

    export_result = exporters.export(train_episodes, "reward", str(epoch_dir / "dataset"))
    rows = _read_rows(export_result["files"][0])
    pairs = critic_mod.pretrain_pairs_from_reward_rows(rows)
    if pairs:
        critic.pretrain(
            pairs,
            epochs=evaluator_config.get("critic_epochs", 1),
            batch_size=evaluator_config.get("critic_batch_size", 8),
        )
    return {"type": "local_critic", "judge": rubric.critic_judge(critic), "critic": critic}


def _refresh_trl_reward(state, evaluator_config, train_episodes, epoch_dir, start):
    export_result = exporters.export(train_episodes, "dpo", str(epoch_dir / "dataset"))
    dpo_path = export_result["files"][0]
    train_out = str(epoch_dir / "model")

    critic_train_config = dict(evaluator_config.get("train_config") or {})
    critic_train_config.setdefault("model", evaluator_config.get("model", DEFAULT_REWARD_MODEL))

    train_manifest = trainers.train("trl-reward", dpo_path, train_out, critic_train_config, cwd=start)
    model_dir = (train_manifest.get("result") or {}).get("model_dir") or train_out
    critic = critic_mod.load_trl_reward_model(model_dir, device=evaluator_config.get("critic_device"))
    return {"type": "trl_reward", "judge": rubric.critic_judge(critic), "critic": critic, "model_dir": model_dir}


def refresh(state, config, train_episodes, epoch_index, out, start):
    evaluator_config = _evaluator_config(config)
    etype = evaluator_config.get("type", DEFAULT_TYPE)
    if etype not in ("local_critic", "trl_reward") or not train_episodes:
        return state

    epoch_dir = Path(out) / "evaluator" / f"epoch_{epoch_index}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    if etype == "local_critic":
        return _refresh_local_critic(state, evaluator_config, train_episodes, epoch_dir)
    return _refresh_trl_reward(state, evaluator_config, train_episodes, epoch_dir, start)
