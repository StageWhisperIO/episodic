import json

from .. import replay, trainers
from ..replay import modelrun
from ..worldmodel.validate import _local_clone_episode, _oracle_diff_runner, _unified_diff
from .gate import empty_runner


def bug_class(episode):
    labels = [label for label in episode.get("labels", []) if label != "swe"]
    return labels[0] if labels else "misc"


def stratified_split(episodes, per_class_held=1):
    by_class = {}
    for episode in sorted(episodes, key=lambda e: e["id"]):
        by_class.setdefault(bug_class(episode), []).append(episode)
    held, train = [], []
    for group in by_class.values():
        k = per_class_held if len(group) > per_class_held else max(1, len(group) // 3)
        held.extend(group[:k])
        train.extend(group[k:])
    return train, held


def build_sft(train, path):
    with open(path, "w") as fh:
        for episode in train:
            user = episode["intent"] + modelrun._DIFF_INSTRUCTION
            assistant = "```diff\n" + _unified_diff(episode) + "```"
            fh.write(json.dumps({"messages": [{"role": "user", "content": user},
                                              {"role": "assistant", "content": assistant}]}) + "\n")
    return path


def solved(episode, runner):
    local = _local_clone_episode(episode)
    replay.create_replay(local)
    replay_id = replay.replay_id_for(local)
    try:
        result = replay.run_replay(replay_id, "candidate", execute=True, runner=runner)
        return bool((result.get("tests") or {}).get("ok"))
    finally:
        replay.cleanup_replay(replay_id)


def measure_lift(held, base_runner_for, trained_runner_for):
    by_class = {}
    base_solved = trained_solved = 0
    for episode in held:
        cls = bug_class(episode)
        base = solved(episode, base_runner_for(episode))
        trained = solved(episode, trained_runner_for(episode))
        base_solved += base
        trained_solved += trained
        stats = by_class.setdefault(cls, {"held": 0, "base": 0, "trained": 0})
        stats["held"] += 1
        stats["base"] += int(base)
        stats["trained"] += int(trained)
    return {"held": len(held), "base_solved": base_solved, "trained_solved": trained_solved,
            "lift": trained_solved - base_solved, "by_class": by_class}


def oracle_vs_empty_lift(held):
    return measure_lift(held,
                        lambda episode: empty_runner,
                        lambda episode: _oracle_diff_runner(_unified_diff(episode)))


def _tinker_runners(model, result, max_tokens):
    import tinker

    from ..trainers import tinker as tk

    service = tinker.ServiceClient()
    rest = service.create_rest_client()
    base_client = service.create_lora_training_client(base_model=model, rank=1)
    base_path = base_client.save_weights_for_sampler(name="episodic-eval-base", ttl_seconds=7200).result().path
    trained_path = result["sampler_path"]
    state_path = result.get("state_path")
    base_sampler = tk.open_sampler(base_path, base_model=model)
    trained_sampler = tk.open_sampler(trained_path, base_model=model)

    def base_gen(model_name, messages):
        return tk.sample_text(base_sampler, messages, max_tokens=max_tokens)

    def trained_gen(model_name, messages):
        return tk.sample_text(trained_sampler, messages, max_tokens=max_tokens)

    def cleanup():
        for path in (trained_path, state_path, base_path):
            if not path:
                continue
            try:
                rest.delete_checkpoint_from_tinker_path(path)
            except Exception:
                pass

    return base_gen, trained_gen, cleanup


def _mlx_runners(model, result, max_tokens):
    from ..trainers.mlx import load_predictor

    adapters = result.get("model_dir")
    base_pred = load_predictor(model, adapter_path=None, max_tokens=max_tokens)
    trained_pred = load_predictor(model, adapter_path=adapters, max_tokens=max_tokens)

    def base_gen(model_name, messages):
        return base_pred(messages)

    def trained_gen(model_name, messages):
        return trained_pred(messages)

    return base_gen, trained_gen, lambda: None


def real_lift(train, held, *, backend, model, sft_path, out_dir, epochs=3, iters=400,
              lora_rank=32, batch_size=4, learning_rate=1e-4, max_tokens=768):
    build_sft(train, sft_path)
    if backend == "tinker":
        trainer_name = "tinker-sft"
        config = {"model": model, "lora_rank": lora_rank, "epochs": epochs,
                  "batch_size": batch_size, "learning_rate": learning_rate}
    else:
        trainer_name = "mlx-sft"
        config = {"model": model, "iters": iters, "batch_size": batch_size, "num_layers": 8,
                  "learning_rate": learning_rate, "max_seq_length": 1024, "valid_frac": 0.1}
    manifest = trainers.train(trainer_name, sft_path, out_dir, config)
    result = manifest["result"]

    if backend == "tinker":
        base_gen, trained_gen, cleanup = _tinker_runners(model, result, max_tokens)
    else:
        base_gen, trained_gen, cleanup = _mlx_runners(model, result, max_tokens)

    base_runner = modelrun.build_runner(base_gen)
    trained_runner = modelrun.build_runner(trained_gen)
    try:
        report = measure_lift(held, lambda episode: base_runner, lambda episode: trained_runner)
    finally:
        cleanup()
    report.update({"backend": backend, "model": model, "train": len(train),
                   "steps": result.get("steps"), "final_loss": result.get("final_loss")})
    return report
