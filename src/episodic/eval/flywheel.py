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


def build_sao_rows(train, path):
    with open(path, "w") as fh:
        for episode in train:
            user = episode["intent"] + modelrun._DIFF_INSTRUCTION
            fh.write(json.dumps({"messages": [{"role": "user", "content": user}],
                                 "meta": episode}) + "\n")
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
    from .gate import graded_score

    by_class = {}
    base_solved = trained_solved = 0
    base_frac = trained_frac = 0.0
    for episode in held:
        cls = bug_class(episode)
        base = graded_score(episode, base_runner_for(episode))
        trained = graded_score(episode, trained_runner_for(episode))
        base_solved += base["ok"]
        trained_solved += trained["ok"]
        base_frac += base["pass_fraction"]
        trained_frac += trained["pass_fraction"]
        stats = by_class.setdefault(cls, {"held": 0, "base": 0, "trained": 0})
        stats["held"] += 1
        stats["base"] += int(base["ok"])
        stats["trained"] += int(trained["ok"])
    n = len(held) or 1
    return {"held": len(held), "base_solved": base_solved, "trained_solved": trained_solved,
            "lift": trained_solved - base_solved, "by_class": by_class,
            "base_pass_fraction": round(base_frac / n, 3),
            "trained_pass_fraction": round(trained_frac / n, 3),
            "fraction_lift": round((trained_frac - base_frac) / n, 3)}


def oracle_vs_empty_lift(held):
    return measure_lift(held,
                        lambda episode: empty_runner,
                        lambda episode: _oracle_diff_runner(_unified_diff(episode)))


def learnable_band(episodes, generate, n=4):
    banded = []
    stats = []
    for episode in episodes:
        runner = modelrun.build_runner(generate)
        solves = sum(bool(solved(episode, runner)) for _ in range(n))
        stats.append({"id": episode["id"], "solves": solves, "n": n})
        if 0 < solves < n:
            banded.append(episode)
    return {"banded": banded, "stats": stats}


def agentic_runner_for(episode, generate, max_turns):
    from . import agentic

    root = (episode.get("repo_state") or {}).get("root")
    test_command, test_cwd = replay._resolve_test_command(episode, root)
    return agentic.build_agentic_runner(generate, test_command, max_turns=max_turns, test_cwd=test_cwd)


def tool_agent_runner_for(episode, generate, max_steps):
    from . import agentic

    root = (episode.get("repo_state") or {}).get("root")
    test_command, test_cwd = replay._resolve_test_command(episode, root)
    return agentic.build_tool_agent(generate, test_command, max_steps=max_steps, test_cwd=test_cwd)


def _completion_solves(episode, text):
    diff = modelrun.extract_diff(text)

    def runner(model, workspace, prompt_text):
        applied, log = modelrun.apply_diff(diff, workspace)
        return log, 0 if applied else 1

    return solved(episode, runner)


def rollout_and_filter(episodes, generate, k=4):
    rows = []
    solved_count = 0
    for episode in episodes:
        prompt = episode["intent"] + modelrun._DIFF_INSTRUCTION
        for _ in range(k):
            text = generate("policy", [{"role": "user", "content": prompt}])
            if _completion_solves(episode, text):
                rows.append({"messages": [{"role": "user", "content": prompt},
                                          {"role": "assistant", "content": text}]})
                solved_count += 1
                break
    return {"rows": rows, "solved": solved_count, "attempted": len(episodes)}


def write_rollout_sft(rollout_rows, path):
    with open(path, "w") as fh:
        for row in rollout_rows:
            fh.write(json.dumps(row) + "\n")
    return path


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
              lora_rank=32, batch_size=4, learning_rate=1e-4, max_tokens=768, agentic_turns=0,
              tool_steps=0):
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

    if tool_steps:
        base_for = lambda episode: tool_agent_runner_for(episode, base_gen, tool_steps)
        trained_for = lambda episode: tool_agent_runner_for(episode, trained_gen, tool_steps)
    elif agentic_turns:
        base_for = lambda episode: agentic_runner_for(episode, base_gen, agentic_turns)
        trained_for = lambda episode: agentic_runner_for(episode, trained_gen, agentic_turns)
    else:
        base_runner = modelrun.build_runner(base_gen)
        trained_runner = modelrun.build_runner(trained_gen)
        base_for = lambda episode: base_runner
        trained_for = lambda episode: trained_runner
    try:
        report = measure_lift(held, base_for, trained_for)
    finally:
        cleanup()
    report.update({"backend": backend, "model": model, "train": len(train),
                   "steps": result.get("steps"), "final_loss": result.get("final_loss"),
                   "agentic_turns": agentic_turns, "tool_steps": tool_steps})
    return report
