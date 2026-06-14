import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import stages


def _ensure_out(out):
    Path(out).mkdir(parents=True, exist_ok=True)
    return Path(out)


def cmd_quality_filter(args):
    episodes = stages.load_episodes(args.home)
    if not episodes:
        print("No episodes found.")
        return None, None
    good, stats = stages.quality_filter(episodes, args.min_composite)
    out = _ensure_out(args.out)
    stages.write_jsonl(out / "quality.jsonl", good)
    print(f"quality-filter: kept={stats['kept']} dropped={stats['dropped']} total={stats['total']}")
    return good, episodes


def cmd_sft(args, good=None):
    if good is None:
        episodes = stages.load_episodes(args.home)
        if not episodes:
            print("No episodes found.")
            return []
        good, _ = stages.quality_filter(episodes, args.min_composite)
    rows = stages.sft_dataset(good)
    out = _ensure_out(args.out)
    stages.write_jsonl(out / "sft.jsonl", rows)
    print(f"sft: rows={len(rows)}")
    return rows


def cmd_pref_pairs(args, episodes=None):
    if episodes is None:
        episodes = stages.load_episodes(args.home)
        if not episodes:
            print("No episodes found.")
            return []
    rows = stages.preference_pairs(episodes)
    out = _ensure_out(args.out)
    stages.write_jsonl(out / "pref_pairs.jsonl", rows)
    print(f"pref-pairs: pairs={len(rows)}")
    return rows


def cmd_reward_model(args, episodes=None):
    if episodes is None:
        episodes = stages.load_episodes(args.home)
        if not episodes:
            print("No episodes found.")
            return None
    model = stages.reward_model_train(episodes)
    out = _ensure_out(args.out)
    (out / "reward_model.json").write_text(
        json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"reward-model: features={len(model['features'])} r2={model['r2']}")
    return model


def cmd_rl_batches(args, episodes=None):
    if episodes is None:
        episodes = stages.load_episodes(args.home)
        if not episodes:
            print("No episodes found.")
            return []
    transitions = stages.rl_batches(episodes)
    out = _ensure_out(args.out)
    stages.write_jsonl(out / "rl_batches.jsonl", transitions)
    print(f"rl-batches: transitions={len(transitions)}")
    return transitions


def cmd_eval(args, episodes=None, model=None):
    if episodes is None:
        episodes = stages.load_episodes(args.home)
        if not episodes:
            print("No episodes found.")
            return {}
    if model is None:
        model_path = Path(args.out) / "reward_model.json"
        if model_path.exists():
            model = json.loads(model_path.read_text(encoding="utf-8"))
        else:
            model = stages.reward_model_train(episodes)
    result = stages.evaluate(episodes, model)
    out = _ensure_out(args.out)
    (out / "eval.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"eval: n={result['n']} pearson={result['pearson']} mae={result['mean_abs_err']}")
    return result


def cmd_all(args):
    episodes = stages.load_episodes(args.home)
    if not episodes:
        print("No episodes found in store. Run the agent first to collect episodes.")
        return

    good, stats = stages.quality_filter(episodes, args.min_composite)
    out = _ensure_out(args.out)
    stages.write_jsonl(out / "quality.jsonl", good)
    print(f"quality-filter: kept={stats['kept']} dropped={stats['dropped']} total={stats['total']}")

    sft_rows = stages.sft_dataset(good)
    stages.write_jsonl(out / "sft.jsonl", sft_rows)
    print(f"sft: rows={len(sft_rows)}")

    pairs = stages.preference_pairs(episodes)
    stages.write_jsonl(out / "pref_pairs.jsonl", pairs)
    print(f"pref-pairs: pairs={len(pairs)}")

    model = stages.reward_model_train(episodes)
    (out / "reward_model.json").write_text(
        json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"reward-model: features={len(model['features'])} r2={model['r2']}")

    transitions = stages.rl_batches(episodes)
    stages.write_jsonl(out / "rl_batches.jsonl", transitions)
    print(f"rl-batches: transitions={len(transitions)}")

    result = stages.evaluate(episodes, model)
    (out / "eval.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"eval: n={result['n']} pearson={result['pearson']} mae={result['mean_abs_err']}")


def main():
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--home", default=None)
    parent.add_argument("--out", default=default_out)
    parent.add_argument("--min-composite", type=float, default=0.5)

    parser = argparse.ArgumentParser(prog="pipeline", parents=[parent])
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("quality-filter", parents=[parent])
    sub.add_parser("sft", parents=[parent])
    sub.add_parser("pref-pairs", parents=[parent])
    sub.add_parser("reward-model", parents=[parent])
    sub.add_parser("rl-batches", parents=[parent])
    sub.add_parser("eval", parents=[parent])
    sub.add_parser("all", parents=[parent])

    args = parser.parse_args()

    if args.command == "quality-filter":
        cmd_quality_filter(args)
    elif args.command == "sft":
        cmd_sft(args)
    elif args.command == "pref-pairs":
        cmd_pref_pairs(args)
    elif args.command == "reward-model":
        cmd_reward_model(args)
    elif args.command == "rl-batches":
        cmd_rl_batches(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "all":
        cmd_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
