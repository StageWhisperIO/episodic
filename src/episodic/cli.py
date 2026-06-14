import argparse
import json
import sys

from . import paths, store, service
from .schema import EPISODE_SCHEMA, FEEDBACK_LABELS, validate_episode
from .core import summary as summary_mod


def _resolve_episode(args):
    if getattr(args, "episode", None):
        episode = store.get_episode(args.episode)
        if episode is None:
            _fail(f"episode '{args.episode}' not found")
        return episode
    episode = service.finalize_session(getattr(args, "session", None))
    if episode is None:
        _fail("no active session found; run `/trace start` or pass --episode")
    return episode


def _fail(message):
    print(f"episodic: {message}", file=sys.stderr)
    raise SystemExit(1)


def _print_json(value):
    print(json.dumps(value, indent=2, ensure_ascii=False))


def cmd_ingest(args):
    from .collector.hook import main as hook_main

    return hook_main()


def cmd_start(args):
    intent = args.intent or ""
    session_id = service.set_intent(intent, args.session)
    if not session_id:
        current = store.get_current()
        if not current:
            print("Episodic is capturing this session. Intent will attach on the first prompt.")
            if intent:
                _stash_pending_intent(intent)
            return 0
        session_id = service.set_intent(intent, current)
    print(f"Episodic tracing session {session_id}")
    if intent:
        print(f"Intent: {intent}")
    return 0


def _stash_pending_intent(intent):
    path = paths.home() / "pending_intent"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(intent, encoding="utf-8")


def cmd_summary(args):
    episode = _resolve_episode(args)
    report = summary_mod.summarize(episode)
    if args.json:
        _print_json(report)
    else:
        print(summary_mod.render_markdown(report))
    return 0


def cmd_mark(args):
    label = "_".join(args.label).lower().replace("-", "_")
    if label not in FEEDBACK_LABELS:
        _fail(f"unknown label '{label}'. choose from: {', '.join(FEEDBACK_LABELS)}")
    episode = service.add_feedback(label, args.note, args.session)
    if episode is None:
        _fail("no active session to label")
    print(f"Recorded feedback '{label}' on {episode['id']}")
    print(f"Composite reward is now {episode['reward_vector']['composite']}")
    return 0


def cmd_pr_notes(args):
    episode = _resolve_episode(args)
    report = summary_mod.summarize(episode)
    if args.json:
        _print_json({
            "title": report["suggested_pr_title"],
            "description": report["suggested_pr_description"],
        })
    else:
        print(f"# {report['suggested_pr_title']}\n")
        print(report["suggested_pr_description"])
    return 0


def cmd_export(args):
    from . import exporters

    if args.all:
        episodes = store.load_episodes()
    else:
        episodes = [_resolve_episode(args)]
    if not episodes:
        _fail("no episodes to export")
    out = args.out or str(paths.exports_dir())
    result = exporters.export(episodes, args.format, out)
    _print_json(result)
    return 0


def cmd_link(args):
    from . import github

    if args.refresh_all:
        return _refresh_all(args)

    episode = _resolve_episode(args)
    if args.refresh:
        new_outcome = github.refresh_outcome(episode, cwd=args.cwd)
        if not new_outcome:
            _fail("cannot refresh: no linked PR on this episode or gh unavailable")
        episode["outcome"] = new_outcome
        service.update_episode(episode)
        print(f"Refreshed {episode['id']} -> '{new_outcome['status']}'")
        _print_json(new_outcome)
        return 0

    outcome = github.link_episode(episode, pr=args.pr, auto=args.auto, cwd=args.cwd)
    updated = service.set_outcome(outcome, session_id_for_episode(episode))
    target = updated or episode
    print(f"Linked {episode['id']} -> outcome '{outcome['status']}'")
    _print_json(outcome)
    return 0


def _refresh_all(args):
    from . import github

    checked = 0
    changed = 0
    for episode in store.load_episodes():
        outcome = episode.get("outcome") or {}
        if not github.should_refresh(outcome):
            continue
        checked += 1
        new_outcome = github.refresh_outcome(episode, cwd=args.cwd)
        if new_outcome and new_outcome != outcome:
            episode["outcome"] = new_outcome
            service.update_episode(episode)
            changed += 1
            print(f"{episode['id']}: {outcome.get('status')} -> {new_outcome['status']} "
                  f"(ci={new_outcome.get('ci_status')})")
    print(f"refreshed {changed}/{checked} in-flight episode(s)")
    return 0


def cmd_regression(args):
    from .github import regression as regression_mod

    episodes = store.load_episodes()
    report = regression_mod.regression_report(args.commit, args.cwd or ".", episodes)
    if args.apply:
        report["applied"] = _apply_regression(report, episodes, args.fuzzy)
    _print_json(report)
    return 0


def _apply_regression(report, episodes, fuzzy):
    by_id = {episode["id"]: episode for episode in episodes}
    applied = []
    for implication in report["implicated"]:
        if implication["via"] == "file" and not fuzzy:
            continue
        episode = by_id.get(implication["episode_id"])
        if not episode:
            continue
        outcome = episode.setdefault("outcome", {})
        outcome["caused_regression"] = True
        commits = set(outcome.get("regression_commits") or [])
        commits.add(report["fix_commit"])
        outcome["regression_commits"] = sorted(commits)
        labels = episode.setdefault("labels", [])
        if "regression" not in labels:
            labels.append("regression")
        service.update_episode(episode)
        applied.append({
            "episode_id": episode["id"],
            "via": implication["via"],
            "composite": episode["reward_vector"]["composite"],
        })
    return applied


def session_id_for_episode(episode):
    for session_id in store.list_sessions():
        from .core.ids import episode_id_from_session

        if episode_id_from_session(session_id) == episode["id"]:
            return session_id
    return None


def cmd_replay(args):
    from . import replay

    if args.replay_command == "create":
        episode = _resolve_episode(args)
        manifest = replay.create_replay(episode)
        _print_json(manifest)
    elif args.replay_command == "run":
        result = replay.run_replay(args.replay, args.model)
        _print_json(result)
    else:
        _fail("use `replay create` or `replay run`")
    return 0


def cmd_list(args):
    rows = store.list_episodes()
    if args.json:
        _print_json(rows)
        return 0
    if not rows:
        print("No episodes captured yet.")
        return 0
    print(f"{'EPISODE':<18} {'OUTCOME':<10} {'REWARD':<7} {'EDITS':<6} INTENT")
    for row in rows:
        print(
            f"{row['id']:<18} {row['outcome']:<10} {row['composite_reward']:<7} "
            f"{row['file_edits']:<6} {row['intent'][:60]}"
        )
    return 0


def cmd_show(args):
    episode = store.get_episode(args.episode)
    if episode is None:
        _fail(f"episode '{args.episode}' not found")
    errors = validate_episode(episode)
    if args.validate:
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            raise SystemExit(1)
        print("valid")
        return 0
    _print_json(episode)
    return 0


def cmd_finalize(args):
    episode = service.finalize_session(args.session)
    if episode is None:
        _fail("no active session to finalize")
    print(f"Finalized {episode['id']}")
    return 0


def cmd_schema(args):
    if args.schema_command == "dump":
        target = paths.resolve_base() / "schemas" / "episode.schema.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(EPISODE_SCHEMA, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {target}")
    else:
        _print_json(EPISODE_SCHEMA)
    return 0


def cmd_train(args):
    from . import trainers

    if args.list:
        for name in trainers.available():
            trainer = trainers.get(name)
            print(f"{name:<12} consumes={','.join(trainer.consumes)}")
        return 0

    dataset = _materialize_dataset(args.dataset)
    config = _load_train_config(args.config)
    if args.model:
        config.setdefault("model", args.model)
    out = args.out or str(paths.exports_dir() / f"train-{args.trainer}")

    try:
        manifest = trainers.train(args.trainer, dataset, out, config)
    except trainers.TrainerUnavailable as exc:
        print(f"episodic: {exc.hint}", file=sys.stderr)
        print(f"dataset is ready at: {dataset}")
        print("install the backend, swap --trainer command, or hand the dataset to any trainer.")
        return 0
    _print_json(manifest)
    return 0


def _materialize_dataset(arg):
    if arg and arg != "-":
        return arg
    import tempfile

    data = sys.stdin.read()
    handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    handle.write(data)
    handle.close()
    return handle.name


def _load_train_config(arg):
    if not arg:
        return {}
    from pathlib import Path

    candidate = Path(arg)
    text = candidate.read_text(encoding="utf-8") if candidate.exists() else arg
    try:
        return json.loads(text)
    except ValueError:
        _fail(f"--config is neither a JSON file nor inline JSON: {arg}")


def cmd_dashboard(args):
    from .dashboard.server import serve

    serve(args.host, args.port)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="episodic", description="Coding episode capture and dataset tooling.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="ingest a hook payload from stdin").set_defaults(func=cmd_ingest)

    start = sub.add_parser("start", help="start tracing and set the session intent")
    start.add_argument("intent", nargs="*", default=[])
    start.add_argument("--session")
    start.set_defaults(func=lambda a: cmd_start(_normalize_intent(a)))

    summary = sub.add_parser("summary", help="summarize the current or a given episode")
    summary.add_argument("--session")
    summary.add_argument("--episode")
    summary.add_argument("--json", action="store_true")
    summary.set_defaults(func=cmd_summary)

    mark = sub.add_parser("mark", help="attach a one-click feedback label")
    mark.add_argument("label", nargs="+")
    mark.add_argument("--note")
    mark.add_argument("--session")
    mark.set_defaults(func=cmd_mark)

    pr_notes = sub.add_parser("create-pr-notes", help="print a suggested PR title and description")
    pr_notes.add_argument("--session")
    pr_notes.add_argument("--episode")
    pr_notes.add_argument("--json", action="store_true")
    pr_notes.set_defaults(func=cmd_pr_notes)

    export = sub.add_parser("export-episode", help="export episodes to a dataset format")
    export.add_argument("--format", default="jsonl", choices=["sft", "dpo", "reward", "rlds", "jsonl", "parquet"])
    export.add_argument("--episode")
    export.add_argument("--session")
    export.add_argument("--all", action="store_true")
    export.add_argument("--out")
    export.set_defaults(func=cmd_export)

    link = sub.add_parser("link", help="link an episode to a PR / CI / merge outcome")
    link.add_argument("--pr")
    link.add_argument("--auto", action="store_true")
    link.add_argument("--episode")
    link.add_argument("--session")
    link.add_argument("--cwd")
    link.add_argument("--refresh", action="store_true", help="re-pull the linked PR for this episode")
    link.add_argument("--refresh-all", dest="refresh_all", action="store_true",
                      help="re-pull every in-flight linked PR (cron/watch friendly)")
    link.set_defaults(func=cmd_link)

    regression = sub.add_parser("regression", help="blame a bugfix/revert commit back to the episodes that caused it")
    regression.add_argument("commit")
    regression.add_argument("--cwd")
    regression.add_argument("--apply", action="store_true", help="mark culprit episodes caused_regression and recompute reward")
    regression.add_argument("--fuzzy", action="store_true", help="also penalize file-overlap matches (lower precision)")
    regression.set_defaults(func=cmd_regression)

    replay = sub.add_parser("replay-task", help="create or run a replayable task")
    replay.add_argument("replay_command", choices=["create", "run"])
    replay.add_argument("--episode")
    replay.add_argument("--session")
    replay.add_argument("--replay")
    replay.add_argument("--model", default="claude-code")
    replay.set_defaults(func=cmd_replay)

    listing = sub.add_parser("list", help="list captured episodes")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=cmd_list)

    show = sub.add_parser("show", help="print or validate a stored episode")
    show.add_argument("episode")
    show.add_argument("--validate", action="store_true")
    show.set_defaults(func=cmd_show)

    finalize = sub.add_parser("finalize", help="finalize the current session into an episode")
    finalize.add_argument("--session")
    finalize.set_defaults(func=cmd_finalize)

    schema = sub.add_parser("schema", help="print or dump the CodingEpisode JSON Schema")
    schema.add_argument("schema_command", nargs="?", default="print", choices=["print", "dump"])
    schema.set_defaults(func=cmd_schema)

    train = sub.add_parser("train", help="train a model on an exported dataset (pluggable backend)")
    train.add_argument("dataset", nargs="?", default="-")
    train.add_argument("--trainer", default="trl-sft")
    train.add_argument("--config")
    train.add_argument("--model")
    train.add_argument("--out")
    train.add_argument("--list", action="store_true")
    train.set_defaults(func=cmd_train)

    dashboard = sub.add_parser("dashboard", help="serve the local episode dashboard")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=4317)
    dashboard.set_defaults(func=cmd_dashboard)

    return parser


def _normalize_intent(args):
    if isinstance(args.intent, list):
        args.intent = " ".join(args.intent).strip()
    return args


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
