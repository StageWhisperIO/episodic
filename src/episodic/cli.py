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
    if out == "-":
        print(json.dumps(result, indent=2, ensure_ascii=False), file=sys.stderr)
    else:
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
        result = replay.run_replay(args.replay, args.model, runner_cmd=args.runner_cmd, execute=args.execute)
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


def cmd_renormalize(args):
    rebuilt = service.renormalize()
    print(f"renormalized {len(rebuilt)} episode(s) from stored sessions")
    return 0


def cmd_label(args):
    from .core import feedback
    from .core.episode import build_episode

    start = args.store
    generate = feedback.command_generate(args.cmd, timeout=args.timeout)
    if args.all:
        session_ids = store.list_sessions(start)
    else:
        session_ids = [args.session or store.get_current(start)]
    if not any(session_ids):
        _fail("no current session; pass --session or --all")

    if args.raw:
        session_id = next(sid for sid in session_ids if sid)
        base = build_episode(store.get_session(session_id, start))
        result = feedback.probe(feedback.build_prompt(base), args.cmd, timeout=args.timeout)
        print(f"command={result['command']}")
        print(f"exit={result['code']} stdout_chars={len(result['stdout'])} stderr_chars={len(result['stderr'])}")
        print("--- stdout (first 1200) ---")
        print(result["stdout"][:1200])
        if result["stderr"].strip():
            print("--- stderr (first 600) ---")
            print(result["stderr"][:600])
        print("--- extracted json ---")
        print(feedback._extract_json(result["stdout"]))
        return 0

    failed = False
    for session_id in session_ids:
        if not session_id:
            continue
        session = store.get_session(session_id, start)
        if not session["events"]:
            continue
        episode = build_episode(session, generate=generate)
        if episode.get("labeler_error"):
            print(f"episodic: labeler failed for {episode['id']} ({session_id[:8]}): "
                  f"{episode['labeler_error']}", file=sys.stderr)
            failed = True
            continue
        mined = [item for item in episode["human_feedback"] if item.get("source") == "mined"]
        hint = episode.get("outcome_hint") or {}
        rv = episode["reward_vector"]
        deploys = episode.get("deployments", [])
        print(f"{episode['id']} ({session_id[:8]}): {len(mined)} mined label(s); "
              f"outcome_hint={hint.get('success', '-')} ({hint.get('confidence', '-')}); "
              f"deploys={len(deploys)}; human_label={rv['human_label']} "
              f"outcome_src={rv['components']['outcome_source']} composite={rv['composite']}")
        for item in mined:
            print(f"   [#{item.get('evidence_step_index')}] {item['label']} "
                  f"({item.get('confidence')}) :: {(item.get('note') or '')[:80]}")
        for deployment in deploys:
            print(f"   deploy {deployment['method']}->{deployment['target_env']} "
                  f"verified={deployment['verified']}")
        if hint.get("rationale"):
            print(f"   outcome: {hint.get('rationale')[:100]}")
        if args.save:
            store.save_episode(episode, start)
    if args.save:
        print("saved.")
    if failed:
        raise SystemExit(1)
    return 0


def cmd_segment(args):
    from .core import feedback, segment as segment_mod

    start = args.store
    generate = feedback.command_generate(args.cmd, timeout=args.timeout) if args.label else None
    session_id = args.session or store.get_current(start)
    if not session_id:
        _fail("no current session; pass --session")
    session = store.get_session(session_id, start)
    if not session["events"]:
        _fail("session has no events")
    children = segment_mod.segment_session(session, generate=generate)
    print(f"{session_id[:8]} -> {len(children)} sub-trajectory(ies)")
    failed = False
    for child in children:
        if child.get("labeler_error"):
            print(f"episodic: labeler failed for {child['id']}: {child['labeler_error']}", file=sys.stderr)
            failed = True
            continue
        rv = child["reward_vector"]
        print(f"   #{child['segment_index']} {child['id']} composite={rv['composite']} "
              f"steps={len(child['steps'])} tests={len(child['tests'])} deploys={len(child['deployments'])} "
              f":: {(child['intent'] or '')[:70]!r}")
        if args.save:
            store.save_episode(child, start)
    if args.save:
        print("saved.")
    if failed:
        raise SystemExit(1)
    return 0


def cmd_audit(args):
    from collections import Counter
    from .core import validity, feedback
    from .core.ids import episode_id_from_session

    start = args.store
    generate = feedback.command_generate(args.cmd, timeout=args.timeout) if args.validate else None
    episodes = store.load_episodes(start)
    if not episodes:
        _fail("no stored episodes to audit")

    session_by_episode_id = {}
    if args.save:
        session_by_episode_id = {
            episode_id_from_session(session_id): session_id
            for session_id in store.list_sessions(start)
        }

    trust_counts = Counter()
    category_counts = Counter()
    low = []
    for episode in episodes:
        result = validity.assess(episode, generate)
        trust_counts[result["trust"]] += 1
        for category in result["categories"]:
            category_counts[category] += 1
        if result["trust"] == "low":
            low.append((episode["id"], result))
        if args.save:
            episode["validity"] = result
            store.save_episode(episode, start)
            session_id = session_by_episode_id.get(episode["id"])
            if session_id:
                store.update_meta(session_id, {"audit_validity": result}, start)

    total = len(episodes)
    broken = trust_counts["low"]
    print(f"audited {total} episode(s){'  (LLM-validated)' if generate else ''}")
    print(f"trust: high={trust_counts['high']} medium={trust_counts['medium']} low={trust_counts['low']}")
    print(f"BROKEN (low-trust / noisy reward): {broken}/{total} = {round(100.0 * broken / total, 1)}%")
    print("categories:")
    for category, count in category_counts.most_common():
        print(f"   {category}: {count}")
    if low:
        print("low-trust episodes:")
        for episode_id, result in low[:args.limit]:
            codes = ",".join(flag["code"] for flag in result["flags"])
            print(f"   {episode_id} [{result['severity']}] {result['categories']} :: {codes}")
    if args.save:
        print("saved validity onto episodes.")
    return 0


def cmd_schema(args):
    if args.schema_command == "dump":
        target = paths.resolve_base() / "schemas" / "episode.schema.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(EPISODE_SCHEMA, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {target}", file=sys.stderr)
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


def cmd_loop(args):
    from . import loop, trainers

    config = _load_train_config(args.config)
    if args.out:
        config["out"] = args.out
    if args.trainer:
        config["trainer"] = args.trainer
    if args.format:
        config["format"] = args.format
    if args.execute:
        config["execute"] = True
    config.setdefault("judge", not args.no_judge)
    if args.judge_cmd:
        config["judge_cmd"] = args.judge_cmd
    if args.judge_timeout is not None:
        config["judge_timeout"] = args.judge_timeout
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.evaluator:
        config.setdefault("evaluator", {})["type"] = args.evaluator
    if args.router:
        config["router"] = True
    if args.sim_prefilter:
        config["sim_prefilter"] = True
    if args.sim_max_turns is not None:
        config["sim_max_turns"] = args.sim_max_turns
    if args.sim_backend in ("mlx", "tinker"):
        from .worldmodel import inference as wm_inference

        try:
            if args.sim_backend == "mlx":
                if not args.sim_model_dir:
                    _fail("--sim-backend mlx needs --sim-model-dir")
                config["sim_predictor"] = wm_inference.mlx_predictor(
                    args.sim_model_dir, adapter_path=args.sim_adapter_path)
            else:
                if not args.sim_sampler_path:
                    _fail("--sim-backend tinker needs --sim-sampler-path")
                config["sim_predictor"] = wm_inference.tinker_predictor(
                    args.sim_sampler_path, base_model=args.sim_model_dir)
        except trainers.TrainerUnavailable as exc:
            print(f"episodic: {exc.hint}", file=sys.stderr)
            return 0
    elif args.sim_backend:
        config["sim_predictor"] = args.sim_backend

    if args.eval_backend:
        config["eval_backend"] = args.eval_backend
    if args.eval_model_dir:
        config["eval_model_dir"] = args.eval_model_dir
    if args.eval_sampler_path:
        config["eval_sampler_path"] = args.eval_sampler_path
    eval_backend_config = {}
    if args.eval_base_url:
        eval_backend_config["base_url"] = args.eval_base_url
    if args.eval_api_key:
        eval_backend_config["api_key"] = args.eval_api_key
    if eval_backend_config:
        config["eval_backend_config"] = eval_backend_config

    try:
        manifest = loop.run_loop(config)
    except trainers.TrainerUnavailable as exc:
        print(f"episodic: {exc.hint}", file=sys.stderr)
        return 0
    except ValueError as exc:
        _fail(str(exc))
    _print_json(manifest)
    return 0


def cmd_worldbench(args):
    from . import worldbench, trainers

    episodes = store.load_episodes()
    if not episodes:
        _fail("no episodes to benchmark")
    if args.backend:
        from .worldmodel import inference as wm_inference

        try:
            if args.backend == "mlx":
                if not args.model_dir:
                    _fail("--backend mlx needs --model-dir")
                predictor = wm_inference.mlx_predictor(args.model_dir, adapter_path=args.adapter_path)
            else:
                if not args.sampler_path:
                    _fail("--backend tinker needs --sampler-path")
                predictor = wm_inference.tinker_predictor(args.sampler_path, base_model=args.model_dir)
        except trainers.TrainerUnavailable as exc:
            print(f"episodic: {exc.hint}", file=sys.stderr)
            return 0
    elif args.cmd:
        if not args.execute:
            _fail("--cmd runs a shell command per turn; pass --execute to allow it")
        try:
            predictor = worldbench.command_predictor(args.cmd)
        except ValueError as exc:
            _fail(str(exc))
    else:
        predictor = args.predictor

    if args.rollout:
        report = worldbench.rollout_bench(episodes, predictor, max_turns=args.max_turns)
        if args.turing:
            report["turing"] = worldbench.rollout_turing_test(
                episodes, predictor, max_turns=args.max_turns, seed=args.seed)
        _print_json(report)
        return 0

    report = worldbench.run_bench(
        episodes, predictor,
        one_per_trajectory=not args.all_turns,
        seed=args.seed,
        source_holdout=args.source_holdout,
    )
    if args.turing:
        report["turing"] = worldbench.turing_test(
            episodes, predictor, one_per_trajectory=not args.all_turns, seed=args.seed)
    _print_json(report)
    return 0


def _wm_fidelity_report(episodes, predictor, max_turns, seed):
    from . import worldbench

    rollout = worldbench.rollout_bench(episodes, predictor, max_turns=max_turns)
    turing = worldbench.rollout_turing_test(episodes, predictor, max_turns=max_turns, seed=seed)
    return {
        "mean_composite": rollout["mean_composite"],
        "mean_drift": rollout["mean_drift"],
        "n_scored": rollout["n_scored"],
        "discriminator_accuracy": turing["discriminator_accuracy"],
        "indistinguishability": turing["indistinguishability"],
    }


def cmd_wm_validate(args):
    from pathlib import Path

    from . import exporters, loop, trainers, worldbench
    from .worldmodel import inference as wm_inference
    from .worldmodel import validate as wm_validate

    episodes = [ep for ep in store.load_episodes() if ep.get("steps")]
    if not episodes:
        _fail("no episodes with steps to validate")

    train_eps, holdout_eps = loop.split_episodes(episodes, args.holdout_frac, args.seed)
    if not holdout_eps:
        _fail("holdout split is empty; lower --holdout-frac or add more episodes")

    out = Path(args.out) if args.out else paths.exports_dir() / "wm_validate"
    out.mkdir(parents=True, exist_ok=True)

    train_config = _load_train_config(args.train_config)

    predictor = None
    predictor_info = {"backend": None, "base_model": None, "adapter_path": None, "trained": False}
    dataset_info = None

    if args.adapter_path or args.execute:
        export_result = exporters.export(train_eps, "wm", str(out / "dataset"))
        dataset_info = {"files": export_result["files"], "count": export_result.get("count")}

        adapter_path = args.adapter_path
        base_model = args.model or train_config.get("model")
        trained = False
        if adapter_path is None:
            try:
                train_manifest = trainers.train(
                    "mlx-sft", export_result["files"][0], str(out / "candidate"), train_config)
            except trainers.TrainerUnavailable as exc:
                print(f"episodic: {exc.hint}", file=sys.stderr)
                return 0
            result = train_manifest.get("result") or {}
            adapter_path = result.get("model_dir")
            base_model = base_model or result.get("base_model")
            trained = True
        try:
            predictor = wm_inference.mlx_predictor(base_model, adapter_path=adapter_path)
        except trainers.TrainerUnavailable as exc:
            print(f"episodic: {exc.hint}", file=sys.stderr)
            return 0
        predictor_info = {
            "backend": "mlx", "base_model": base_model, "adapter_path": adapter_path, "trained": trained,
        }

    fidelity_report = {
        name: _wm_fidelity_report(holdout_eps, name, args.max_turns, args.seed)
        for name in ("oracle", "prefix", "empty")
    }
    if predictor is not None:
        fidelity_report["trained"] = _wm_fidelity_report(holdout_eps, predictor, args.max_turns, args.seed)

    replay_correlation = None
    if args.replay_correlate:
        if not args.execute:
            _fail("--replay-correlate needs --execute (clones repos and runs recorded test commands)")
        sim_predictor = predictor or worldbench.NAMED_PREDICTORS["prefix"]
        if args.replay_correlate_all:
            correlate_eps = holdout_eps
        else:
            correlate_eps = [ep for ep in holdout_eps if wm_validate.has_captured_verifier(ep)]
        sim = wm_validate.sim_scores(correlate_eps, sim_predictor, max_turns=args.max_turns)
        real = wm_validate.offline_replay_scores(correlate_eps)
        replay_correlation = wm_validate.correlate(sim, real)
        replay_correlation["verifier_filtered"] = not args.replay_correlate_all
        replay_correlation["n_holdout"] = len(holdout_eps)

    report = {
        "holdout_frac": args.holdout_frac,
        "seed": args.seed,
        "n_train": len(train_eps),
        "n_holdout": len(holdout_eps),
        "dataset": dataset_info,
        "predictor": predictor_info,
        "fidelity": fidelity_report,
        "replay_correlation": replay_correlation,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_json(report)
    return 0


def cmd_doctor(args):
    from . import selfcheck

    report = selfcheck.run_checks()
    if args.json:
        _print_json(report)
        return 0 if report["ok"] else 1
    symbol = {True: "ok  ", False: "FAIL", None: "skip"}
    for check in report["checks"]:
        detail = check["detail"]
        detail = json.dumps(detail) if isinstance(detail, dict) else detail
        print(f"[{symbol[check['ok']]}] {check['name']:<18} {detail}")
    if report["ok"]:
        print(f"\n{report['passed']}/{report['total']} checks ok — install is healthy")
    else:
        print(f"\n{report['passed']}/{report['total']} checks ok — FAILED: {', '.join(report['failed'])}")
    return 0 if report["ok"] else 1


def cmd_dashboard(args):
    from .dashboard.server import serve

    serve(args.host, args.port)
    return 0


def _set_tier(config, tier, backend, base_url, model):
    if backend or base_url or model:
        tier_config = config.setdefault(tier, {})
        if backend:
            tier_config["backend"] = backend
        if base_url:
            tier_config["base_url"] = base_url
        if model:
            tier_config["model"] = model


def cmd_eval(args):
    from .eval import flywheel, gate, redgreen

    if args.certify:
        eps = [ep for ep in store.load_episodes()
               if any((d.get("unified") or "").strip() for d in ep.get("diffs", []))]
        cert = gate.certify_corpus(eps)
        if args.json:
            _print_json(cert)
        else:
            print(f"certified (test-necessary): {cert['certified']}/{cert['total']} episodes with a diff")
            for episode_id in cert["certified_ids"]:
                print(f"  {episode_id}")
        return 0 if cert["certified"] else 1

    repos = args.repos or str(paths.home() / "eval_repos")
    if args.generate:
        episodes = redgreen.generate_corpus(repos, variants=args.variants)
    else:
        episodes = [ep for ep in store.load_episodes() if "swe" in (ep.get("labels") or [])]
        if not episodes:
            episodes = redgreen.generate_corpus(repos, variants=args.variants)

    report = {"corpus": len(episodes)}
    gate_rep = gate.gate_report(episodes)
    report["gate"] = {"total": gate_rep["total"], "clean": gate_rep["clean"],
                      "all_clean": gate_rep["all_clean"]}
    if not args.json:
        print(f"gate: {gate_rep['clean']}/{gate_rep['total']} tasks clean "
              f"(oracle green, empty+broken red)")

    if not args.gate_only:
        train, held = flywheel.stratified_split(episodes, per_class_held=args.per_class_held)
        if args.backend == "stub":
            lift = flywheel.oracle_vs_empty_lift(held)
        else:
            sft = str(paths.home() / "eval_sft.jsonl")
            out = str(paths.home() / "eval_candidate")
            lift = flywheel.real_lift(train, held, backend=args.backend, model=args.model,
                                      sft_path=sft, out_dir=out, epochs=args.epochs, iters=args.iters,
                                      lora_rank=args.lora_rank, max_tokens=args.max_tokens,
                                      agentic_turns=args.agentic_turns)
        report["lift"] = lift
        if not args.json:
            print(f"lift [{args.backend}]: base {lift['base_solved']}/{lift['held']} -> "
                  f"trained {lift['trained_solved']}/{lift['held']} (+{lift['lift']})")
            for cls, stats in sorted(lift["by_class"].items()):
                print(f"  [{cls}] base={stats['base']}/{stats['held']} trained={stats['trained']}/{stats['held']}")

    if args.json:
        _print_json(report)
    return 0 if gate_rep["all_clean"] else 1


def cmd_mine(args):
    from .eval import mine

    out = args.out or str(paths.home() / "mined_repos")
    episodes = mine.mine_repo(args.repo, out, max_commits=args.max_commits, limit=args.limit,
                              save=not args.no_save)
    if args.json:
        _print_json({"repo": args.repo, "mined": len(episodes), "ids": [ep["id"] for ep in episodes]})
    else:
        print(f"mined {len(episodes)} certified red->green tasks from {args.repo}")
        for ep in episodes:
            print(f"  {ep['id']}  {[d['file'] for d in ep['diffs']][:3]}")
    return 0 if episodes else 1


def cmd_serve(args):
    from .serving.server import serve

    config = _load_train_config(args.config)
    _set_tier(config, "distilled", args.distilled_backend, args.distilled_base_url, args.distilled_model)
    _set_tier(config, "frontier", args.frontier_backend, args.frontier_base_url, args.frontier_model)
    if args.router_model:
        config["router_model_path"] = args.router_model
    if args.router_threshold is not None:
        config["router_threshold"] = args.router_threshold
    serve(args.host, args.port, config)
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
    export.add_argument("--format", default="jsonl", choices=["sft", "dpo", "reward", "rlds", "wm", "jsonl", "parquet", "harbor", "molt"])
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
    replay.add_argument("--runner-cmd", dest="runner_cmd",
                        help="shell template to drive the model: {model} {prompt_file} {workspace}")
    replay.add_argument("--execute", action="store_true",
                        help="clone the repo and run the recorded test command + runner (off by default)")
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

    renormalize = sub.add_parser(
        "renormalize", help="rebuild all stored episodes from their raw session events")
    renormalize.set_defaults(func=cmd_renormalize)

    label = sub.add_parser("label", help="mine user-feedback labels + outcome hint from a session via an LLM labeler")
    label.add_argument("--session", help="session id (default: current)")
    label.add_argument("--all", action="store_true", help="relabel every stored session")
    label.add_argument("--cmd", help="labeler command reading the prompt on stdin (default: $EPISODIC_LABELER_CMD or claude -p haiku; needs ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN for the default)")
    label.add_argument("--timeout", type=int, default=120)
    label.add_argument("--save", action="store_true", help="persist the relabeled episode(s)")
    label.add_argument("--store", help="path to a project store to label (default: current directory), enables self-capture of any repo")
    label.add_argument("--raw", action="store_true", help="print the raw labeler output for one session and exit (diagnostic)")
    label.set_defaults(func=cmd_label)

    segment = sub.add_parser("segment", help="split a session into attempt-scoped sub-trajectories (child episodes)")
    segment.add_argument("--session", help="session id (default: current)")
    segment.add_argument("--label", action="store_true", help="also mine feedback per sub-trajectory via the LLM labeler")
    segment.add_argument("--cmd", help="labeler command when --label is set")
    segment.add_argument("--timeout", type=int, default=120)
    segment.add_argument("--store", help="path to a project store (default: current directory)")
    segment.add_argument("--save", action="store_true", help="persist the child episodes")
    segment.set_defaults(func=cmd_segment)

    audit = sub.add_parser("audit", help="reward-quality audit: flag episodes whose reward is untrustworthy (signal vs noise)")
    audit.add_argument("--validate", action="store_true", help="also run the LLM validator on each episode (agent-assisted)")
    audit.add_argument("--cmd", help="labeler command when --validate is set")
    audit.add_argument("--timeout", type=int, default=120)
    audit.add_argument("--store", help="path to a project store (default: current directory)")
    audit.add_argument("--limit", type=int, default=15, help="max low-trust episodes to list")
    audit.add_argument("--save", action="store_true", help="persist the computed validity onto each episode")
    audit.set_defaults(func=cmd_audit)

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

    loop = sub.add_parser("loop", help="closed RL loop: filter -> train -> replay-eval -> promote")
    loop.add_argument("--config")
    loop.add_argument("--trainer")
    loop.add_argument("--format", choices=["sft", "dpo", "reward"])
    loop.add_argument("--out")
    loop.add_argument("--execute", action="store_true",
                      help="actually run replay-eval (clones repos and runs recorded test commands)")
    loop.add_argument("--no-judge", action="store_true",
                      help="disable the default agent-as-a-judge rubric scoring (on by default; "
                           "needs ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN or --judge-cmd)")
    loop.add_argument("--judge-cmd", help="judge command reading the prompt on stdin "
                                          "(default: $EPISODIC_LABELER_CMD or claude -p haiku)")
    loop.add_argument("--judge-timeout", type=int, help="judge subprocess timeout in seconds (default: 120)")
    loop.add_argument("--epochs", type=int, help="number of epochs to run (default: 1); the evaluator "
                                                  "(judge/critic) only refreshes between epoch boundaries")
    loop.add_argument("--evaluator", choices=["rubric_judge", "local_critic", "trl_reward"],
                      help="co-evolving evaluator backend for the judge-only rubric criteria "
                           "(default: rubric_judge, i.e. the plain agent-as-a-judge)")
    loop.add_argument("--router", action="store_true",
                      help="learn a cost-aware small-vs-frontier router from reward/validity difficulty "
                           "signals and write <out>/router_model.json for `episodic serve --router-model`")
    loop.add_argument("--sim-prefilter", action="store_true",
                      help="rank holdout episodes with a cheap WorldModelEnv rollout before spending the "
                           "real replay-eval budget, instead of arbitrary id order")
    loop.add_argument("--sim-backend", choices=["prefix", "oracle", "empty", "echo", "mlx", "tinker"],
                      help="predictor used by --sim-prefilter (default: prefix)")
    loop.add_argument("--sim-model-dir", help="mlx model path/HF repo id or tinker base model id "
                                              "(--sim-backend mlx/tinker)")
    loop.add_argument("--sim-sampler-path", help="tinker sampler weights path (--sim-backend tinker, required)")
    loop.add_argument("--sim-adapter-path", help="mlx LoRA adapter dir layered on --sim-model-dir "
                                                  "(--sim-backend mlx)")
    loop.add_argument("--sim-max-turns", type=int, help="cap sim rollout turns per episode")
    loop.add_argument("--eval-backend", choices=["stub", "mlx", "tinker", "serving"],
                      help="model-driven runner for --execute replay-eval: generates a unified diff per "
                           "episode and git-applies it before running the captured test command, instead of "
                           "leaving candidate/base workspaces at the bare base commit (default: none)")
    loop.add_argument("--eval-model-dir", help="mlx model path/HF repo id or tinker base model id "
                                                "(--eval-backend mlx/tinker)")
    loop.add_argument("--eval-sampler-path", help="tinker sampler weights path (--eval-backend tinker, required)")
    loop.add_argument("--eval-base-url", help="upstream OpenAI-compatible base_url for --eval-backend serving "
                                              "(required unless --eval-api-key is set; refuses the public "
                                              "OpenAI API by default)")
    loop.add_argument("--eval-api-key", help="api key for --eval-backend serving")
    loop.set_defaults(func=cmd_loop)

    worldbench = sub.add_parser("worldbench", help="evaluate next-observation prediction (world-model fidelity)")
    worldbench.add_argument("--predictor", default="prefix", choices=["prefix", "oracle", "empty", "echo"])
    worldbench.add_argument("--cmd", help="shell template to run a model predictor; receives {prompt_file}")
    worldbench.add_argument("--execute", action="store_true", help="allow --cmd to run shell commands")
    worldbench.add_argument("--all-turns", dest="all_turns", action="store_true",
                            help="score every turn (default: one per trajectory, the Echo-Trap-safe pool)")
    worldbench.add_argument("--source-holdout", dest="source_holdout", action="store_true",
                            help="evaluate only held-out data sources (OOD generalization)")
    worldbench.add_argument("--seed", type=int, default=0)
    worldbench.add_argument("--turing", action="store_true",
                            help="also run the double-blind Turing-test discriminator")
    worldbench.add_argument("--backend", choices=["mlx", "tinker"],
                            help="use a real trained model as the predictor instead of --predictor/--cmd")
    worldbench.add_argument("--model-dir", help="mlx model path/HF repo id (--backend mlx) or tinker base "
                                                "model id (--backend tinker)")
    worldbench.add_argument("--adapter-path", help="mlx LoRA adapter dir layered on --model-dir (--backend mlx)")
    worldbench.add_argument("--sampler-path", help="tinker sampler weights path (--backend tinker, required)")
    worldbench.add_argument("--rollout", action="store_true",
                            help="closed-loop trajectory rollout (WorldModelEnv) instead of per-turn "
                                 "teacher-forced scoring")
    worldbench.add_argument("--max-turns", type=int, help="cap rollout turns per episode (--rollout only)")
    worldbench.set_defaults(func=cmd_worldbench)

    wm_validate = sub.add_parser(
        "wm-validate",
        help="train/load a world model, validate its fidelity against prefix/empty/oracle on a real "
             "holdout split, and optionally correlate sim rollout scores with real replay-eval scores",
    )
    wm_validate.add_argument("--out", help="output dir (default: <home>/exports/wm_validate)")
    wm_validate.add_argument("--holdout-frac", type=float, default=0.25,
                             help="fraction of episodes held out for fidelity scoring (default: 0.25)")
    wm_validate.add_argument("--seed", type=int, default=0)
    wm_validate.add_argument("--max-turns", type=int, help="cap rollout turns per episode")
    wm_validate.add_argument("--model", help="mlx base model id (default: mlx-sft trainer default, "
                                              "or --train-config's 'model' key)")
    wm_validate.add_argument("--adapter-path", help="reuse an already-trained mlx LoRA adapter instead of "
                                                     "training a fresh one with --execute")
    wm_validate.add_argument("--train-config", help="mlx-sft training config: JSON file or inline JSON")
    wm_validate.add_argument("--execute", action="store_true",
                             help="actually run local mlx-sft training on the train split (no Tinker, "
                                  "no network billing; needs mlx-lm on Apple Silicon)")
    wm_validate.add_argument("--replay-correlate", action="store_true",
                             help="also score the holdout split with a real offline replay-eval (git clone "
                                  "+ recorded test command) and correlate it with the sim rollout scores; "
                                  "needs --execute")
    wm_validate.add_argument("--replay-correlate-all", action="store_true",
                             help="correlate over the whole holdout instead of only episodes with a captured "
                                  "test verifier; the replay score for verifier-less episodes is diff-overlap "
                                  "only, so the correlation conflates two different targets (default: off)")
    wm_validate.set_defaults(func=cmd_wm_validate)

    eval_cmd = sub.add_parser(
        "eval-flywheel",
        help="generate red->green tasks, verify the replay-eval gate discriminates, and measure "
             "base-vs-trained flywheel lift through it (stub oracle-vs-empty by default; mlx/tinker "
             "for a real trained model)")
    eval_cmd.add_argument("--generate", action="store_true",
                          help="(re)build the red->green task corpus into the store before evaluating")
    eval_cmd.add_argument("--variants", type=int, default=1,
                          help="distinct instances per bug-class template (default: 1 => 12 tasks)")
    eval_cmd.add_argument("--repos", help="directory for the generated task repos (default: <home>/eval_repos)")
    eval_cmd.add_argument("--gate-only", dest="gate_only", action="store_true",
                          help="only verify gate discrimination; skip the flywheel lift measurement")
    eval_cmd.add_argument("--certify", action="store_true",
                          help="instead of the synthetic corpus, certify the current store's captured "
                               "episodes: clone at base and keep only those whose captured test is "
                               "test-necessary (fails red without the diff, passes green with it)")
    eval_cmd.add_argument("--backend", choices=["stub", "mlx", "tinker"], default="stub",
                          help="lift backend: stub (oracle-vs-empty, deterministic, no model) or a real "
                               "base-vs-trained run on mlx/tinker (default: stub)")
    eval_cmd.add_argument("--model", help="mlx base model id or tinker base model id (--backend mlx/tinker)")
    eval_cmd.add_argument("--per-class-held", dest="per_class_held", type=int, default=1,
                          help="held-out tasks per bug class (default: 1)")
    eval_cmd.add_argument("--epochs", type=int, default=3, help="tinker-sft epochs (--backend tinker)")
    eval_cmd.add_argument("--iters", type=int, default=400, help="mlx-sft iters (--backend mlx)")
    eval_cmd.add_argument("--lora-rank", dest="lora_rank", type=int, default=32,
                          help="tinker LoRA rank (--backend tinker)")
    eval_cmd.add_argument("--max-tokens", dest="max_tokens", type=int, default=768,
                          help="generation budget per task for the trained/base model")
    eval_cmd.add_argument("--agentic-turns", dest="agentic_turns", type=int, default=0,
                          help="score with a multi-turn agentic runner (generate -> apply -> run test -> "
                               "feed the failure back -> retry) up to this many turns instead of a single "
                               "shot (default: 0, single shot)")
    eval_cmd.add_argument("--json", action="store_true")
    eval_cmd.set_defaults(func=cmd_eval)

    mine_cmd = sub.add_parser(
        "mine-history",
        help="mine a repo's git history into certified red->green tasks: for each commit that changes "
             "both a test and source, inject the test at the parent (must fail red) and apply the source "
             "change (must pass green), keeping only test-necessary units")
    mine_cmd.add_argument("repo", help="path to a local git repo to mine")
    mine_cmd.add_argument("--out", help="dir for the scratch task clones (default: <home>/mined_repos)")
    mine_cmd.add_argument("--max-commits", dest="max_commits", type=int, default=200,
                          help="how many recent commits to scan (default: 200)")
    mine_cmd.add_argument("--limit", type=int, help="stop after this many certified tasks")
    mine_cmd.add_argument("--no-save", dest="no_save", action="store_true",
                          help="do not persist mined episodes to the store")
    mine_cmd.add_argument("--json", action="store_true")
    mine_cmd.set_defaults(func=cmd_mine)

    doctor = sub.add_parser("doctor", help="run end-to-end self-checks on the install")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    dashboard = sub.add_parser("dashboard", help="serve the local episode dashboard")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=4317)
    dashboard.set_defaults(func=cmd_dashboard)

    serve_cmd = sub.add_parser("serve", help="thin OpenAI-compatible proxy/router for the promoted model")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8000)
    serve_cmd.add_argument("--config", help="serving config: JSON file or inline JSON "
                                            "({'distilled': {...}, 'frontier': {...}})")
    serve_cmd.add_argument("--distilled-backend", choices=["openai", "ollama", "vllm", "tinker"])
    serve_cmd.add_argument("--distilled-base-url")
    serve_cmd.add_argument("--distilled-model")
    serve_cmd.add_argument("--frontier-backend", choices=["openai", "ollama", "vllm", "tinker"])
    serve_cmd.add_argument("--frontier-base-url")
    serve_cmd.add_argument("--frontier-model")
    serve_cmd.add_argument("--router-model", help="path to a router_model.json trained by "
                                                   "`episodic loop --router`, for cost-aware small-vs-frontier "
                                                   "escalation instead of the plain char-count heuristic")
    serve_cmd.add_argument("--router-threshold", type=float,
                           help="escalation probability threshold for the learned router (default: 0.5)")
    serve_cmd.set_defaults(func=cmd_serve)

    return parser


def list_commands():
    return sorted(build_parser()._subparsers._group_actions[0].choices)


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
