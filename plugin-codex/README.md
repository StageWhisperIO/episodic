# plugin-codex

Episodic integration for [OpenAI Codex CLI](https://github.com/openai/codex).

Sessions are tagged `agent="codex"` via `EPISODIC_AGENT=codex` (set automatically by both scripts).

## Two integration methods

### 1. Live capture via `notify` (real-time)

Codex calls a notify program on every agent turn. Add this to `~/.codex/config.toml`:

```toml
notify = ["python3", "/absolute/path/to/episodic/plugin-codex/notify.py"]
```

See `config.example.toml` for a full snippet. Codex will invoke `notify.py` with a JSON string as `argv[1]` on each turn event. Episodic ingests each event immediately.

### 2. Import rollout files after the fact (most reliable)

Codex writes JSONL session logs under `~/.codex/sessions/**/rollout-*.jsonl`. Use `import_rollout.py` to replay any file into Episodic:

```bash
python plugin-codex/import_rollout.py ~/.codex/sessions/2026/.../rollout-xxx.jsonl
```

This emits a `SessionStart`, maps each JSONL line to a hook payload, ingests them, then emits `SessionEnd`. The return value is the count of ingested events.

To import all sessions at once:

```bash
find ~/.codex/sessions -name "rollout-*.jsonl" | xargs -I{} python plugin-codex/import_rollout.py {}
```

## Caveat

Codex's internal event schemas evolve across versions. Both `notify.py` and `import_rollout.py` map events on a best-effort basis using `.get()` with safe defaults — unknown event shapes are silently skipped and errors never propagate to Codex.
