# collector

Ingestion. Two surfaces, both feeding the same [episode-store](../episode-store/):

1. **Hooks (primary)** — `src/episodic/collector/hook.py`. Claude Code / Codex call
   `episodic ingest` on each lifecycle event and pipe a JSON payload on stdin. The hook
   normalizes it (`src/episodic/core/normalize.py`) into a semantic capture event
   (`tool_pre`, `file_edit`, `shell_command`, `denial`, …) and appends it. It is silent,
   never blocks, and never crashes the agent (all failures exit 0).

2. **OTel (optional plumbing)** — `src/episodic/collector/otel.py`. A local OTLP/HTTP-JSON
   receiver that captures the **token / cost** stats hooks don't expose, and merges them into
   the session so they land in `episode.stats`. Hidden from the user.

   ```bash
   python -m episodic.collector.otel 127.0.0.1 4318
   ```

   Point Claude Code at it (it stays invisible):

   ```bash
   export CLAUDE_CODE_ENABLE_TELEMETRY=1
   export OTEL_METRICS_EXPORTER=otlp
   export OTEL_LOGS_EXPORTER=otlp
   export OTEL_EXPORTER_OTLP_PROTOCOL=http/json
   export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
   ```

See [`src/episodic/collector/README.md`](../src/episodic/collector/README.md) for the
metric mapping detail.
