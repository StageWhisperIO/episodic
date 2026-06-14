# collector

Hooks (hook.py) are the primary capture mechanism for tool calls and file edits.
The OTel adapter is optional plumbing that adds token counts and cost data that
hooks cannot see, merging them into each session's meta so they appear in
episode.stats.

## Enable OTel in Claude Code

Set these environment variables before starting Claude Code:

```
CLAUDE_CODE_ENABLE_TELEMETRY=1
OTEL_METRICS_EXPORTER=otlp
OTEL_LOGS_EXPORTER=otlp
OTEL_EXPORTER_OTLP_PROTOCOL=http/json
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
```

## Run the receiver

```
python -m episodic.collector.otel 127.0.0.1 4318
```

Token and cost totals are written to session meta under `usage` and merge into
`episode.stats` automatically. This is background plumbing and is not exposed to
the user.
