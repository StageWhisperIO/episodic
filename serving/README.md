# Episodic Serving

A thin, backend-agnostic OpenAI-compatible proxy/router that fronts the model `episodic loop`
just promoted. It never holds weights itself — it forwards to an external backend (`openai`,
`ollama`, `vllm`, `tinker`) reachable at a configured `base_url` (or, for `tinker`, a saved
sampler/model path via the SDK).

## Starting the proxy

```
episodic serve
```

Custom host/port and a distilled/frontier config:

```
episodic serve --port 8000 \
  --distilled-backend ollama --distilled-base-url http://localhost:11434 \
  --frontier-backend openai --frontier-model gpt-4o-mini
```

Or point `--config` at a JSON file / inline JSON with `distilled` / `frontier` blocks
(same shape `episodic loop`/`episodic train` already use for `--config`).

If `distilled.model` is left unset, the router auto-fills it from the most recent
`promoted.json`'s `served_ref` (falling back to `model_dir`) written by `episodic loop`.

## Routes

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | OpenAI-compatible chat completion (streaming + non-streaming) |
| GET | `/v1/models` | Lists the two configured tiers (`distilled`, `frontier`) |

## Two-tier router

Every request routes to the `distilled` (promoted) tier by default. A request can force a tier
by setting `"model": "frontier"` (or `"distilled"`), or the router escalates automatically via
a cheap signal (`episodic_escalate: true` in the request body, or a request longer than
`escalate_chars`). A custom `escalate(payload, config) -> bool` callable can be supplied in the
config for more advanced routing. The learned, cost-aware router is Phase 2.

See `src/episodic/serving/` for the implementation: `base.py` (shared HTTP backend + OpenAI
response/SSE helpers), `openai.py` / `vllm.py` / `ollama.py` / `tinker.py` (adapters, registered
the same way `src/episodic/trainers/__init__.py` registers trainers), `router.py` (two-tier
routing + `served_ref` resolution), `server.py` (the `ThreadingHTTPServer` handler, same pattern
as `src/episodic/dashboard/server.py`).
