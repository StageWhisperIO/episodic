# episode-store

The local, append-only store for captured sessions and finalized episodes.
Implementation: [`src/episodic/store.py`](../src/episodic/store.py) and
[`src/episodic/paths.py`](../src/episodic/paths.py).

## Layout

The store lives at `$EPISODIC_HOME` or `./.episodic/` (anchored to the nearest `.git`
or existing `.episodic`, so it stays stable across subdirectories):

```
.episodic/
  current                       # id of the most recently active session
  sessions/<session_id>/
    events.jsonl                # append-only capture log (one normalized event per line)
    meta.json                   # intent, agent, repo_state, usage, human_feedback, outcome
  episodes/
    <episode_id>.json           # finalized CodingEpisode (schema-validated)
    index.jsonl                 # one summary row per episode (for list / dashboard)
  exports/                      # dataset-exporters output
  replays/                      # replay-runner snapshots
```

## Why append-only events

Capture must be fast and crash-proof inside the agent: each hook appends one JSON line and
returns. Episodes are *derived* from events on demand (`episodic finalize` / `summary`),
so the expensive normalization (git diff, test parsing, reward) never blocks the session.

## API (stable)

```python
from episodic import store
store.append_event(event, start=None)
store.get_session(session_id) -> {"id", "meta", "events"}
store.save_episode(episode) / store.get_episode(id) / store.list_episodes()
store.load_episodes() -> list[CodingEpisode]
store.read_meta(id) / store.update_meta(id, patch) / store.get_current()
```
