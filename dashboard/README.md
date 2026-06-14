# Episodic Dashboard

A local web dashboard for browsing and annotating Episodic episodes.

## Starting the dashboard

```
episodic dashboard
```

Custom port:

```
episodic dashboard --port 8800
```

The dashboard reads episodes from the local `.episodic` store in your home directory (or `EPISODIC_HOME` if set).

## Routes

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Episode index table |
| GET | `/episode/<id>` | Episode detail page |
| GET | `/api/episodes` | JSON list of all index rows |
| GET | `/api/episode/<id>` | JSON full episode |
| POST | `/api/feedback` | Record feedback (body: `{episode_id, label, note?}`) |

## One-click feedback

The episode detail page (`/episode/<id>`) shows seven feedback buttons:

- `useful` — the agent did exactly what was needed
- `wrong` — the output was incorrect or harmful
- `too_broad` — the agent changed too much
- `too_slow` — the agent took too many steps
- `needed_human_rescue` — a human had to intervene
- `accepted_as_is` — accepted without any edits
- `accepted_after_edits` — accepted after manual corrections

Clicking a button POSTs to `/api/feedback` via a small inline `fetch()` call (no external libraries). The reward vector is recomputed and the episode is re-saved immediately.
