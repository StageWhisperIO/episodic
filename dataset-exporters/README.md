# Episodic Dataset Exporters

Export `CodingEpisode` records to training-ready formats for fine-tuning, RLHF, and reward modelling.

Code lives in `src/episodic/exporters`.

## CLI

```bash
episodic export-episode --format <fmt> --all
```

Replace `<fmt>` with one of: `jsonl`, `sft`, `dpo`, `reward`, `rlds`, `parquet`.

## Formats

### jsonl

Every episode exported verbatim, one JSON object per line.

```json
{"id": "ep_abc123", "intent": "Add retry logic", "steps": [...], "outcome": {"status": "merged"}, ...}
```

```bash
episodic export-episode --format jsonl --all
```

---

### sft (Supervised Fine-Tuning)

Only episodes that pass the `is_good` quality filter. Each record is a chat-style pair.

```json
{
  "messages": [
    {"role": "user", "content": "Add retry logic to the http client"},
    {"role": "assistant", "content": "USER: Add retry logic...\nACTION Edit({\"file_path\": \"src/http.py\"})\nOBS: applied"}
  ],
  "meta": {"episode_id": "ep_abc123", "reward": 0.82}
}
```

```bash
episodic export-episode --format sft --all
```

---

### dpo (Direct Preference Optimisation)

Episodes grouped by normalised intent. Within each group the highest-reward good episode becomes `chosen`; the lowest-reward bad (or lowest overall) episode becomes `rejected`. Groups with no usable pair are skipped.

```json
{
  "prompt": "Add retry logic to the http client",
  "chosen": "USER: Add retry...\nACTION Edit(...)\nOBS: applied",
  "rejected": "USER: Add retry...\nACTION Edit(...)\nOBS: error",
  "meta": {"chosen_id": "ep_abc123", "rejected_id": "ep_xyz789", "chosen_reward": 0.82, "rejected_reward": 0.1}
}
```

```bash
episodic export-episode --format dpo --all
```

---

### reward

Every episode with its full reward vector and scalar reward. Used to train reward models.

```json
{
  "prompt": "Add retry logic to the http client",
  "trajectory": "USER: Add retry...\nACTION Edit(...)\nOBS: applied",
  "reward_vector": {"test_pass": 1.0, "human_label": 1.0, "composite": 0.82, ...},
  "scalar_reward": 0.82
}
```

```bash
episodic export-episode --format reward --all
```

---

### rlds (RLDS / TFDS step format)

Each episode becomes a sequence of RL transition steps. The terminal step carries the composite reward; all earlier steps have reward `0.0`.

```json
{
  "episode_id": "ep_abc123",
  "steps": [
    {"observation": "", "action": {"tool": null, "input": {"prompt": "..."}, "type": "user_prompt"}, "reward": 0.0, "is_first": true, "is_last": false, "is_terminal": false, "discount": 1.0},
    {"observation": "applied", "action": {"tool": "Edit", "input": {"file_path": "src/http.py"}, "type": "file_edit"}, "reward": 0.82, "is_first": false, "is_last": true, "is_terminal": true, "discount": 1.0}
  ]
}
```

```bash
episodic export-episode --format rlds --all
```

---

### parquet

Flattened columnar export: `id`, `intent`, `agent`, `branch`, `outcome_status`, `composite`, `test_pass`, `file_edits`, `tests_run`, `additions`, `deletions`.

Requires `pyarrow`. If `pyarrow` is not installed the exporter degrades gracefully and writes `episodes.jsonl` instead, setting `result["fallback"]` in the return value.

```bash
pip install pyarrow
episodic export-episode --format parquet --all
```

---

## Python API

```python
from episodic import store, exporters

episodes = store.load_episodes()
result = exporters.export(episodes, fmt="sft", out_dir="exports/sft")
print(result)
# {"format": "sft", "out_dir": "exports/sft", "files": ["exports/sft/sft.jsonl"], "count": 12}
```

## Quality helpers

| Helper | Returns `True` when |
|---|---|
| `is_good(ep)` | composite ≥ 0.5, or status in `{accepted, merged}`, or feedback label in `{useful, accepted_as_is, accepted_after_edits}` |
| `is_bad(ep)` | status in `{failed, reverted}`, or feedback label `wrong` |
| `norm_intent(ep)` | lowercased, stripped intent string (used for DPO grouping) |
| `trajectory_text(ep)` | `"USER: <intent>\nACTION <tool>(<input>)\nOBS: <obs>\n..."` |
