import json

from episodic import exporters, worldmodel
from episodic.testing import make_episode, make_population


def test_expand_turns_skips_empty_observations():
    ep = make_episode("ep_wm_1")
    samples = worldmodel.expand_turns(ep)
    assert len(samples) == len([s for s in ep["steps"] if (s["observation"] or "").strip()])
    assert all(s["target_observation"].strip() for s in samples)
    first = samples[0]
    assert first["turn_index"] >= 1
    assert "INTENT:" in first["history"]
    assert first["history"].rstrip().endswith("OBSERVATION:")


def test_wm_samples_one_per_trajectory_is_deterministic():
    pop = make_population(6, seed=2)
    a = worldmodel.wm_samples(pop, one_per_trajectory=True, seed=0)
    b = worldmodel.wm_samples(pop, one_per_trajectory=True, seed=0)
    assert len(a) == len(pop)
    assert [(s["episode_id"], s["turn_index"]) for s in a] == [(s["episode_id"], s["turn_index"]) for s in b]
    per_ep = {}
    for s in a:
        per_ep.setdefault(s["episode_id"], 0)
        per_ep[s["episode_id"]] += 1
    assert all(count == 1 for count in per_ep.values())


def test_ood_split_sources_are_disjoint():
    sources = [f"repo-{i}" for i in range(10)]
    pop = make_population(40, seed=1, sources=sources)
    train, holdout, mapping = worldmodel.ood_split(pop, holdout_frac=0.4, seed=7)
    train_sources = {worldmodel.source_key(e) for e in train}
    holdout_sources = {worldmodel.source_key(e) for e in holdout}
    assert train_sources.isdisjoint(holdout_sources)
    assert len(train) + len(holdout) == len(pop)
    assert train and holdout
    assert set(mapping) == train_sources | holdout_sources


def test_source_key_precedence():
    ep = make_episode("ep_src", source="repo-x")
    assert worldmodel.source_key(ep) == "repo-x"


def test_to_messages_assistant_is_target_observation():
    ep = make_episode("ep_wm_2")
    sample = worldmodel.expand_turns(ep)[-1]
    msg = worldmodel.to_messages(sample)
    roles = [m["role"] for m in msg["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert msg["messages"][2]["content"] == sample["target_observation"]
    assert msg["meta"]["source"] == sample["source"]


def test_wm_exporter_writes_messages(tmp_path):
    pop = make_population(5, seed=4)
    result = exporters.export(pop, "wm", str(tmp_path / "out"))
    assert result["format"] == "wm"
    rows = [json.loads(line) for line in open(result["files"][0]) if line.strip()]
    assert rows
    assert all(len(r["messages"]) == 3 for r in rows)
    assert all(r["messages"][0]["role"] == "system" for r in rows)
