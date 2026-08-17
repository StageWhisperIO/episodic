from episodic import cli
from episodic.serving import server as serving_server


def _run_serve_cli(monkeypatch, argv):
    captured = {}

    def fake_serve(host, port, config=None, start=None):
        captured["host"] = host
        captured["port"] = port
        captured["config"] = config

    monkeypatch.setattr(serving_server, "serve", fake_serve)
    cli.main(argv)
    return captured


def test_cmd_serve_defaults(monkeypatch):
    captured = _run_serve_cli(monkeypatch, ["serve"])
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8000
    assert captured["config"] == {}


def test_cmd_serve_host_and_port_flags(monkeypatch):
    captured = _run_serve_cli(monkeypatch, ["serve", "--host", "0.0.0.0", "--port", "9001"])
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9001


def test_cmd_serve_wires_tier_flags(monkeypatch):
    captured = _run_serve_cli(monkeypatch, [
        "serve",
        "--distilled-backend", "ollama", "--distilled-base-url", "http://localhost:11434", "--distilled-model", "llama3",
        "--frontier-backend", "openai", "--frontier-model", "gpt-4o-mini",
    ])
    assert captured["config"]["distilled"] == {
        "backend": "ollama", "base_url": "http://localhost:11434", "model": "llama3",
    }
    assert captured["config"]["frontier"] == {"backend": "openai", "model": "gpt-4o-mini"}


def test_cmd_serve_wires_router_model_and_threshold(monkeypatch, tmp_path):
    model_path = tmp_path / "router_model.json"
    model_path.write_text("{}", encoding="utf-8")
    captured = _run_serve_cli(monkeypatch, [
        "serve", "--router-model", str(model_path), "--router-threshold", "0.7",
    ])
    assert captured["config"]["router_model_path"] == str(model_path)
    assert captured["config"]["router_threshold"] == 0.7


def test_cmd_serve_omits_router_keys_when_not_passed(monkeypatch):
    captured = _run_serve_cli(monkeypatch, ["serve"])
    assert "router_model_path" not in captured["config"]
    assert "router_threshold" not in captured["config"]


def test_cmd_serve_config_file_is_loaded_and_flags_layer_on_top(monkeypatch, tmp_path):
    config_path = tmp_path / "serve.json"
    config_path.write_text('{"distilled": {"base_url": "http://d.local"}, "escalate_chars": 500}', encoding="utf-8")

    captured = _run_serve_cli(monkeypatch, [
        "serve", "--config", str(config_path), "--distilled-model", "distilled-1",
    ])

    assert captured["config"]["escalate_chars"] == 500
    assert captured["config"]["distilled"]["base_url"] == "http://d.local"
    assert captured["config"]["distilled"]["model"] == "distilled-1"
