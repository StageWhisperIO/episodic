import pytest

from episodic.collector.otel import (
    aggregate_usage,
    apply_usage_to_session,
    parse_metrics,
)
from episodic import store


SAMPLE_OTLP = {
    "resourceMetrics": [
        {
            "resource": {"attributes": []},
            "scopeMetrics": [
                {
                    "metrics": [
                        {
                            "name": "claude_code.token.usage",
                            "sum": {
                                "dataPoints": [
                                    {
                                        "asInt": "1000",
                                        "attributes": [
                                            {"key": "type", "value": {"stringValue": "input"}},
                                            {"key": "session.id", "value": {"stringValue": "s1"}},
                                        ],
                                    },
                                    {
                                        "asInt": "500",
                                        "attributes": [
                                            {"key": "type", "value": {"stringValue": "output"}},
                                            {"key": "session.id", "value": {"stringValue": "s1"}},
                                        ],
                                    },
                                ]
                            },
                        },
                        {
                            "name": "claude_code.cost.usage",
                            "sum": {
                                "dataPoints": [
                                    {
                                        "asDouble": 0.42,
                                        "attributes": [
                                            {"key": "session.id", "value": {"stringValue": "s1"}},
                                        ],
                                    }
                                ]
                            },
                        },
                    ]
                }
            ],
        }
    ]
}


def test_parse_metrics_returns_records():
    records = parse_metrics(SAMPLE_OTLP)
    assert len(records) >= 3


def test_aggregate_usage():
    records = parse_metrics(SAMPLE_OTLP)
    usage = aggregate_usage(records)
    assert usage["s1"]["input_tokens"] == 1000
    assert usage["s1"]["output_tokens"] == 500
    assert abs(usage["s1"]["cost_usd"] - 0.42) < 1e-9


def test_apply_usage_to_session(monkeypatch, tmp_path):
    monkeypatch.setenv("EPISODIC_HOME", str(tmp_path))
    records = parse_metrics(SAMPLE_OTLP)
    apply_usage_to_session(records)
    meta = store.read_meta("s1")
    assert meta["usage"]["input_tokens"] == 1000
    assert meta["usage"]["output_tokens"] == 500
    assert abs(meta["usage"]["cost_usd"] - 0.42) < 1e-9
