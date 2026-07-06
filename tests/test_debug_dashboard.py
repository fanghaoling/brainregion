from __future__ import annotations

from brainregion.viz.debug_server import (
    DebugDashboardOptions,
    build_debug_dashboard_html,
    build_model_calls_payload,
    build_snapshot_payload,
    summarize_model_events,
)


def test_debug_dashboard_html_is_self_contained():
    html = build_debug_dashboard_html(
        DebugDashboardOptions(
            problem="<wake?>",
            goal="observe regions",
            gold_regions=("memory", "debugging"),
            refresh_ms=1200,
        )
    )

    assert html.startswith("<!DOCTYPE html>")
    assert "BrainRegion 调试面板" in html
    assert "/api/snapshot" in html
    assert "data-refresh-ms=\"1200\"" in html
    assert "&lt;wake?&gt;" in html
    assert "localStorage" in html
    assert "EventSource" in html
    assert "/api/events/stream" in html
    assert "/api/events?limit=50" in html
    assert "/api/models?limit=5000&recent=20" in html
    assert "模型调用面板" in html
    assert "model-summary" in html
    assert "model-rows" in html
    assert "recent-model-calls" in html
    assert "function esc" in html
    assert "esc(r.region)" in html
    assert "esc(tools)" in html
    assert "漏唤醒" in html
    assert "调用状态" in html
    assert "脑区状态" in html
    assert "激活强度" in html
    assert "src=" not in html
    assert "https://" not in html


def test_debug_snapshot_payload_merges_query_params(monkeypatch):
    calls = {}
    emitted = []

    class FakeSnapshot:
        def to_dict(self):
            return {
                "schema_version": 3,
                "regions": [{"region": "memory", "confidence": 1.0, "score": 10}],
                "activation": {"call_status": {"woken_count": 1}},
            }

    def fake_build_snapshot(**kwargs):
        calls.update(kwargs)
        return FakeSnapshot()

    monkeypatch.setattr("brainregion.viz.debug_server.build_snapshot", fake_build_snapshot)
    monkeypatch.setattr(
        "brainregion.viz.debug_server.emit_event",
        lambda event_type, **fields: emitted.append((event_type, fields)),
    )
    options = DebugDashboardOptions(problem="default", gold_regions=("debugging",), top_k=3)

    payload = build_snapshot_payload(
        options,
        {"problem": ["override"], "gold_regions": ["memory,review"], "top_k": ["8"]},
    )

    assert calls["problem"] == "override"
    assert calls["gold_regions"] == ["memory", "review"]
    assert calls["top_k"] == 8
    assert payload["debug"]["query"]["problem"] == "override"
    assert payload["debug"]["query"]["gold_regions"] == ["memory", "review"]
    assert payload["debug"]["query"]["top_k"] == 8
    assert payload["debug"]["refresh_ms"] == options.refresh_ms
    assert [event[0] for event in emitted] == [
        "dashboard.snapshot_built",
        "dashboard.call_status",
        "region.activation",
    ]


def test_model_call_summary_aggregates_usage_cost_and_failures():
    events = [
        {
            "sequence": 1,
            "timestamp": "2026-01-01T00:00:00Z",
            "type": "model.call_started",
            "model": "claude-opus-4-8",
            "payload": {
                "provider": "anthropic",
                "resolved_model": "anthropic/claude-opus-4-8",
                "endpoint_id": "modelbridge_anthropic",
            },
        },
        {
            "sequence": 2,
            "timestamp": "2026-01-01T00:00:01Z",
            "type": "model.call_finished",
            "model": "claude-opus-4-8",
            "payload": {
                "provider": "anthropic",
                "model": "claude-opus-4-8",
                "resolved_model": "anthropic/claude-opus-4-8",
                "canonical_model": "claude-opus-4-8",
                "endpoint_id": "modelbridge_anthropic",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 25,
                    "total_tokens": 125,
                    "cached_tokens": 30,
                    "reasoning_tokens": 5,
                },
                "cost_usd": 0.0042,
                "cost_source": "provider",
                "latency_ms": 420.5,
                "status": "ok",
            },
        },
        {
            "sequence": 3,
            "timestamp": "2026-01-01T00:00:02Z",
            "type": "model.call_failed",
            "model": "gpt-5.5",
            "payload": {
                "provider": "openai",
                "model": "gpt-5.5",
                "resolved_model": "openai/gpt-5.5",
                "canonical_model": "gpt-5.5",
                "usage": {},
                "cost_usd": None,
                "cost_source": "missing_price",
                "latency_ms": 120.0,
                "status": "error",
                "error": "TimeoutError",
            },
        },
    ]

    payload = summarize_model_events(events, recent_limit=2)

    assert payload["totals"]["started_calls"] == 1
    assert payload["totals"]["successful_calls"] == 1
    assert payload["totals"]["failed_calls"] == 1
    assert payload["totals"]["input_tokens"] == 100
    assert payload["totals"]["output_tokens"] == 25
    assert payload["totals"]["total_tokens"] == 125
    assert payload["totals"]["cached_tokens"] == 30
    assert payload["totals"]["reasoning_tokens"] == 5
    assert payload["totals"]["cost_usd"] == 0.0042
    assert payload["totals"]["missing_cost_calls"] == 1
    assert payload["totals"]["avg_latency_ms"] == 270.25

    by_model = {item["canonical_model"]: item for item in payload["models"]}
    assert by_model["claude-opus-4-8"]["label"] == "modelbridge_anthropic/claude-opus-4-8"
    assert by_model["claude-opus-4-8"]["cost_sources"] == ["provider"]
    assert by_model["gpt-5.5"]["failed_calls"] == 1
    assert by_model["gpt-5.5"]["last_error"] == "TimeoutError"
    assert [item["type"] for item in payload["recent"]] == ["model.call_failed", "model.call_finished"]


def test_build_model_calls_payload_reads_runtime_events(monkeypatch):
    captured = {}

    def fake_list_events(*, after_sequence, limit):
        captured["after_sequence"] = after_sequence
        captured["limit"] = limit
        return [{"sequence": 9, "type": "model.call_started", "model": "gpt-5.5", "payload": {}}]

    monkeypatch.setattr("brainregion.viz.debug_server.list_events", fake_list_events)

    payload = build_model_calls_payload({"after": ["7"], "limit": ["50"], "recent": ["3"]})

    assert captured == {"after_sequence": 7, "limit": 50}
    assert payload["debug"]["after_sequence"] == 7
    assert payload["debug"]["event_limit"] == 50
    assert payload["debug"]["recent_limit"] == 3
    assert payload["totals"]["started_calls"] == 1


def test_cli_debug_subcommand_wires_to_dashboard(monkeypatch):
    from brainregion import cli
    from brainregion.viz import debug_server

    called = {}

    def fake_serve(options, *, open_browser=False):
        called["options"] = options
        called["open_browser"] = open_browser

    monkeypatch.setattr(debug_server, "serve_debug_dashboard", fake_serve)
    args = cli.build_parser().parse_args(
        [
            "debug",
            "--host", "127.0.0.1",
            "--port", "9876",
            "--problem", "route memory",
            "--gold-regions", "memory,debugging",
            "--top-k", "7",
            "--refresh-ms", "1500",
            "--open",
        ]
    )

    cli.run_debug(args)

    assert called["open_browser"] is True
    assert called["options"].host == "127.0.0.1"
    assert called["options"].port == 9876
    assert called["options"].problem == "route memory"
    assert called["options"].gold_regions == ("memory", "debugging")
    assert called["options"].top_k == 7
    assert called["options"].refresh_ms == 1500
