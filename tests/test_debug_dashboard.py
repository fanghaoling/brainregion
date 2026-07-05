from __future__ import annotations

from brainregion.viz.debug_server import DebugDashboardOptions, build_debug_dashboard_html, build_snapshot_payload


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
    assert "BrainRegion Debug" in html
    assert "/api/snapshot" in html
    assert "data-refresh-ms=\"1200\"" in html
    assert "&lt;wake?&gt;" in html
    assert "localStorage" in html
    assert "function esc" in html
    assert "esc(r.region)" in html
    assert "esc(tools)" in html
    assert "Missed Wake" in html
    assert "src=" not in html
    assert "https://" not in html


def test_debug_snapshot_payload_merges_query_params(monkeypatch):
    calls = {}

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
