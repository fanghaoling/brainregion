from __future__ import annotations

import pytest

from brainregion.core.context_pressure import (
    ContextPressureObserver,
    disabled_context_pressure_metrics,
)


def test_context_pressure_observes_saturation_without_returning_context(monkeypatch):
    events = []
    monkeypatch.setattr(
        "brainregion.core.context_pressure.emit_event",
        lambda event_type, **fields: events.append((event_type, fields)),
    )
    observer = ContextPressureObserver(
        model_context_limits={"relay/expert-model": 1000}
    )

    sample = observer.observe(
        step=3,
        task_id="task-1",
        assignment_id="debug-1",
        region="Debugging",
        model="expert-model",
        endpoint_id="relay",
        context_tokens=900,
        context_budget_tokens=1000,
        blocks_loaded=8,
        block_budget=8,
        context_truncated=True,
        usage={"input_tokens": 800, "output_tokens": 20},
        attempt=2,
        model_called=True,
        error_observed=True,
        context_status="loaded",
        lifecycle_state="awake",
    )

    assert sample.pressure_score == 0.945
    assert sample.pressure_band == "saturated"
    assert sample.model_window_fill_ratio == 0.8
    assert set(sample.signals) == {
        "context_budget_near_limit",
        "block_budget_near_limit",
        "context_truncated",
        "model_window_near_limit",
        "repeated_attempt",
        "error_observed",
    }
    payload = sample.to_dict()
    assert payload["changes_model_input"] is False
    assert payload["changes_routing"] is False
    assert payload["context_content_returned"] is False
    assert "content" not in payload
    assert events[0][0] == "context.pressure_observed"
    assert events[0][1]["region_id"] == "debugging"


def test_context_pressure_tracks_growth_and_bounded_region_model_history():
    observer = ContextPressureObserver(max_samples=1, emit_events=False)
    observer.observe(
        step=1,
        task_id="task-1",
        assignment_id="a-1",
        region="review",
        model="expert-model",
        context_tokens=100,
        context_budget_tokens=1000,
        blocks_loaded=1,
        block_budget=8,
        usage={"input_tokens": 100, "output_tokens": 10},
        model_called=True,
    )
    latest = observer.observe(
        step=2,
        task_id="task-2",
        assignment_id="a-2",
        region="review",
        model="expert-model",
        context_tokens=200,
        context_budget_tokens=1000,
        blocks_loaded=2,
        block_budget=8,
        usage={"input_tokens": 150, "output_tokens": 10},
        model_called=True,
    )

    assert latest.input_growth_ratio == 1.5
    assert "input_tokens_growing" in latest.signals
    snapshot = observer.snapshot()
    assert snapshot["sample_count"] == 1
    assert snapshot["total_observations"] == 2
    assert snapshot["dropped_sample_count"] == 1
    assert snapshot["model_capacity_coverage_rate"] == 0.0
    assert snapshot["score_interpretation"] == (
        "risk_proxy_not_measured_model_fatigue"
    )
    assert snapshot["region_models"][0]["peak_input_tokens"] == 150


def test_context_pressure_rejects_invalid_limits_and_can_be_disabled():
    with pytest.raises(ValueError, match="model context limit"):
        ContextPressureObserver(model_context_limits={"model": 0})
    with pytest.raises(ValueError, match="context_tokens"):
        ContextPressureObserver(emit_events=False).observe(
            step=0,
            task_id="task",
            assignment_id="assignment",
            region="review",
            model="model",
            context_tokens=-1,
        )

    disabled = disabled_context_pressure_metrics()
    assert disabled["enabled"] is False
    assert disabled["changes_model_input"] is False
    assert disabled["changes_routing"] is False


def test_context_pressure_observer_builds_exact_limits_from_model_routes():
    observer = ContextPressureObserver.from_model_routes(
        {
            "resolved_panel": [
                {
                    "model": "same-model",
                    "endpoint_id": None,
                    "profile": {"context_window_tokens": 1000},
                },
                {
                    "model": "same-model",
                    "endpoint_id": "relay",
                    "profile": {"context_window_tokens": 2000},
                },
                {
                    "model": "unknown-model",
                    "endpoint_id": "relay",
                    "profile": {},
                },
            ]
        },
        emit_events=False,
    )

    bare = observer.observe(
        step=0,
        task_id="task",
        assignment_id="bare",
        region="planning",
        model="same-model",
        usage={"input_tokens": 500},
        model_called=True,
    )
    relayed = observer.observe(
        step=1,
        task_id="task",
        assignment_id="relay",
        region="planning",
        model="same-model",
        endpoint_id="relay",
        usage={"input_tokens": 500},
        model_called=True,
    )

    assert bare.model_context_limit_tokens == 1000
    assert bare.model_window_fill_ratio == 0.5
    assert relayed.model_context_limit_tokens == 2000
    assert relayed.model_window_fill_ratio == 0.25


def test_explicit_context_limit_wins_over_route_metadata():
    observer = ContextPressureObserver.from_model_routes(
        {
            "resolved_panel": [
                {
                    "model": "model",
                    "endpoint_id": "relay",
                    "profile": {"context_window_tokens": 2000},
                }
            ]
        },
        model_context_limits={"relay/model": 3000},
        emit_events=False,
    )

    sample = observer.observe(
        step=0,
        task_id="task",
        assignment_id="assignment",
        region="planning",
        model="model",
        endpoint_id="relay",
        usage={"input_tokens": 1500},
        model_called=True,
    )

    assert sample.model_context_limit_tokens == 3000
    assert sample.model_window_fill_ratio == 0.5
