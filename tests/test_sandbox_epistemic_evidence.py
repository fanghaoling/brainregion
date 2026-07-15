from __future__ import annotations

import json

import pytest

from brainregion.sandbox.epistemic_evidence import EpistemicEvidenceWorkspace


def _evidence(
    action: str,
    scale: str,
    *,
    matched: bool,
    changed_cells: int,
) -> dict:
    return {
        "action": action,
        "matched": matched,
        "mismatch_fields": [] if matched else ["change_scale"],
        "expected": {"rule": "private model prediction"},
        "actual": {
            "change_scale": scale,
            "changed_cells": changed_cells,
            "total_cells": 100,
            "level_delta": 0,
            "state": "NOT_FINISHED",
            "private_note": "must not escape",
        },
    }


def test_workspace_deduplicates_objective_transition_and_aggregates_prediction_outcomes():
    workspace = EpistemicEvidenceWorkspace(max_events=4)

    first = workspace.record(
        _evidence("action1", "local", matched=False, changed_cells=2),
        step=1,
    )
    second = workspace.record(
        _evidence("action1", "local", matched=True, changed_cells=2),
        step=3,
    )

    assert first == second
    view = workspace.model_view()
    assert len(view["events"]) == 1
    event = view["events"][0]
    assert event == {
        "event_id": first,
        "action": "action1",
        "actual": {
            "change_scale": "local",
            "changed_cells": 2,
            "total_cells": 100,
            "level_delta": 0,
            "state": "NOT_FINISHED",
        },
        "observations": 2,
        "prediction_matches": 1,
        "prediction_mismatches": 1,
        "mismatch_fields": ["change_scale"],
        "first_step": 1,
        "last_step": 3,
    }
    serialized = json.dumps(view)
    assert "private model prediction" not in serialized
    assert "private_note" not in serialized
    metrics = workspace.public_metrics()
    assert metrics["events"] == 1
    assert metrics["observations"] == 2
    assert metrics["deduplicated_observations"] == 1


def test_workspace_evicts_least_recent_event_and_rejects_unattributed_feedback():
    workspace = EpistemicEvidenceWorkspace(max_events=2)
    local = workspace.record(
        _evidence("action1", "local", matched=True, changed_cells=2), step=0
    )
    workspace.record(
        _evidence("action2", "none", matched=True, changed_cells=0), step=1
    )
    assert workspace.record({"actual": {"change_scale": "global"}}, step=2) == ""
    assert (
        workspace.record(
            _evidence("action1", "local", matched=True, changed_cells=2), step=3
        )
        == local
    )
    workspace.record(
        _evidence("action3", "global", matched=False, changed_cells=30), step=4
    )

    assert [event["action"] for event in workspace.model_view()["events"]] == [
        "action1",
        "action3",
    ]
    metrics = workspace.public_metrics()
    assert metrics["events"] == 2
    assert metrics["evicted_events"] == 1
    assert metrics["evicted_observations"] == 1


def test_workspace_requires_positive_capacity():
    with pytest.raises(ValueError, match="positive integer"):
        EpistemicEvidenceWorkspace(max_events=0)
