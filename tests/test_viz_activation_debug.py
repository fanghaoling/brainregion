from __future__ import annotations

import pytest

from brainregion.viz import build_snapshot, render_html


@pytest.fixture
def root(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))
    return tmp_path


def test_build_snapshot_projects_activation_strength(root, monkeypatch):
    def _wake_with_matrix(**_k):
        return {
            "activated_regions": {
                "woken": ["memory"],
                "retrieved": [{"id": "memory", "score": 8, "source": "text"}],
                "escalated": ["memory"],
                "shadow": [],
                "reasons": {"memory": "matched 4 trigger(s)"},
                "confidence": {"memory": 1.0},
            },
            "wake_metrics": {"hit": ["memory"], "missed": [], "false_wake": [],
                             "metrics_status": "scored"},
            "suggested_actions": [
                {"tool": "recall_experiences", "source_regions": ["memory"],
                 "requires_user_approval": True},
            ],
            "trace": {"shadow_promoted": 0, "sentinel_hits": [], "models_called": False},
        }

    monkeypatch.setattr("brainregion.inspector.activation.wake_gate", _wake_with_matrix)
    snap = build_snapshot(problem="整理项目理解和记忆卡片")
    by_region = {r.region: r for r in snap.regions}
    assert by_region["memory"].phase == "woken"
    assert by_region["memory"].confidence == 1.0
    assert by_region["memory"].score == 8
    assert by_region["memory"].suggested_actions == 1
    assert by_region["memory"].action_tools == ("recall_experiences",)

    html_out = render_html(snap)
    assert "intensity 100%" in html_out
    assert "models_called" in html_out
    assert "recall_experiences" in html_out
