from __future__ import annotations

import asyncio
import json
from pathlib import Path

from brainregion.cli import build_parser
from brainregion.sandbox import cleanup_run_dir, make_run_dir, materialize_fixture, run_agent
from brainregion.sandbox.effort_routing import (
    PhaseEffortShadow,
    controls_for_tier,
    disabled_effort_shadow_metrics,
)
from brainregion.sandbox.fixtures import get_fixture
from brainregion.sandbox.phase_control import CognitivePhase, ComputeTier, DifficultyVector


def _difficulty(*, stagnation: float = 0.0) -> DifficultyVector:
    return DifficultyVector(
        scope=0.3,
        ambiguity=0.2,
        novelty=0.5,
        risk=0.2,
        irreversibility=0.1,
        verification_gap=0.2,
        stagnation=stagnation,
    )


def test_compute_tiers_map_to_provider_neutral_effort_controls():
    assert controls_for_tier(ComputeTier.DETERMINISTIC).to_dict() == {
        "thinking": False,
        "effort": None,
    }
    assert controls_for_tier(ComputeTier.ECONOMY).to_dict() == {
        "thinking": False,
        "effort": None,
    }
    assert controls_for_tier(ComputeTier.STANDARD).to_dict() == {
        "thinking": True,
        "effort": "medium",
    }
    assert controls_for_tier(ComputeTier.STRONG).to_dict() == {
        "thinking": True,
        "effort": "high",
    }


def test_shadow_trace_compares_controls_without_content():
    shadow = PhaseEffortShadow()
    understand = shadow.observe(
        step=0,
        phase=CognitivePhase.UNDERSTAND,
        difficulty=_difficulty(),
        actual_thinking=False,
        actual_effort=None,
    )
    execute = shadow.observe(
        step=1,
        phase=CognitivePhase.EXECUTE,
        difficulty=_difficulty(),
        actual_thinking=False,
        actual_effort=None,
    )
    recover = shadow.observe(
        step=2,
        phase=CognitivePhase.RECOVER,
        difficulty=_difficulty(stagnation=1.0),
        actual_thinking=False,
        actual_effort=None,
    )

    assert understand.recommended_tier is ComputeTier.STANDARD
    assert understand.would_change is True
    assert execute.recommended_tier is ComputeTier.ECONOMY
    assert execute.would_change is False
    assert recover.recommended_tier is ComputeTier.STRONG
    assert recover.would_change is True

    snapshot = shadow.snapshot()
    assert snapshot["mode"] == "shadow"
    assert snapshot["decision_count"] == 3
    assert snapshot["would_change_calls"] == 2
    assert snapshot["applied_change_calls"] == 0
    assert snapshot["agreement_calls"] == 1
    assert snapshot["by_phase"]["recover"]["recommended_tiers"] == {"strong": 1}
    assert snapshot["changes_model_routing"] is False
    assert snapshot["changes_inference_controls"] is False
    assert snapshot["contains_reasoning"] is False
    assert snapshot["contains_content"] is False


def test_active_routing_applies_recommended_controls_and_keeps_configured_values():
    router = PhaseEffortShadow(mode="active")

    decision = router.observe(
        step=0,
        phase=CognitivePhase.RECOVER,
        difficulty=_difficulty(stagnation=1.0),
        actual_thinking=False,
        actual_effort=None,
    )

    assert decision.to_dict()["configured"] == {"thinking": False, "effort": None}
    assert decision.to_dict()["effective"] == {"thinking": True, "effort": "high"}
    assert decision.to_dict()["actual"] == {"thinking": True, "effort": "high"}
    assert decision.to_dict()["control_scope"] == "backend_request"
    assert decision.to_dict()["recommendation_applied"] is True
    assert decision.to_dict()["controls_changed"] is True
    snapshot = router.snapshot()
    assert snapshot["mode"] == "active"
    assert snapshot["policy"] == "same_model_phase_effort_active_v1"
    assert snapshot["applied_change_calls"] == 1
    assert snapshot["changes_model_routing"] is False
    assert snapshot["changes_inference_controls"] is True
    assert snapshot["provider_execution_telemetry"] == "not_collected"


def test_recovery_only_policy_preserves_controls_until_recovery():
    router = PhaseEffortShadow(mode="active", activation_policy="recovery_only")

    understand = router.observe(
        step=0,
        phase=CognitivePhase.UNDERSTAND,
        difficulty=_difficulty(),
        actual_thinking=False,
        actual_effort=None,
    )
    recover = router.observe(
        step=1,
        phase=CognitivePhase.RECOVER,
        difficulty=_difficulty(stagnation=1.0),
        actual_thinking=False,
        actual_effort=None,
    )

    assert understand.recommended.thinking is True
    assert understand.activation_eligible is False
    assert understand.recommendation_applied is False
    assert understand.actual_thinking is False
    assert recover.activation_eligible is True
    assert recover.recommendation_applied is True
    assert recover.actual_effort == "high"
    snapshot = router.snapshot()
    assert snapshot["activation_policy"] == "recovery_only"
    assert snapshot["policy"] == "same_model_phase_effort_active_recovery_only_v1"
    assert snapshot["would_change_calls"] == 2
    assert snapshot["applied_change_calls"] == 1


def test_disabled_shadow_metrics_are_explicit_and_empty():
    assert disabled_effort_shadow_metrics() == {
        "enabled": False,
        "mode": "off",
        "policy": "same_model_phase_effort_shadow_v1",
        "decision_count": 0,
        "would_change_calls": 0,
        "applied_change_calls": 0,
        "agreement_calls": 0,
        "recommended_thinking_calls": 0,
        "actual_thinking_calls": 0,
        "by_phase": {},
        "decisions": [],
        "changes_model_routing": False,
        "changes_inference_controls": False,
        "control_scope": "backend_request",
        "provider_execution_telemetry": "not_collected",
        "contains_reasoning": False,
        "contains_content": False,
    }


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content
        self.error = None
        self.usage: dict = {}
        self.cost_usd = 0.0
        self.cost_source = None

    @property
    def ok(self) -> bool:
        return bool(self.content)


class _Backend:
    def __init__(self, script: list[str]) -> None:
        self.script = script
        self.index = 0
        self.kwargs: list[dict] = []

    async def complete_messages(self, messages, **kwargs):
        del messages
        self.kwargs.append(dict(kwargs))
        content = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        return _Response(content)


def test_run_agent_shadow_observes_recovery_but_preserves_backend_controls(monkeypatch):
    import brainregion.sandbox.loop as loop_module

    task = get_fixture("off_by_one")
    run_dir = make_run_dir(prefix="brainregion-effort-shadow-")
    materialize_fixture(task, Path(run_dir))
    backend = _Backend(
        [
            "not-json",
            json.dumps({"thought": "stop", "done": True, "answer": "not repaired"}),
        ]
    )
    events: list[tuple[str, dict]] = []

    def capture_event(event_type: str, **fields):
        events.append((event_type, fields))
        return {"type": event_type, **fields}

    monkeypatch.setattr(loop_module, "emit_event", capture_event)
    try:
        trajectory = asyncio.run(
            run_agent(
                backend,
                "mock",
                task,
                run_dir=run_dir,
                max_steps=2,
                thinking=False,
                effort=None,
                effort_routing_shadow=True,
                verify_fn=lambda *_args, **_kwargs: {"tests_green": False},
            )
        )
    finally:
        cleanup_run_dir(run_dir)

    assert [(call["thinking"], call["effort"]) for call in backend.kwargs] == [
        (False, None),
        (False, None),
    ]
    snapshot = trajectory.to_dict()["effort_routing_shadow"]
    assert [item["phase"] for item in snapshot["decisions"]] == [
        "understand",
        "recover",
    ]
    assert [item["recommended"]["effort"] for item in snapshot["decisions"]] == [
        "medium",
        "high",
    ]
    assert all(item["would_change"] for item in snapshot["decisions"])
    assert trajectory.steps[1].effort_routing_shadow["actual"] == {
        "thinking": False,
        "effort": None,
    }
    assert trajectory.steps[1].effort_routing_shadow["configured"] == {
        "thinking": False,
        "effort": None,
    }
    shadow_events = [fields["payload"] for event_type, fields in events if event_type == "sandbox.effort.shadow"]
    assert len(shadow_events) == 2
    assert shadow_events[-1]["phase"] == "recover"
    assert shadow_events[-1]["model"] == "mock"


def test_run_agent_active_routing_changes_controls_without_switching_model(monkeypatch):
    import brainregion.sandbox.loop as loop_module

    task = get_fixture("off_by_one")
    run_dir = make_run_dir(prefix="brainregion-effort-active-")
    materialize_fixture(task, Path(run_dir))
    backend = _Backend(
        [
            "not-json",
            json.dumps({"thought": "stop", "done": True, "answer": "not repaired"}),
        ]
    )
    events: list[tuple[str, dict]] = []

    def capture_event(event_type: str, **fields):
        events.append((event_type, fields))
        return {"type": event_type, **fields}

    monkeypatch.setattr(loop_module, "emit_event", capture_event)
    try:
        trajectory = asyncio.run(
            run_agent(
                backend,
                "mock",
                task,
                run_dir=run_dir,
                max_steps=2,
                thinking=False,
                effort=None,
                effort_routing_active=True,
                verify_fn=lambda *_args, **_kwargs: {"tests_green": False},
            )
        )
    finally:
        cleanup_run_dir(run_dir)

    assert [call["model"] for call in backend.kwargs] == ["mock", "mock"]
    assert [(call["thinking"], call["effort"]) for call in backend.kwargs] == [
        (True, "medium"),
        (True, "high"),
    ]
    snapshot = trajectory.to_dict()["effort_routing_shadow"]
    assert snapshot["mode"] == "active"
    assert snapshot["applied_change_calls"] == 2
    assert snapshot["changes_model_routing"] is False
    assert trajectory.steps[0].effort_routing_shadow["configured"] == {
        "thinking": False,
        "effort": None,
    }
    assert trajectory.steps[0].effort_routing_shadow["actual"] == {
        "thinking": True,
        "effort": "medium",
    }
    applied_events = [
        fields["payload"]
        for event_type, fields in events
        if event_type == "sandbox.effort.applied"
    ]
    assert [event["actual"]["effort"] for event in applied_events] == ["medium", "high"]


def test_run_agent_recovery_only_routing_activates_after_parse_failure(monkeypatch):
    import brainregion.sandbox.loop as loop_module

    task = get_fixture("off_by_one")
    run_dir = make_run_dir(prefix="brainregion-effort-recovery-only-")
    materialize_fixture(task, Path(run_dir))
    backend = _Backend(
        [
            "not-json",
            json.dumps({"thought": "stop", "done": True, "answer": "not repaired"}),
        ]
    )
    events: list[tuple[str, dict]] = []

    def capture_event(event_type: str, **fields):
        events.append((event_type, fields))
        return {"type": event_type, **fields}

    monkeypatch.setattr(loop_module, "emit_event", capture_event)
    try:
        trajectory = asyncio.run(
            run_agent(
                backend,
                "mock",
                task,
                run_dir=run_dir,
                max_steps=2,
                thinking=False,
                effort=None,
                effort_routing_active=True,
                effort_routing_policy="recovery_only",
                verify_fn=lambda *_args, **_kwargs: {"tests_green": False},
            )
        )
    finally:
        cleanup_run_dir(run_dir)

    assert [(call["thinking"], call["effort"]) for call in backend.kwargs] == [
        (False, None),
        (True, "high"),
    ]
    snapshot = trajectory.to_dict()["effort_routing_shadow"]
    assert snapshot["activation_policy"] == "recovery_only"
    assert snapshot["applied_change_calls"] == 1
    assert [event_type for event_type, _fields in events if event_type.startswith("sandbox.effort")] == [
        "sandbox.effort.shadow",
        "sandbox.effort.applied",
    ]


def test_cli_shadow_flag_is_opt_in_for_run_and_env():
    parser = build_parser()
    run_default = parser.parse_args(["sandbox", "run", "--main-brain", "mock"])
    run_enabled = parser.parse_args(
        ["sandbox", "run", "--main-brain", "mock", "--effort-routing-shadow"]
    )
    env_default = parser.parse_args(["sandbox", "env", "--main-brain", "mock"])
    env_enabled = parser.parse_args(
        ["sandbox", "env", "--main-brain", "mock", "--effort-routing-shadow"]
    )
    run_active = parser.parse_args(
        ["sandbox", "run", "--main-brain", "mock", "--effort-routing-active"]
    )
    env_active = parser.parse_args(
        [
            "sandbox",
            "env",
            "--main-brain",
            "mock",
            "--effort-routing-active",
            "--effort-routing-policy",
            "recovery_only",
        ]
    )

    assert run_default.effort_routing_shadow is False
    assert run_default.effort_routing_active is False
    assert run_enabled.effort_routing_shadow is True
    assert run_enabled.effort_routing_active is False
    assert env_default.effort_routing_shadow is False
    assert env_default.effort_routing_active is False
    assert env_enabled.effort_routing_shadow is True
    assert env_enabled.effort_routing_active is False
    assert run_active.effort_routing_active is True
    assert run_active.effort_routing_shadow is False
    assert run_active.effort_routing_policy == "phase"
    assert env_active.effort_routing_active is True
    assert env_active.effort_routing_shadow is False
    assert env_active.effort_routing_policy == "recovery_only"
