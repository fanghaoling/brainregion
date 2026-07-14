from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from brainregion.sandbox import cleanup_run_dir, make_run_dir, materialize_fixture, run_agent
from brainregion.sandbox.fixtures import get_fixture
from brainregion.sandbox.phase_control import (
    CognitivePhase,
    ComputeTier,
    DifficultyVector,
    PhaseController,
    assess_task_difficulty,
)
from brainregion.sandbox.task import SandboxTask, WorktreeTask
from brainregion.workspace import read_text
from brainregion.workspace.files import scoped_workspace_root


def test_difficulty_vector_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="scope"):
        DifficultyVector(
            scope=1.1,
            ambiguity=0.0,
            novelty=0.0,
            risk=0.0,
            irreversibility=0.0,
            verification_gap=0.0,
        )


def test_task_assessment_is_metadata_only_and_explainable():
    sandbox = SandboxTask(
        id="small",
        goal="Fix the bounded defect.",
        files={"service.py": "pass\n"},
        tests={"test_service.py": "def test_ok(): assert True\n"},
    )
    worktree = WorktreeTask(
        id="repo",
        goal="Repair the repository behavior.",
        repo_path=".",
        test_args=[],
    )

    sandbox_difficulty = assess_task_difficulty(sandbox)
    worktree_difficulty = assess_task_difficulty(worktree)

    assert sandbox_difficulty.verification_gap == 0.20
    assert "isolated_sandbox" in sandbox_difficulty.reasons
    assert worktree_difficulty.scope > sandbox_difficulty.scope
    assert worktree_difficulty.risk > sandbox_difficulty.risk
    assert worktree_difficulty.verification_gap == 0.70
    assert "weak_verification_contract" in worktree_difficulty.reasons


def test_phase_controller_tracks_happy_path_and_failed_recovery():
    task = SandboxTask(
        id="flow",
        goal="Fix and verify.",
        files={"service.py": "pass\n"},
        tests={"test_service.py": "def test_ok(): assert True\n"},
    )
    controller = PhaseController.for_task(task)

    assert controller.phase is CognitivePhase.UNDERSTAND
    assert controller.tier is ComputeTier.STANDARD

    transition = controller.after_operation(step=0, operation="read_text")
    assert transition is not None and transition.current is CognitivePhase.PLAN

    transition = controller.before_operation(step=1, operation="apply_text_patch")
    assert transition is not None and transition.current is CognitivePhase.EXECUTE
    transition = controller.after_operation(
        step=1,
        operation="apply_text_patch",
        workspace_effect=True,
    )
    assert transition is not None and transition.current is CognitivePhase.VERIFY
    assert controller.tier is ComputeTier.DETERMINISTIC

    transition = controller.after_operation(
        step=2,
        operation="workspace_run_check",
        verification_passed=False,
    )
    assert transition is not None and transition.current is CognitivePhase.RECOVER
    assert controller.tier is ComputeTier.STRONG

    controller.before_operation(step=3, operation="apply_text_patch")
    controller.after_operation(
        step=3,
        operation="apply_text_patch",
        workspace_effect=True,
    )
    transition = controller.after_operation(
        step=4,
        operation="workspace_run_check",
        verification_passed=True,
    )
    assert transition is not None and transition.current is CognitivePhase.SYNTHESIZE
    assert [item.reason for item in controller.transitions] == [
        "initial_evidence_collected",
        "effectful_action_selected",
        "workspace_effect_requires_verification",
        "verification_failed",
        "effectful_action_selected",
        "workspace_effect_requires_verification",
        "verification_passed",
    ]


def test_repeated_failures_raise_stagnation_without_model_reasoning():
    controller = PhaseController.for_task(SandboxTask(id="x", goal="Inspect."))

    controller.observe_model_failure(step=0, reason="parse_error")
    controller.observe_model_failure(step=1, reason="parse_error")
    snapshot = controller.snapshot()

    assert snapshot["phase"] == "recover"
    assert snapshot["recommended_tier"] == "strong"
    assert snapshot["difficulty"]["stagnation"] == 1.0
    assert snapshot["changes_model_routing"] is False
    assert snapshot["contains_reasoning"] is False


def test_productive_investigation_leaves_plan_and_does_not_fake_stagnation():
    task = SandboxTask(
        id="investigation",
        goal="Inspect several files, patch one, and verify.",
        files={"settings.py": "VALUE = 1\n", "loader.py": "pass\n"},
        tests={"test_settings.py": "def test_ok(): assert True\n"},
    )
    controller = PhaseController.for_task(task)

    controller.after_operation(
        step=0,
        operation="search_text",
        target_is_new=True,
    )
    assert controller.phase is CognitivePhase.PLAN

    transition = controller.before_operation(step=1, operation="read_text")
    assert transition is not None
    assert transition.reason == "plan_execution_started"
    assert transition.current is CognitivePhase.EXECUTE
    controller.after_operation(
        step=1,
        operation="read_text",
        target_is_new=True,
    )
    controller.after_operation(
        step=2,
        operation="search_text",
        target_is_new=True,
    )

    assert controller.phase is CognitivePhase.EXECUTE
    assert controller.stagnation == 0.0
    assert controller.tier is ComputeTier.ECONOMY

    controller.after_operation(
        step=3,
        operation="read_text",
        target_is_new=False,
    )
    assert controller.stagnation == 0.5

    controller.before_operation(step=4, operation="apply_text_patch")
    transition = controller.after_operation(
        step=4,
        operation="apply_text_patch",
        workspace_effect=True,
        target_is_new=False,
    )
    assert transition is not None and transition.current is CognitivePhase.VERIFY
    assert controller.stagnation == 0.0
    assert controller.tier is ComputeTier.DETERMINISTIC


class _ScriptBackend:
    def __init__(self, script: list[str]) -> None:
        self.script = script
        self.index = 0

    async def complete_messages(self, messages, **kwargs):
        del messages
        content = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        return _Response(model=kwargs.get("model", "mock"), content=content)


class _Response:
    def __init__(self, *, model: str, content: str) -> None:
        self.model = model
        self.content = content
        self.error = None
        self.usage: dict = {}
        self.cost_usd = 0.0
        self.cost_source = None

    @property
    def ok(self) -> bool:
        return bool(self.content)


def _j(value: dict) -> str:
    return json.dumps(value)


def test_run_agent_records_phase_trajectory_and_transition_events(monkeypatch):
    import brainregion.sandbox.loop as loop_module

    task = get_fixture("off_by_one")
    run_dir = make_run_dir(prefix="brainregion-phase-control-")
    materialize_fixture(task, Path(run_dir))
    with scoped_workspace_root(run_dir):
        sha = read_text("ranges.py")["sha256"]
    script = [
        _j({"thought": "read", "tool": "read_text", "args": {"path": "ranges.py"}}),
        _j(
            {
                "thought": "fix",
                "tool": "apply_text_patch",
                "args": {
                    "path": "ranges.py",
                    "expected_sha256": sha,
                    "replacements": [
                        {"old_text": "range(start, end)", "new_text": "range(start, end + 1)"}
                    ],
                    "dry_run": False,
                },
            }
        ),
        _j(
            {
                "thought": "done",
                "done": True,
                "answer": "fixed",
            }
        ),
    ]
    events: list[tuple[str, dict]] = []

    def capture_event(event_type: str, **fields):
        events.append((event_type, fields))
        return {"type": event_type, **fields}

    monkeypatch.setattr(loop_module, "emit_event", capture_event)
    try:
        trajectory = asyncio.run(
            run_agent(
                _ScriptBackend(script),
                "mock",
                task,
                run_dir=run_dir,
                max_steps=3,
                verify_fn=lambda *_args, **_kwargs: {"tests_green": True},
            )
        )
    finally:
        cleanup_run_dir(run_dir)

    phase_control = trajectory.to_dict()["phase_control"]
    assert trajectory.tests_green is True
    assert phase_control["phase"] == "synthesize"
    assert phase_control["changes_model_routing"] is False
    assert trajectory.to_dict()["effort_routing_shadow"]["enabled"] is False
    assert not any(event_type == "sandbox.effort.shadow" for event_type, _fields in events)
    assert [item["to"] for item in phase_control["transitions"]] == [
        "plan",
        "execute",
        "verify",
        "synthesize",
    ]
    assert [(step.phase_at_call, step.phase_after) for step in trajectory.steps] == [
        ("understand", "plan"),
        ("plan", "verify"),
        ("verify", "synthesize"),
    ]
    phase_events = [event for event in events if event[0].startswith("sandbox.phase.")]
    assert phase_events[0][0] == "sandbox.phase.status"
    assert sum(event[0] == "sandbox.phase.transition" for event in phase_events) == 4
    assert phase_events[-1][1]["payload"]["reason"] == "run_verified"
    step_events = [fields["payload"] for event_type, fields in events if event_type == "sandbox.step"]
    assert step_events[-1]["phase"] == "synthesize"
    assert step_events[-1]["recommended_tier"] == "economy"
    assert isinstance(step_events[-1]["difficulty_score"], float)
