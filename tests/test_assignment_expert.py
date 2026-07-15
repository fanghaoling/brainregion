"""Wake-gated assignment runner integration tests."""

from __future__ import annotations

import asyncio
import json

import pytest

from brainregion.core.activation import ActivationPlan
from brainregion.core.assignment_expert import AssignmentExpertRunner
from brainregion.core.cognitive_workspace import CognitiveWorkspace
from brainregion.core.context import ContextBlock
from brainregion.core.context_loader import ActivatedContext, ContextLoadRecord
from brainregion.core.region_expert import RegionExpertEngine
from brainregion.core.region_reporting import RegionContextReceipt, RegionCoordinationBoard
from brainregion.core.task_coordination import TaskCoordinationBoard
from brainregion.providers.base import ModelResponse


class _Backend:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return ModelResponse(
            model=kwargs["model"],
            content=json.dumps(
                {
                    "state": "done",
                    "summary": "The parser evidence supports a bounded fallback.",
                    "implication": "No architecture change is required.",
                    "recommended_action": "Run the parser fallback test.",
                    "uncertainty": "The test result is still pending.",
                    "evidence_refs": ["memory:id:parser-evidence"],
                    "decision_scope": "routine",
                    "risk": "low",
                    "memory_impact": "supporting",
                    "reversible": True,
                    "repeated_failure": False,
                    "requires_user_choice": False,
                    "needs_more_context": False,
                }
            ),
            usage={"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120},
            cost_usd=0.005,
            cost_source="provider",
        )


def _activated(*, evidence_id: str, content: str) -> ActivatedContext:
    return ActivatedContext(
        activation=ActivationPlan(
            decisions=(),
            woken_regions=("memory", "debugging"),
            context_requests=(),
            trace={"models_called": False},
        ),
        blocks=(
            ContextBlock(
                source="memory",
                title="Private assignment evidence",
                content=content,
                metadata={"id": evidence_id, "region": "debugging"},
            ),
        ),
        loads=(
            ContextLoadRecord(
                skill_id="memory-recall",
                region="memory",
                status="loaded",
                provider="memory",
                blocks_loaded=1,
            ),
        ),
        trace={"models_called": False, "estimated_tokens": 30},
    )


def _runtime(*, with_context: bool = True):
    tasks = TaskCoordinationBoard()
    tasks.create_task({"task_id": "root", "goal": "Resolve the regression"})
    tasks.delegate(
        "root",
        {
            "assignment_id": "parser",
            "region": "debugging",
            "question": "Choose the next parser diagnostic.",
            "scope": "Only inspect parser loading.",
        },
    )
    tasks.delegate(
        "root",
        {
            "assignment_id": "network",
            "region": "debugging",
            "question": "Inspect endpoint behavior.",
        },
    )
    workspace = CognitiveWorkspace()
    coordination = RegionCoordinationBoard()
    if with_context:
        for assignment_id, evidence_id, content in (
            (
                "parser",
                "parser-evidence",
                "Private parser evidence says prose-wrapped JSON needs a fallback.",
            ),
            (
                "network",
                "network-evidence",
                "Private network evidence describes an unrelated endpoint timeout.",
            ),
        ):
            activated = _activated(evidence_id=evidence_id, content=content)
            delivery = workspace.stage(
                activated,
                task_id="root",
                audience="region",
                target_region="debugging",
                assignment_id=assignment_id,
            )
            coordination.record_receipt(
                RegionContextReceipt.from_activated(
                    activated,
                    task_id="root",
                    region="debugging",
                    assignment_id=assignment_id,
                    evidence_refs=tuple(delivery.entry["evidence_refs"]),
                )
            )
    backend = _Backend()
    runner = AssignmentExpertRunner(
        engine=RegionExpertEngine(backend=backend),
        tasks=tasks,
        workspace=workspace,
        coordination=coordination,
    )
    return runner, tasks, workspace, coordination, backend


def _run(runner: AssignmentExpertRunner, assignment_id: str):
    return asyncio.run(
        runner.run(
            task_id="root",
            assignment_id=assignment_id,
            model="expert-model",
            endpoint_id="relay",
        )
    ).to_dict()


def test_assignment_runner_sleeps_without_reading_another_assignments_wake(monkeypatch):
    runner, tasks, workspace, _, backend = _runtime()
    tasks.request_evidence_wake(
        "root",
        "parser",
        reason="expert_request",
        source="region_expert",
        ttl_reads=2,
    )
    reads = {"count": 0}
    original_read = workspace.read

    def counted_read(*args, **kwargs):
        reads["count"] += 1
        return original_read(*args, **kwargs)

    monkeypatch.setattr(workspace, "read", counted_read)

    result = _run(runner, "network")

    assert result["assignment_lifecycle"]["state"] == "sleeping"
    assert result["model_called"] is False
    assert reads["count"] == 0
    assert backend.calls == []
    assert tasks.evidence_wake_status("root", "parser")["wakes"][0][
        "remaining_reads"
    ] == 2


def test_assignment_runner_consumes_exact_wake_and_returns_only_grounded_report():
    runner, tasks, _, coordination, backend = _runtime()
    private = "Private parser evidence says prose-wrapped JSON needs a fallback."
    tasks.request_evidence_wake(
        "root",
        "parser",
        reason="expert_request",
        source="region_expert",
        ttl_reads=2,
    )

    first = _run(runner, "parser")
    second = _run(runner, "parser")
    sleeping = _run(runner, "parser")

    assert first["assignment_lifecycle"] == {
        "state": "awake",
        "wake_required": True,
        "wake_delivered": True,
        "wake_delivery_count": 1,
        "wake_request_ids": ["wake-00000001"],
        "wake_reasons": ["expert_request"],
        "wake_sources": ["region_expert"],
        "pending_wake_requests": 1,
        "pending_provider_reads": 1,
        "contains_context_content": False,
        "authorization_boundary": False,
    }
    assert second["assignment_lifecycle"]["pending_wake_requests"] == 0
    assert second["assignment_lifecycle"]["pending_provider_reads"] == 0
    assert sleeping["assignment_lifecycle"]["state"] == "sleeping"
    assert len(backend.calls) == 2
    assert private in backend.calls[0]["user"]
    assert "Private network evidence" not in backend.calls[0]["user"]
    assert "Choose the next parser diagnostic" in backend.calls[0]["user"]
    assert "Only inspect parser loading" in backend.calls[0]["user"]
    assert first["published_report"]["report"]["assignment_id"] == "parser"
    assert first["published_report"]["report"]["evidence_refs"] == [
        "memory:id:parser-evidence"
    ]
    assert private not in json.dumps(first)
    assert coordination.reports("root", assignment_id="parser")["count"] == 2


def test_assignment_runner_preserves_wake_while_waiting_for_private_context():
    runner, tasks, _, coordination, backend = _runtime(with_context=False)
    tasks.request_evidence_wake(
        "root",
        "parser",
        reason="explicit_recall",
        source="main_brain",
    )

    result = _run(runner, "parser")
    pending = tasks.evidence_wake_status("root", "parser")

    assert result["assignment_lifecycle"]["state"] == "waiting_context"
    assert result["assignment_lifecycle"]["wake_delivered"] is False
    assert result["assignment_lifecycle"]["pending_wake_requests"] == 1
    assert result["assignment_lifecycle"]["pending_provider_reads"] == 1
    assert result["model_called"] is False
    assert result["published_report"]["decision"]["action"] == "request_context"
    assert pending["wakes"][0]["remaining_reads"] == 1
    assert coordination.reports("root", assignment_id="parser")["count"] == 1
    assert backend.calls == []


def test_assignment_runner_rejects_mismatched_snapshot_before_consuming_wake():
    runner, tasks, workspace, _, backend = _runtime()
    tasks.request_evidence_wake(
        "root",
        "parser",
        reason="expert_request",
        source="region_expert",
    )
    network_view = workspace.read(
        "root",
        consumer="region",
        region="debugging",
        assignment_id="network",
        max_context_tokens=2000,
        max_blocks=12,
    )

    with pytest.raises(ValueError, match="assignment evidence view must match"):
        asyncio.run(
            runner.run(
                task_id="root",
                assignment_id="parser",
                model="expert-model",
                evidence_view=network_view,
            )
        )

    pending = tasks.evidence_wake_status("root", "parser")
    assert pending["wakes"][0]["remaining_reads"] == 1
    assert backend.calls == []
