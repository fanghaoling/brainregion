"""Wake-gated assignment runner integration tests."""

from __future__ import annotations

import asyncio
import json

import pytest

from brainregion.core.activation import ActivationPlan
from brainregion.core.assignment_expert import AssignmentExpertRunner
from brainregion.core.cognitive_workspace import CognitiveWorkspace
from brainregion.core.context import ContextBlock, ContextQuery, RetrieveResult
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


class _MemoryProvider:
    def __init__(self, *, blocks: list[ContextBlock] | None = None, error: str = "") -> None:
        self.blocks = list(blocks or [])
        self.error = error
        self.calls: list[ContextQuery] = []

    def retrieve(self, query: ContextQuery) -> RetrieveResult:
        self.calls.append(query)
        if self.error:
            raise RuntimeError(self.error)
        return RetrieveResult(
            provider="memory",
            blocks=list(self.blocks),
            meta={"candidates_before_top_k": len(self.blocks)},
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


def _runtime(
    *,
    with_context: bool = True,
    memory_provider: _MemoryProvider | None = None,
    parser_memory_request: dict | None = None,
):
    tasks = TaskCoordinationBoard()
    tasks.create_task({"task_id": "root", "goal": "Resolve the regression"})
    parser_assignment = {
        "assignment_id": "parser",
        "region": "debugging",
        "question": "Choose the next parser diagnostic.",
        "scope": "Only inspect parser loading.",
    }
    if parser_memory_request is not None:
        parser_assignment["memory_request"] = parser_memory_request
    tasks.delegate("root", parser_assignment)
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
        memory_provider=memory_provider,
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
    status = tasks.status("root")
    assert status["assignments"][0]["status"] == "done"
    assert status["task"]["status"] == "working"


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
    status = tasks.status("root")
    assert status["assignments"][0]["status"] == "working"
    assert status["task"]["status"] == "working"


def test_assignment_runner_retrieves_stages_and_retries_exact_private_context():
    private = "Private parser evidence says prose-wrapped JSON needs a fallback."
    provider = _MemoryProvider(
        blocks=[
            ContextBlock(
                source="memory",
                title="Parser failure lesson",
                content=private,
                metadata={
                    "id": "parser-evidence",
                    "region": "debugging",
                    "selectors": ["failure_lessons"],
                },
            )
        ]
    )
    runner, tasks, workspace, coordination, backend = _runtime(
        with_context=False,
        memory_provider=provider,
        parser_memory_request={
            "query": "parser fallback regression",
            "purpose": "reuse failure lessons",
            "regions": ["memory", "debugging"],
            "selectors": ["failure_lessons"],
            "top_k": 2,
            "max_context_tokens": 900,
        },
    )
    tasks.request_evidence_wake(
        "root",
        "parser",
        reason="expert_request",
        source="region_expert",
    )

    result = _run(runner, "parser")

    assert result["assignment_lifecycle"]["state"] == "awake"
    assert result["assignment_lifecycle"]["pending_wake_requests"] == 0
    assert result["context_retrieval"]["status"] == "loaded"
    assert result["context_retrieval"]["provider"] == "memory"
    assert result["context_retrieval"]["blocks_staged"] == 1
    assert result["context_retrieval"]["truncated"] is False
    assert result["context_retrieval"]["reason"] == ""
    assert result["context_retrieval"]["contains_context_content"] is False
    assert result["context_retrieval"]["models_called"] is False
    assert result["context_retrieval"]["estimated_tokens"] > 0
    assert result["model_called"] is True
    assert len(provider.calls) == 1
    assert provider.calls[0].text == "parser fallback regression"
    assert provider.calls[0].regions == frozenset({"memory", "debugging"})
    assert provider.calls[0].selectors == ("failure_lessons",)
    assert provider.calls[0].top_k == 2
    assert private in backend.calls[0]["user"]
    assert private not in json.dumps(result)
    assert coordination.reports("root", assignment_id="parser")["count"] == 1
    network_view = workspace.read(
        "root",
        consumer="region",
        region="debugging",
        assignment_id="network",
    )
    assert network_view.blocks == ()


@pytest.mark.parametrize(
    ("provider", "expected_status"),
    [
        (_MemoryProvider(), "empty"),
        (_MemoryProvider(error="memory unavailable"), "failed"),
    ],
)
def test_assignment_runner_preserves_wake_when_memory_retrieval_has_no_context(
    provider,
    expected_status,
):
    runner, tasks, _, _, backend = _runtime(
        with_context=False,
        memory_provider=provider,
        parser_memory_request={"query": "missing parser evidence"},
    )
    tasks.request_evidence_wake(
        "root",
        "parser",
        reason="explicit_recall",
        source="main_brain",
    )

    result = _run(runner, "parser")

    assert result["assignment_lifecycle"]["state"] == "waiting_context"
    assert result["assignment_lifecycle"]["pending_provider_reads"] == 1
    assert result["context_retrieval"]["status"] == expected_status
    assert result["context_retrieval"]["blocks_staged"] == 0
    if expected_status == "failed":
        assert result["context_retrieval"]["reason"] == "provider_error"
        assert "memory unavailable" not in json.dumps(result)
    assert result["model_called"] is False
    assert backend.calls == []
    assert len(provider.calls) == 1


def test_assignment_runner_does_not_retrieve_without_explicit_memory_request():
    provider = _MemoryProvider(
        blocks=[ContextBlock(source="memory", title="Unexpected", content="private")]
    )
    runner, tasks, _, _, backend = _runtime(
        with_context=False,
        memory_provider=provider,
    )
    tasks.request_evidence_wake(
        "root",
        "parser",
        reason="expert_request",
        source="region_expert",
    )

    result = _run(runner, "parser")

    assert result["assignment_lifecycle"]["state"] == "waiting_context"
    assert result["context_retrieval"]["status"] == "skipped"
    assert result["context_retrieval"]["reason"] == "memory_request_empty"
    assert provider.calls == []
    assert backend.calls == []


def test_assignment_runner_rejects_memory_request_for_another_target_region():
    provider = _MemoryProvider(
        blocks=[ContextBlock(source="memory", title="Cross-region", content="private")]
    )
    runner, tasks, _, _, backend = _runtime(
        with_context=False,
        memory_provider=provider,
        parser_memory_request={
            "query": "parser evidence",
            "target_region": "review",
        },
    )
    tasks.request_evidence_wake(
        "root",
        "parser",
        reason="expert_request",
        source="region_expert",
    )

    result = _run(runner, "parser")

    assert result["assignment_lifecycle"]["state"] == "waiting_context"
    assert result["context_retrieval"]["status"] == "skipped"
    assert result["context_retrieval"]["reason"] == "target_region_mismatch"
    assert provider.calls == []
    assert backend.calls == []


def test_assignment_runner_preserves_wake_when_private_stage_fails(monkeypatch):
    provider = _MemoryProvider(
        blocks=[
            ContextBlock(
                source="memory",
                title="Parser evidence",
                content="private",
                metadata={"id": "parser-evidence"},
            )
        ]
    )
    runner, tasks, workspace, _, backend = _runtime(
        with_context=False,
        memory_provider=provider,
        parser_memory_request={"query": "parser evidence"},
    )
    tasks.request_evidence_wake(
        "root",
        "parser",
        reason="expert_request",
        source="region_expert",
    )

    def fail_stage(*args, **kwargs):
        raise RuntimeError("private storage path")

    monkeypatch.setattr(workspace, "stage", fail_stage)
    result = _run(runner, "parser")

    assert result["assignment_lifecycle"]["state"] == "waiting_context"
    assert result["assignment_lifecycle"]["pending_provider_reads"] == 1
    assert result["context_retrieval"]["status"] == "failed"
    assert result["context_retrieval"]["reason"] == "private_stage_failed"
    assert "private storage path" not in json.dumps(result)
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
