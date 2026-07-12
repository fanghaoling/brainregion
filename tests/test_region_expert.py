"""Private workspace -> expert model -> grounded RegionReport tests."""

from __future__ import annotations

import asyncio
import json

from brainregion.core.activation import ActivationPlan
from brainregion.core.cognitive_workspace import CognitiveWorkspace
from brainregion.core.context import ContextBlock
from brainregion.core.context_loader import ActivatedContext, ContextLoadRecord
from brainregion.core.region_expert import RegionExpertEngine
from brainregion.core.region_reporting import RegionContextReceipt, RegionCoordinationBoard
from brainregion.providers.base import ModelResponse


class _Backend:
    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _activated(content: str, *, provider_meta: dict | None = None) -> ActivatedContext:
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
                title="Parser history",
                content=content,
                metadata={"id": "exp-parser", "region": "debugging"},
            ),
        ),
        loads=(
            ContextLoadRecord(
                skill_id="memory-recall",
                region="memory",
                status="loaded",
                provider="memory",
                selectors=("failure_lessons", "evidence_anchors"),
                estimated_tokens=40,
                blocks_loaded=1,
                provider_meta=provider_meta or {},
            ),
        ),
        trace={"models_called": False, "estimated_tokens": 40},
    )


def _runtime(content: str, *, provider_meta: dict | None = None):
    activated = _activated(content, provider_meta=provider_meta)
    workspace = CognitiveWorkspace()
    delivery = workspace.stage(
        activated,
        task_id="expert-task",
        audience="region",
        target_region="debugging",
    )
    board = RegionCoordinationBoard()
    board.record_receipt(
        RegionContextReceipt.from_activated(
            activated,
            task_id="expert-task",
            region="debugging",
            evidence_refs=tuple(delivery.entry["evidence_refs"]),
        )
    )
    return workspace, board


def _response(**overrides) -> ModelResponse:
    report = {
        "state": "working",
        "summary": "A reversible parser fallback test is supported by prior evidence.",
        "implication": "No global decision is required yet.",
        "recommended_action": "Run the bounded fallback test.",
        "uncertainty": "The result still needs an objective test.",
        "evidence_refs": ["memory:id:exp-parser"],
        "decision_scope": "routine",
        "risk": "low",
        "memory_impact": "supporting",
        "reversible": True,
        "repeated_failure": False,
        "requires_user_choice": False,
        "needs_more_context": False,
    }
    report.update(overrides)
    return ModelResponse(
        model="expert-model",
        content=json.dumps(report),
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        cost_usd=0.012,
        cost_source="provider",
    )


def _run(backend, workspace, board):
    return asyncio.run(
        RegionExpertEngine(backend=backend).run(
            workspace=workspace,
            coordination=board,
            task_id="expert-task",
            region="debugging",
            task="Determine the next parser debugging action.",
            model="expert-model",
            endpoint_id="relay",
        )
    )


def test_region_expert_consumes_private_context_but_returns_only_published_report():
    private = "The old strict parser failed repeatedly when a gateway wrapped JSON in prose."
    workspace, board = _runtime(private)
    backend = _Backend(_response())

    result = _run(backend, workspace, board).to_dict()

    assert result["ok"] is True
    assert result["published_report"]["decision"]["action"] == "continue"
    assert result["context"]["private_context_returned"] is False
    assert private not in json.dumps(result)
    assert result["usage"]["total_tokens"] == 150
    assert result["cost_usd"] == 0.012 and result["cost_source"] == "provider"
    assert backend.calls[0]["endpoint_id"] == "relay"
    assert "<<<CONTEXT_DATA_BEGIN" in backend.calls[0]["user"]
    assert private in backend.calls[0]["user"]


def test_decision_changing_expert_report_enters_main_inbox():
    workspace, board = _runtime("Historical constraints require compatibility with prose-wrapped JSON responses.")
    backend = _Backend(
        _response(
            state="needs_decision",
            summary="Historical compatibility constraints change the parser architecture.",
            implication="The main brain must select a compatibility policy.",
            decision_scope="architecture",
            risk="medium",
            memory_impact="decision_changing",
        )
    )

    result = _run(backend, workspace, board).to_dict()

    assert result["published_report"]["decision"]["action"] == "notify_main"
    assert board.inbox("expert-task")["count"] == 1


def test_runtime_context_state_overrides_model_claim_and_forces_conflict_escalation():
    workspace, board = _runtime(
        "Two historical records disagree about strict versus tolerant parsing.",
        provider_meta={"conflicts": 2},
    )
    backend = _Backend(_response(context_state="ready"))

    result = _run(backend, workspace, board).to_dict()

    report = result["published_report"]["report"]
    assert report["context_state"] == "conflicted"
    assert result["published_report"]["decision"]["action"] == "notify_main"


def test_expert_rejects_invented_evidence_reference_without_publishing():
    workspace, board = _runtime("A grounded parser record exists in the private workspace.")
    backend = _Backend(_response(evidence_refs=["memory:id:invented"]))

    result = _run(backend, workspace, board).to_dict()

    assert result["ok"] is False
    assert result["error"].startswith("grounding_error")
    assert board.status("expert-task")["region_statuses"] == []


def test_expert_rejects_verbatim_private_context_leak():
    private = "This exact private memory sentence is deliberately longer than thirty two characters for leak detection"
    workspace, board = _runtime(private)
    backend = _Backend(_response(summary=private))

    result = _run(backend, workspace, board).to_dict()

    assert result["ok"] is False
    assert result["error"].startswith("privacy_error")
    assert private not in json.dumps(result)


def test_expert_isolates_model_and_parse_failures():
    workspace, board = _runtime("Private evidence for model failure isolation.")
    model_error = _Backend(ModelResponse(model="m", error="endpoint offline"))
    parse_error = _Backend(ModelResponse(model="m", content="not json"))

    failed = _run(model_error, workspace, board).to_dict()
    invalid = _run(parse_error, workspace, board).to_dict()

    assert failed["ok"] is False and failed["error"].startswith("model_error")
    assert invalid["ok"] is False and invalid["error"].startswith("parse_error")
    assert board.inbox("expert-task")["count"] == 0


def test_empty_expert_view_requests_context_without_calling_model():
    workspace = CognitiveWorkspace()
    board = RegionCoordinationBoard()
    backend = _Backend(_response())

    result = _run(backend, workspace, board).to_dict()

    assert result["ok"] is True
    assert result["model_called"] is False
    assert result["published_report"]["decision"]["action"] == "request_context"
    assert backend.calls == []
    assert board.inbox("expert-task")["count"] == 0
