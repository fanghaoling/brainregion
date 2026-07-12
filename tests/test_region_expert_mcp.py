"""MCP wiring, routing metadata, and budget guard for run_region_expert."""

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
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return ModelResponse(
            model=kwargs["model"],
            content=json.dumps(
                {
                    "state": "working",
                    "summary": "A bounded fallback test is supported by private evidence.",
                    "implication": "The expert can continue inside its delegated scope.",
                    "recommended_action": "Run the fallback test.",
                    "uncertainty": "Objective verification is still pending.",
                    "evidence_refs": ["memory:id:expert-mcp"],
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


def _runtime(private: str):
    activated = ActivatedContext(
        activation=ActivationPlan(
            decisions=(),
            woken_regions=("memory", "debugging"),
            context_requests=(),
            trace={"models_called": False},
        ),
        blocks=(
            ContextBlock(
                source="memory",
                title="Private parser context",
                content=private,
                metadata={"id": "expert-mcp", "region": "debugging"},
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
    workspace = CognitiveWorkspace()
    delivery = workspace.stage(
        activated,
        task_id="expert-mcp-task",
        audience="region",
        target_region="debugging",
    )
    board = RegionCoordinationBoard()
    board.record_receipt(
        RegionContextReceipt.from_activated(
            activated,
            task_id="expert-mcp-task",
            region="debugging",
            evidence_refs=tuple(delivery.entry["evidence_refs"]),
        )
    )
    return workspace, board


def _configure(monkeypatch, server, backend, workspace, board):
    monkeypatch.setattr(
        server._defaults_mod,
        "apply",
        lambda **_kwargs: {"endpoints": {}, "timeout": 90, "effort": None},
    )
    monkeypatch.setattr(server, "_resolve_endpoints", lambda _cfg: {})
    monkeypatch.setattr(
        server,
        "_build_region_expert_engine",
        lambda _dd, _registry: RegionExpertEngine(backend=backend),
    )
    monkeypatch.setattr(server, "_cognitive_workspace", workspace)
    monkeypatch.setattr(server, "_region_coordination_board", board)


def test_mcp_run_region_expert_returns_report_without_private_context(monkeypatch):
    from brainregion import server

    private = "Private parser history remains inside the debugging region workspace."
    workspace, board = _runtime(private)
    backend = _Backend()
    _configure(monkeypatch, server, backend, workspace, board)

    result = asyncio.run(
        server.run_region_expert(
            task_id="expert-mcp-task",
            region="debugging",
            task="Choose the next parser debugging action.",
            model="mock-model",
        )
    )

    assert result["ok"] is True
    assert result["published_report"]["decision"]["action"] == "continue"
    assert result["context"]["private_context_returned"] is False
    assert private not in json.dumps(result)
    assert result["routing"] == {
        "requested_model": "mock-model",
        "resolved_model": "mock-model",
        "endpoint_id": None,
    }
    assert result["budget"] == {
        "max_usd": None,
        "estimated_usd": None,
        "exhausted": False,
    }
    assert backend.calls and backend.calls[0]["model"] == "mock-model"
    assert board.inbox("expert-mcp-task")["count"] == 0


def test_mcp_region_expert_budget_guard_skips_model(monkeypatch):
    from brainregion import server

    workspace, board = _runtime("Private context must not be sent when budget is exhausted.")
    backend = _Backend()
    _configure(monkeypatch, server, backend, workspace, board)

    result = asyncio.run(
        server.run_region_expert(
            task_id="expert-mcp-task",
            region="debugging",
            task="Do not call the model over budget.",
            model="unknown-priced-model",
            max_cost_usd=0.0,
        )
    )

    assert result["ok"] is False
    assert result["error"].startswith("budget_exceeded")
    assert result["model_called"] is False
    assert result["budget"]["exhausted"] is True
    assert backend.calls == []


def test_mcp_empty_expert_view_requests_context_without_model(monkeypatch):
    from brainregion import server

    backend = _Backend()
    workspace = CognitiveWorkspace()
    board = RegionCoordinationBoard()
    _configure(monkeypatch, server, backend, workspace, board)

    result = asyncio.run(
        server.run_region_expert(
            task_id="empty-expert-task",
            region="debugging",
            task="Need private context before analysis.",
            model="mock-model",
        )
    )

    assert result["model_called"] is False
    assert result["published_report"]["decision"]["action"] == "request_context"
    assert backend.calls == []


def test_mcp_region_expert_ignores_unselected_endpoint_without_credentials(monkeypatch):
    from brainregion import server

    workspace, board = _runtime("A selected relay has grounded private context.")
    backend = _Backend()
    captured = {}
    monkeypatch.setenv("SELECTED_RELAY_KEY", "selected-secret")
    monkeypatch.delenv("UNUSED_RELAY_KEY", raising=False)
    monkeypatch.setattr(
        server._defaults_mod,
        "apply",
        lambda **_kwargs: {
            "endpoints": {
                "selected": {
                    "provider": "anthropic",
                    "base_url": "https://selected.example",
                    "api_key_env": "SELECTED_RELAY_KEY",
                    "models": ["expert-model"],
                },
                "unused": {
                    "provider": "openai",
                    "base_url": "https://unused.example/v1",
                    "api_key_env": "UNUSED_RELAY_KEY",
                    "models": ["offline-model"],
                },
            },
            "timeout": 90,
            "effort": None,
        },
    )

    def build_engine(_dd, registry):
        captured.update(registry)
        return RegionExpertEngine(backend=backend)

    monkeypatch.setattr(server, "_build_region_expert_engine", build_engine)
    monkeypatch.setattr(server, "_cognitive_workspace", workspace)
    monkeypatch.setattr(server, "_region_coordination_board", board)

    result = asyncio.run(
        server.run_region_expert(
            task_id="expert-mcp-task",
            region="debugging",
            task="Use only the selected relay.",
            model="selected/expert-model",
        )
    )

    assert result["ok"] is True
    assert set(captured) == {"selected"}
    assert result["routing"]["endpoint_id"] == "selected"
