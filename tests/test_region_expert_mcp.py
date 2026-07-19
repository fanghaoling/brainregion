"""MCP wiring, routing metadata, and budget guard for run_region_expert."""

from __future__ import annotations

import asyncio
import json

from brainregion.core.activation import ActivationPlan
from brainregion.core.cognitive_workspace import CognitiveWorkspace
from brainregion.core.context import (
    ContextBlock,
    ContextQuery,
    ProviderRegistry,
    RetrieveResult,
)
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


class _MemoryProvider:
    def __init__(self, block: ContextBlock) -> None:
        self.block = block
        self.calls: list[ContextQuery] = []

    def retrieve(self, query: ContextQuery) -> RetrieveResult:
        self.calls.append(query)
        return RetrieveResult(provider="memory", blocks=[self.block])


def _runtime(private: str, *, assignment_id: str = ""):
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
        assignment_id=assignment_id,
    )
    board = RegionCoordinationBoard()
    board.record_receipt(
        RegionContextReceipt.from_activated(
            activated,
            task_id="expert-mcp-task",
            region="debugging",
            assignment_id=assignment_id,
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


def _assignment_board(*, memory_request: dict | None = None) -> TaskCoordinationBoard:
    tasks = TaskCoordinationBoard()
    tasks.create_task(
        {"task_id": "expert-mcp-task", "goal": "Resolve the parser regression"}
    )
    assignment = {
        "assignment_id": "parser",
        "region": "debugging",
        "question": "Choose the next parser debugging action.",
        "scope": "Parser loading only.",
    }
    if memory_request is not None:
        assignment["memory_request"] = memory_request
    tasks.delegate("expert-mcp-task", assignment)
    return tasks


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


def test_mcp_assignment_expert_sleeps_before_context_or_endpoint_resolution(
    monkeypatch,
):
    from brainregion import server

    tasks = _assignment_board()
    workspace = CognitiveWorkspace()
    backend = _Backend()
    monkeypatch.setattr(server, "_task_coordination_board", tasks)
    monkeypatch.setattr(server, "_cognitive_workspace", workspace)

    def should_not_apply(**_kwargs):
        raise AssertionError("sleeping assignment resolved model defaults")

    def should_not_read(*_args, **_kwargs):
        raise AssertionError("sleeping assignment read private context")

    monkeypatch.setattr(server._defaults_mod, "apply", should_not_apply)
    monkeypatch.setattr(workspace, "read", should_not_read)

    result = asyncio.run(
        server.run_assignment_expert(
            task_id="expert-mcp-task",
            assignment_id="parser",
            model="unconfigured-model",
        )
    )

    assert result["ok"] is True
    assert result["model_called"] is False
    assert result["assignment_lifecycle"]["state"] == "sleeping"
    assert result["routing"]["resolution_skipped"] == "assignment_sleeping"
    assert backend.calls == []


def test_mcp_assignment_expert_wakes_exact_private_view_and_returns_report(
    monkeypatch,
):
    from brainregion import server

    private = "Private parser evidence remains inside the parser assignment."
    workspace, board = _runtime(private, assignment_id="parser")
    backend = _Backend()
    tasks = _assignment_board()
    tasks.request_evidence_wake(
        "expert-mcp-task",
        "parser",
        reason="expert_request",
        source="region_expert",
        ttl_reads=2,
    )
    _configure(monkeypatch, server, backend, workspace, board)
    monkeypatch.setattr(server, "_task_coordination_board", tasks)
    reads = {"count": 0}
    original_read = workspace.read

    def counted_read(*args, **kwargs):
        reads["count"] += 1
        return original_read(*args, **kwargs)

    monkeypatch.setattr(workspace, "read", counted_read)

    result = asyncio.run(
        server.run_assignment_expert(
            task_id="expert-mcp-task",
            assignment_id="parser",
            model="mock-model",
        )
    )

    assert result["ok"] is True
    assert result["assignment_lifecycle"]["state"] == "awake"
    assert result["assignment_lifecycle"]["wake_reasons"] == ["expert_request"]
    assert result["assignment_lifecycle"]["pending_wake_requests"] == 1
    assert result["assignment_lifecycle"]["pending_provider_reads"] == 1
    assert result["context_retrieval"]["status"] == "existing"
    assert result["context_retrieval"]["reason"] == "private_view_ready"
    assert result["published_report"]["report"]["assignment_id"] == "parser"
    assert result["published_report"]["report"]["evidence_refs"] == [
        "memory:id:expert-mcp"
    ]
    assert private in backend.calls[0]["user"]
    assert "Choose the next parser debugging action" in backend.calls[0]["user"]
    assert "Parser loading only" in backend.calls[0]["user"]
    assert private not in json.dumps(result)
    assert reads["count"] == 1


def test_mcp_assignment_expert_retrieves_private_memory_before_model_call(monkeypatch):
    from brainregion import server

    private = "Retrieved parser history remains private to the parser assignment."
    provider = _MemoryProvider(
        ContextBlock(
            source="memory",
            title="Parser history",
            content=private,
            metadata={"id": "expert-mcp", "region": "debugging"},
        )
    )
    providers = ProviderRegistry()
    providers.register("memory", provider)
    tasks = _assignment_board(
        memory_request={
            "query": "parser fallback history",
            "selectors": ["failure_lessons"],
            "top_k": 2,
            "max_context_tokens": 700,
        }
    )
    tasks.request_evidence_wake(
        "expert-mcp-task",
        "parser",
        reason="expert_request",
        source="region_expert",
    )
    workspace = CognitiveWorkspace()
    board = RegionCoordinationBoard()
    backend = _Backend()
    _configure(monkeypatch, server, backend, workspace, board)
    monkeypatch.setattr(server, "_task_coordination_board", tasks)
    monkeypatch.setattr(server, "_default_provider_registry", providers)
    monkeypatch.setattr(server, "_ensure_default_providers", lambda: None)

    result = asyncio.run(
        server.run_assignment_expert(
            task_id="expert-mcp-task",
            assignment_id="parser",
            model="mock-model",
        )
    )

    assert result["ok"] is True
    assert result["assignment_lifecycle"]["state"] == "awake"
    assert result["context_retrieval"]["status"] == "loaded"
    assert result["context_retrieval"]["blocks_staged"] == 1
    assert result["context_export"]["action"] == "bypass"
    assert result["published_report"]["report"]["evidence_refs"] == [
        "memory:id:expert-mcp"
    ]
    assert len(provider.calls) == 1
    assert provider.calls[0].text == "parser fallback history"
    assert provider.calls[0].selectors == ("failure_lessons",)
    assert private in backend.calls[0]["user"]
    assert private not in json.dumps(result)
    assert tasks.evidence_wake_status("expert-mcp-task", "parser")["count"] == 0


def test_mcp_assignment_export_guard_evaluates_newly_retrieved_memory(monkeypatch):
    from brainregion import server

    private = "Newly retrieved private memory must not reach an external endpoint."
    provider = _MemoryProvider(
        ContextBlock(
            source="memory",
            title="Private parser history",
            content=private,
            metadata={"id": "expert-mcp", "region": "debugging"},
        )
    )
    providers = ProviderRegistry()
    providers.register("memory", provider)
    tasks = _assignment_board(memory_request={"query": "private parser history"})
    tasks.request_evidence_wake(
        "expert-mcp-task",
        "parser",
        reason="expert_request",
        source="region_expert",
    )
    workspace = CognitiveWorkspace()
    board = RegionCoordinationBoard()
    backend = _Backend()
    monkeypatch.setattr(
        server._defaults_mod,
        "apply",
        lambda **_kwargs: {
            "endpoints": {},
            "timeout": 90,
            "effort": None,
            "context_export_policy": {"mode": "enforce"},
        },
    )
    monkeypatch.setattr(server, "_resolve_endpoints", lambda _cfg: {})
    monkeypatch.setattr(
        server,
        "_build_region_expert_engine",
        lambda _dd, _registry: RegionExpertEngine(backend=backend),
    )
    monkeypatch.setattr(server, "_task_coordination_board", tasks)
    monkeypatch.setattr(server, "_cognitive_workspace", workspace)
    monkeypatch.setattr(server, "_region_coordination_board", board)
    monkeypatch.setattr(server, "_default_provider_registry", providers)
    monkeypatch.setattr(server, "_ensure_default_providers", lambda: None)

    result = asyncio.run(
        server.run_assignment_expert(
            task_id="expert-mcp-task",
            assignment_id="parser",
            model="mock-model",
        )
    )
    pending = tasks.evidence_wake_status("expert-mcp-task", "parser")

    assert result["ok"] is False
    assert result["error"].startswith("context_export_denied")
    assert result["context_retrieval"]["status"] == "loaded"
    assert result["context_export"]["action"] == "deny"
    assert result["context_export"]["highest_sensitivity"] == "private"
    assert result["assignment_lifecycle"]["pending_provider_reads"] == 1
    assert pending["wakes"][0]["remaining_reads"] == 1
    assert len(provider.calls) == 1
    assert backend.calls == []
    assert private not in json.dumps(result)


def test_mcp_assignment_budget_guard_preserves_unread_wake(monkeypatch):
    from brainregion import server

    workspace, board = _runtime(
        "Private context must remain asleep when the budget guard blocks the call.",
        assignment_id="parser",
    )
    backend = _Backend()
    tasks = _assignment_board()
    tasks.request_evidence_wake(
        "expert-mcp-task",
        "parser",
        reason="explicit_recall",
        source="main_brain",
    )
    _configure(monkeypatch, server, backend, workspace, board)
    monkeypatch.setattr(server, "_task_coordination_board", tasks)

    result = asyncio.run(
        server.run_assignment_expert(
            task_id="expert-mcp-task",
            assignment_id="parser",
            model="unknown-priced-model",
            max_cost_usd=0.0,
        )
    )
    pending = tasks.evidence_wake_status("expert-mcp-task", "parser")

    assert result["ok"] is False
    assert result["error"].startswith("budget_exceeded")
    assert result["assignment_lifecycle"]["state"] == "blocked"
    assert result["assignment_lifecycle"]["wake_delivered"] is False
    assert pending["wakes"][0]["remaining_reads"] == 1
    assert tasks.status("expert-mcp-task")["task"]["status"] == "blocked"
    assert tasks.assignment("expert-mcp-task", "parser")["status"] == "blocked"
    assert backend.calls == []


def test_mcp_assignment_export_guard_preserves_unread_wake(monkeypatch):
    from brainregion import server

    workspace, board = _runtime(
        "Private assignment evidence must not leave through an untrusted endpoint.",
        assignment_id="parser",
    )
    backend = _Backend()
    tasks = _assignment_board()
    tasks.request_evidence_wake(
        "expert-mcp-task",
        "parser",
        reason="expert_request",
        source="region_expert",
    )
    monkeypatch.setattr(
        server._defaults_mod,
        "apply",
        lambda **_kwargs: {
            "endpoints": {},
            "timeout": 90,
            "effort": None,
            "context_export_policy": {"mode": "enforce"},
        },
    )
    monkeypatch.setattr(server, "_resolve_endpoints", lambda _cfg: {})
    monkeypatch.setattr(
        server,
        "_build_region_expert_engine",
        lambda _dd, _registry: RegionExpertEngine(backend=backend),
    )
    monkeypatch.setattr(server, "_task_coordination_board", tasks)
    monkeypatch.setattr(server, "_cognitive_workspace", workspace)
    monkeypatch.setattr(server, "_region_coordination_board", board)

    result = asyncio.run(
        server.run_assignment_expert(
            task_id="expert-mcp-task",
            assignment_id="parser",
            model="mock-model",
        )
    )
    pending = tasks.evidence_wake_status("expert-mcp-task", "parser")

    assert result["ok"] is False
    assert result["error"].startswith("context_export_denied")
    assert result["assignment_lifecycle"]["state"] == "blocked"
    assert result["assignment_lifecycle"]["pending_provider_reads"] == 1
    assert pending["wakes"][0]["remaining_reads"] == 1
    assert tasks.status("expert-mcp-task")["task"]["status"] == "blocked"
    assert tasks.assignment("expert-mcp-task", "parser")["status"] == "blocked"
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


def test_mcp_export_off_and_audit_send_identical_model_prompts(monkeypatch):
    from brainregion import server

    workspace, board = _runtime("Private context must be byte-identical in off and audit.")
    backend = _Backend()
    policy = {"value": {"mode": "off"}}
    read_count = {"value": 0}
    original_read = workspace.read

    def counted_read(*args, **kwargs):
        read_count["value"] += 1
        return original_read(*args, **kwargs)

    monkeypatch.setattr(workspace, "read", counted_read)
    monkeypatch.setattr(
        server._defaults_mod,
        "apply",
        lambda **_kwargs: {
            "endpoints": {},
            "timeout": 90,
            "effort": None,
            "context_export_policy": policy["value"],
        },
    )
    monkeypatch.setattr(server, "_resolve_endpoints", lambda _cfg: {})
    monkeypatch.setattr(
        server,
        "_build_region_expert_engine",
        lambda _dd, _registry: RegionExpertEngine(backend=backend),
    )
    monkeypatch.setattr(server, "_cognitive_workspace", workspace)
    monkeypatch.setattr(server, "_region_coordination_board", board)

    off = asyncio.run(
        server.run_region_expert(
            task_id="expert-mcp-task",
            region="debugging",
            task="Keep the prompt unchanged.",
            model="mock-model",
        )
    )
    policy["value"] = {"mode": "audit"}
    audit = asyncio.run(
        server.run_region_expert(
            task_id="expert-mcp-task",
            region="debugging",
            task="Keep the prompt unchanged.",
            model="mock-model",
        )
    )

    assert backend.calls[0]["system"] == backend.calls[1]["system"]
    assert backend.calls[0]["user"] == backend.calls[1]["user"]
    assert off["context_export"]["action"] == "bypass"
    assert audit["context_export"]["action"] == "would_deny"
    assert audit["model_called"] is True
    assert read_count["value"] == 3  # off:engine only; audit:policy + engine


def test_mcp_export_enforce_denies_before_model_call(monkeypatch):
    from brainregion import server

    workspace, board = _runtime("Private memory must not leave through an external model.")
    backend = _Backend()
    monkeypatch.setattr(
        server._defaults_mod,
        "apply",
        lambda **_kwargs: {
            "endpoints": {},
            "timeout": 90,
            "effort": None,
            "context_export_policy": {"mode": "enforce"},
        },
    )
    monkeypatch.setattr(server, "_resolve_endpoints", lambda _cfg: {})
    monkeypatch.setattr(
        server,
        "_build_region_expert_engine",
        lambda _dd, _registry: RegionExpertEngine(backend=backend),
    )
    monkeypatch.setattr(server, "_cognitive_workspace", workspace)
    monkeypatch.setattr(server, "_region_coordination_board", board)

    result = asyncio.run(
        server.run_region_expert(
            task_id="expert-mcp-task",
            region="debugging",
            task="This call must be denied.",
            model="mock-model",
        )
    )

    assert result["ok"] is False
    assert result["model_called"] is False
    assert result["error"].startswith("context_export_denied")
    assert result["context_export"]["action"] == "deny"
    assert result["context_export"]["context_modified"] is False
    assert backend.calls == []
