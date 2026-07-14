"""Task-scoped CognitiveWorkspace delivery, visibility, budget, and TTL tests."""

from __future__ import annotations

import json

import pytest

from brainregion.core.activation import ActivationPlan
from brainregion.core.cognitive_workspace import CognitiveWorkspace
from brainregion.core.context import ContextBlock, ProviderRegistry
from brainregion.core.context_loader import ActivatedContext, ContextLoadRecord
from brainregion.memory import ExperienceEvent, MemoryProvider


def _activated(
    content: str,
    *,
    title: str = "Project understanding",
    region: str = "debugging",
    evidence_id: str = "memory-1",
) -> ActivatedContext:
    return ActivatedContext(
        activation=ActivationPlan(
            decisions=(),
            woken_regions=("memory", region),
            context_requests=(),
            trace={"models_called": False},
        ),
        blocks=(
            ContextBlock(
                source="memory",
                title=title,
                content=content,
                metadata={"id": evidence_id, "region": region},
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
        trace={"models_called": False},
    )


def test_region_private_delivery_does_not_leak_into_main_view_or_receipt():
    workspace = CognitiveWorkspace()
    secret = "专家需要的解析器历史细节"

    receipt = workspace.stage(
        _activated(secret),
        task_id="task-private",
        audience="region",
        target_region="debugging",
        ttl_steps=3,
    ).to_dict()

    assert receipt["status"] == "staged"
    assert secret not in json.dumps(receipt, ensure_ascii=False)
    assert receipt["entry"]["target_region"] == "debugging"
    assert receipt["entry"]["evidence_refs"] == ["memory:id:memory-1"]
    assert workspace.read("task-private", consumer="main").blocks == ()
    assert workspace.read("task-private", consumer="region", region="security").blocks == ()
    expert_view = workspace.read("task-private", consumer="region", region="debugging")
    assert expert_view.blocks[0].content == secret


def test_shared_and_consumer_specific_visibility_rules():
    workspace = CognitiveWorkspace()
    workspace.stage(_activated("shared"), task_id="task-views", audience="shared")
    workspace.stage(_activated("main-only"), task_id="task-views", audience="main")
    workspace.stage(
        _activated("region-only"),
        task_id="task-views",
        audience="region",
        target_region="debugging",
    )

    main = workspace.read("task-views", consumer="main")
    expert = workspace.read("task-views", consumer="region", region="debugging")

    assert [block.content for block in main.blocks] == ["shared", "main-only"]
    assert [block.content for block in expert.blocks] == ["shared", "region-only"]


def test_workspace_read_reapplies_aggregate_budget():
    workspace = CognitiveWorkspace()
    workspace.stage(_activated("a" * 200), task_id="task-budget", audience="shared")
    workspace.stage(_activated("b" * 200), task_id="task-budget", audience="shared")

    view = workspace.read(
        "task-budget",
        consumer="main",
        max_context_tokens=24,
        max_blocks=1,
    ).to_dict()

    assert len(view["context_blocks"]) == 1
    assert view["trace"]["estimated_tokens"] <= 24
    assert view["trace"]["truncated"] is True


def test_workspace_ttl_requires_explicit_task_advance_and_then_unloads():
    workspace = CognitiveWorkspace()
    workspace.stage(
        _activated("temporary"),
        task_id="task-ttl",
        audience="shared",
        ttl_steps=2,
    )

    first = workspace.advance("task-ttl")
    assert first == {
        "task_id": "task-ttl",
        "advanced_steps": 1,
        "active_entries": 1,
        "expired_entries": 0,
    }
    assert workspace.read("task-ttl", consumer="main").blocks
    second = workspace.advance("task-ttl")
    assert second["expired_entries"] == 1
    assert workspace.read("task-ttl", consumer="main").blocks == ()


def test_workspace_inspection_never_returns_context_content_and_clear_is_scoped():
    workspace = CognitiveWorkspace()
    workspace.stage(_activated("do-not-inspect"), task_id="task-a", audience="shared")
    workspace.stage(_activated("keep"), task_id="task-b", audience="shared")

    inspected = workspace.inspect("task-a")
    assert inspected["contains_context_content"] is False
    assert "do-not-inspect" not in json.dumps(inspected)
    assert workspace.clear("task-a")["removed_entries"] == 1
    assert workspace.read("task-b", consumer="main").blocks[0].content == "keep"


def test_workspace_rejects_ambiguous_region_delivery_and_capacity_overflow():
    workspace = CognitiveWorkspace(max_entries=1)
    with pytest.raises(ValueError, match="target_region is required"):
        workspace.stage(_activated("x"), task_id="task", audience="region")
    with pytest.raises(ValueError, match="only valid"):
        workspace.stage(
            _activated("x"),
            task_id="task",
            audience="main",
            target_region="debugging",
        )

    workspace.stage(_activated("first"), task_id="task", audience="shared")
    with pytest.raises(RuntimeError, match="capacity"):
        workspace.stage(_activated("second"), task_id="task", audience="shared")


def test_region_publish_reuses_workspace_visibility_provenance_and_evidence_refs():
    workspace = CognitiveWorkspace()
    block = ContextBlock(
        source="evidence_region",
        title="Source snapshot",
        content="grounded source",
        metadata={"path": "src/parser.py", "sha": "abc123"},
    )

    delivery = workspace.publish(
        [block],
        task_id="task-publish",
        source_region="evidence",
        source_skill="sandbox-evidence",
        audience="shared",
        ttl_steps=2,
    ).to_dict()

    assert delivery["entry"]["source_regions"] == ["evidence"]
    assert delivery["entry"]["source_skills"] == ["sandbox-evidence"]
    assert delivery["entry"]["evidence_refs"] == [
        "evidence_region:sha:abc123",
        "evidence_region:path:src/parser.py",
    ]
    assert workspace.read("task-publish", consumer="main").blocks[0].content == "grounded source"


def test_region_publish_rejects_non_blocks():
    workspace = CognitiveWorkspace()
    with pytest.raises(ValueError, match="ContextBlock"):
        workspace.publish(  # type: ignore[list-item]
            ["not-a-block"],
            task_id="task-publish",
            source_region="evidence",
        )


def test_same_region_assignments_cannot_read_each_others_private_context():
    workspace = CognitiveWorkspace()
    workspace.stage(
        _activated("parser-only", evidence_id="parser"),
        task_id="root",
        audience="region",
        target_region="debugging",
        assignment_id="parser",
    )
    workspace.stage(
        _activated("network-only", evidence_id="network"),
        task_id="root",
        audience="region",
        target_region="debugging",
        assignment_id="network",
    )

    parser = workspace.read(
        "root", consumer="region", region="debugging", assignment_id="parser"
    )
    network = workspace.read(
        "root", consumer="region", region="debugging", assignment_id="network"
    )
    legacy = workspace.read("root", consumer="region", region="debugging")

    assert [block.content for block in parser.blocks] == ["parser-only"]
    assert [block.content for block in network.blocks] == ["network-only"]
    assert legacy.blocks == ()
    assert parser.to_dict()["assignment_id"] == "parser"
    assert workspace.inspect("root")["entries"][0]["assignment_id"] == "parser"


def test_assignment_clear_is_scoped_and_assignment_is_region_only():
    workspace = CognitiveWorkspace()
    for assignment_id in ("a", "b"):
        workspace.stage(
            _activated(assignment_id),
            task_id="root",
            audience="region",
            target_region="debugging",
            assignment_id=assignment_id,
        )

    assert workspace.clear("root", assignment_id="a")["removed_entries"] == 1
    assert workspace.read(
        "root", consumer="region", region="debugging", assignment_id="b"
    ).blocks
    with pytest.raises(ValueError, match="only valid for region"):
        workspace.stage(
            _activated("x"), task_id="root", audience="shared", assignment_id="x"
        )


def test_mcp_stage_keeps_region_context_out_of_main_view(monkeypatch):
    from brainregion import server

    secret = "MCP 专家私有历史"
    providers = ProviderRegistry()
    providers.register(
        "memory",
        MemoryProvider.from_records(
            [
                ExperienceEvent(
                    id="workspace-mcp",
                    region="debugging",
                    summary="工作台卡片",
                    details=secret,
                    triggers=["workspace-private"],
                )
            ]
        ),
    )
    monkeypatch.setattr(server, "_default_provider_registry", providers)
    monkeypatch.setattr(server, "_skill_registry_singleton", None)
    monkeypatch.setattr(server, "_cognitive_workspace", CognitiveWorkspace())

    staged = server.stage_region_context(
        task_id="mcp-task",
        query="workspace-private",
        audience="region",
        target_region="debugging",
        events=["repeated_attempt_failed"],
        max_context_tokens=2000,
    )

    assert staged["delivery"]["status"] == "staged"
    assert staged["trace"]["context_blocks_returned"] == 0
    assert secret not in json.dumps(staged, ensure_ascii=False)
    assert server.workspace_context("mcp-task", consumer="main")["context_blocks"] == []
    expert = server.workspace_context("mcp-task", consumer="region", region="debugging")
    assert expert["context_blocks"][0]["content"] == secret
    inspected = server.workspace_context("mcp-task", operation="inspect")
    assert inspected["contains_context_content"] is False
    assert server.workspace_context("mcp-task", operation="clear")["removed_entries"] == 1
