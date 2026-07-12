"""MCP wiring for context receipts, region status, and main-brain escalation inbox."""

from __future__ import annotations

import json

from brainregion.core.cognitive_workspace import CognitiveWorkspace
from brainregion.core.context import ProviderRegistry
from brainregion.core.region_reporting import RegionCoordinationBoard
from brainregion.memory import ExperienceEvent, MemoryProvider


def test_mcp_region_report_keeps_routine_work_quiet_and_escalates_decision_change(
    monkeypatch,
):
    from brainregion import server

    private_memory = "专家私有：旧解析器修复曾因严格 JSON 假设失败。"
    providers = ProviderRegistry()
    providers.register(
        "memory",
        MemoryProvider.from_records(
            [
                ExperienceEvent(
                    id="report-mcp-memory",
                    region="debugging",
                    summary="解析器历史",
                    details=private_memory,
                    triggers=["reporting-private"],
                )
            ]
        ),
    )
    monkeypatch.setattr(server, "_default_provider_registry", providers)
    monkeypatch.setattr(server, "_skill_registry_singleton", None)
    monkeypatch.setattr(server, "_cognitive_workspace", CognitiveWorkspace())
    monkeypatch.setattr(server, "_region_coordination_board", RegionCoordinationBoard())

    staged = server.stage_region_context(
        task_id="report-mcp-task",
        query="reporting-private",
        audience="region",
        target_region="debugging",
        events=["repeated_attempt_failed"],
        scope_regions=["debugging"],
        max_context_tokens=2000,
    )

    receipt = staged["context_receipt"]
    assert receipt["state"] == "ready"
    assert receipt["selector_coverage"] == "unverified"
    assert "failure_lessons" in receipt["requested_selectors"]
    assert "failing_tests" not in receipt["requested_selectors"]
    assert private_memory not in json.dumps(staged, ensure_ascii=False)

    routine = server.workspace_context(
        "report-mcp-task",
        operation="publish_report",
        report={
            "region": "debugging",
            "state": "working",
            "summary": "已有足够证据运行一次可逆的 fallback 测试。",
            "recommended_action": "运行解析器 fallback 测试。",
            "evidence_refs": receipt["evidence_refs"],
            "context_state": receipt["state"],
            "decision_scope": "routine",
            "risk": "low",
            "memory_impact": "supporting",
            "reversible": True,
        },
    )
    assert routine["decision"]["action"] == "continue"
    assert server.workspace_context("report-mcp-task", operation="inbox")["count"] == 0
    status = server.workspace_context("report-mcp-task", operation="status")
    assert status["region_statuses"][0]["needs_main_attention"] is False
    assert status["contains_private_context"] is False

    important = server.workspace_context(
        "report-mcp-task",
        operation="publish_report",
        report={
            "region": "debugging",
            "state": "needs_decision",
            "summary": "历史约束会改变当前解析器方案。",
            "implication": "继续实现前需要主脑选择兼容策略。",
            "recommended_action": "在宽容解析与严格失败之间做架构取舍。",
            "evidence_refs": receipt["evidence_refs"],
            "context_state": receipt["state"],
            "decision_scope": "architecture",
            "risk": "medium",
            "memory_impact": "decision_changing",
            "reversible": True,
        },
    )
    assert important["decision"]["action"] == "notify_main"
    inbox = server.workspace_context("report-mcp-task", operation="inbox")
    assert inbox["count"] == 1
    assert inbox["reports"][0]["report"]["summary"] == "历史约束会改变当前解析器方案。"
    assert private_memory not in json.dumps(inbox, ensure_ascii=False)
    assert server.workspace_context("report-mcp-task", consumer="main")["context_blocks"] == []

    cleared = server.workspace_context("report-mcp-task", operation="clear")
    assert cleared["removed_entries"] == 1
    assert cleared["removed_receipts"] == 1
    assert cleared["removed_reports"] == 2
