"""MCP task delegation, status, and independent report collection tests."""

from __future__ import annotations

import json

from brainregion.core.cognitive_workspace import CognitiveWorkspace
from brainregion.core.region_reporting import RegionCoordinationBoard
from brainregion.core.task_coordination import TaskCoordinationBoard


def _report(assignment_id: str, summary: str) -> dict:
    return {
        "region": "debugging",
        "assignment_id": assignment_id,
        "state": "done",
        "summary": summary,
        "context_state": "ready",
        "decision_scope": "routine",
        "risk": "low",
        "memory_impact": "supporting",
        "reversible": True,
        "covered_scope": "bounded test scope",
    }


def test_mcp_task_delegation_and_report_collection_are_assignment_scoped(monkeypatch):
    from brainregion import server

    monkeypatch.setattr(server, "_task_coordination_board", TaskCoordinationBoard())
    monkeypatch.setattr(server, "_region_coordination_board", RegionCoordinationBoard())
    monkeypatch.setattr(server, "_cognitive_workspace", CognitiveWorkspace())

    created = server.create_task(
        task_id="root",
        goal="Resolve the regression",
        success_criteria=["test passes"],
        constraints=["bounded changes"],
    )
    parser = server.delegate_task(
        task_id="root",
        assignment_id="parser",
        region="debugging",
        question="Find the parser failure",
        memory_request={
            "query": "parser regression",
            "selectors": ["failure_lessons"],
        },
    )
    server.delegate_task(
        task_id="root",
        assignment_id="architecture",
        region="review",
        question="Check the ownership boundary",
    )
    server.workspace_context(
        "root",
        operation="publish_report",
        assignment_id="parser",
        report=_report("wrong-model-value", "Parser conclusion"),
    )
    server.workspace_context(
        "root",
        operation="publish_report",
        assignment_id="architecture",
        report={**_report("architecture", "Architecture conclusion"), "region": "review"},
    )

    status = server.task_status("root")
    parser_reports = server.collect_reports("root", assignment_id="parser")
    all_reports = server.collect_reports("root")

    assert created["task"]["goal"] == "Resolve the regression"
    assert parser["assignment"]["memory_request"]["target_region"] == "debugging"
    assert status["assignment_count"] == 2
    assert status["assignments"][0]["report_count"] == 1
    assert status["assignments"][0]["latest_report"]["report"]["summary"] == (
        "Parser conclusion"
    )
    assert parser_reports["count"] == 1
    assert parser_reports["reports"][0]["report"]["assignment_id"] == "parser"
    assert all_reports["count"] == 2
    assert all_reports["contains_private_context"] is False
    assert "wrong-model-value" not in json.dumps(parser_reports)


def test_workspace_clear_also_unloads_task_metadata(monkeypatch):
    from brainregion import server

    task_board = TaskCoordinationBoard()
    monkeypatch.setattr(server, "_task_coordination_board", task_board)
    monkeypatch.setattr(server, "_region_coordination_board", RegionCoordinationBoard())
    monkeypatch.setattr(server, "_cognitive_workspace", CognitiveWorkspace())
    server.create_task(task_id="root", goal="g")
    server.delegate_task(
        task_id="root",
        assignment_id="a",
        region="debugging",
        question="q",
    )

    cleared = server.workspace_context("root", operation="clear")

    assert cleared["removed_task"] is True
    assert cleared["removed_assignments"] == 1


def test_task_status_never_contains_workspace_context(monkeypatch):
    from brainregion import server

    monkeypatch.setattr(server, "_task_coordination_board", TaskCoordinationBoard())
    monkeypatch.setattr(server, "_region_coordination_board", RegionCoordinationBoard())
    server.create_task(task_id="root", goal="g")
    server.delegate_task(
        task_id="root",
        assignment_id="a",
        region="memory",
        question="Recall project understanding",
        memory_request={"query": "public metadata only"},
    )

    status = server.task_status("root")

    assert status["contains_context_content"] is False
    assert status["contains_private_context"] is False
    assert "context_blocks" not in json.dumps(status)
