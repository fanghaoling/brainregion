"""MCP task delegation, status, and independent report collection tests."""

from __future__ import annotations

import json

import pytest

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


def test_mcp_evidence_wake_is_structured_and_assignment_scoped(monkeypatch):
    from brainregion import server

    monkeypatch.setattr(server, "_task_coordination_board", TaskCoordinationBoard())
    monkeypatch.setattr(server, "_region_coordination_board", RegionCoordinationBoard())
    server.create_task(task_id="root", goal="g")
    server.delegate_task(
        task_id="root",
        assignment_id="parser",
        region="debugging",
        question="q",
    )

    requested = server.request_evidence_wake(
        task_id="root",
        assignment_id="parser",
        reason="explicit_recall",
        ttl_reads=2,
    )
    status = server.task_status("root")

    assert requested["wake"]["source"] == "mcp_request"
    assert requested["wake"]["region"] == "debugging"
    assert requested["wake"]["remaining_reads"] == 2
    assert requested["contains_context_content"] is False
    assert requested["authorization_boundary"] is False
    assert status["evidence_wake_count"] == 1
    assert "question" not in json.dumps(requested)

    cleared = server.workspace_context(
        "root", operation="clear", assignment_id="parser"
    )

    assert cleared["removed_evidence_wakes"] == 1
    assert server.task_status("root")["evidence_wake_count"] == 0


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


def _metric_record(task_id: str, arm: str, solved: bool) -> dict:
    expert = arm != "main_only"
    return {
        "task_id": task_id,
        "repeat": 0,
        "arm": arm,
        "solved": solved,
        "score": float(solved),
        "steps": 2,
        "repeated_attempts": 0,
        "reports_produced": int(expert),
        "reports_adopted": int(expert),
        "expert_failures": 0,
        "main_input_tokens": 100,
        "main_total_tokens": 120,
        "expert_total_tokens": 30 if expert else 0,
        "total_tokens": 150 if expert else 120,
        "main_cost_usd": 0.02,
        "expert_cost_usd": 0.01 if expert else 0.0,
        "total_cost_usd": 0.03 if expert else 0.02,
        "main_error": False,
    }


def test_mcp_delegation_experiment_plan_and_metric_summary(monkeypatch):
    from brainregion import server

    monkeypatch.setattr(server, "_task_coordination_board", TaskCoordinationBoard())
    server.create_task(task_id="root", goal="g")
    server.delegate_task(
        task_id="root",
        assignment_id="debug",
        region="debugging",
        question="q1",
    )
    server.delegate_task(
        task_id="root",
        assignment_id="review",
        region="review",
        question="q2",
    )

    plan = server.plan_delegation_experiment("root", repeats=2)
    records = []
    for task_id in ("a", "b"):
        records.extend(
            [
                _metric_record(task_id, "main_only", False),
                _metric_record(task_id, "single_expert", True),
                _metric_record(task_id, "multi_expert", True),
            ]
        )
    summary = server.summarize_delegation_experiment(
        records,
        run_id="mcp-delegation",
        bootstrap_samples=50,
    )

    assert len(plan["runs"]) == 6
    assert plan["runs"][1]["assignment_ids"] == ["debug"]
    assert plan["runs"][2]["assignment_ids"] == ["debug", "review"]
    assert plan["models_called"] is False
    assert summary["bootstrap_unit"] == "task"
    assert summary["pairwise"]["main_only_vs_single_expert"]["n_tasks"] == 2
    assert "cases" not in summary


def test_mcp_delegation_summary_rejects_context_fields():
    from brainregion import server

    with pytest.raises(ValueError, match="unknown field"):
        server.summarize_delegation_experiment([{**_metric_record("a", "main_only", False), "raw_context": "no"}])
