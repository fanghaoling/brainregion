"""Main-task decomposition and expert-assignment contract tests."""

from __future__ import annotations

import json

import pytest

from brainregion.core.task_coordination import (
    ExpertAssignment,
    MemoryRequest,
    TaskCoordinationBoard,
    TaskSpec,
)


def test_task_and_assignment_contracts_keep_memory_request_directional():
    task = TaskSpec.from_dict(
        {
            "task_id": "root",
            "goal": "Resolve the parser regression",
            "success_criteria": ["focused test passes"],
            "constraints": ["no unrelated refactor"],
        }
    )
    assignment = ExpertAssignment.from_dict(
        task.task_id,
        {
            "assignment_id": "debug-parser",
            "region": "debugging",
            "question": "Find the next bounded diagnostic step",
            "scope": "Configuration loading only",
            "depends_on": ["reproduce"],
            "memory_request": {
                "query": "parser config regression",
                "purpose": "reuse failure lessons",
                "regions": ["memory", "debugging"],
                "selectors": ["failure_lessons", "evidence_anchors"],
                "top_k": 3,
                "max_context_tokens": 900,
            },
        },
    )

    assert task.status == "queued"
    assert assignment.memory_request.target_region == "debugging"
    assert assignment.memory_request.selectors == (
        "failure_lessons",
        "evidence_anchors",
    )
    assert assignment.to_dict()["expected_output"] == "region_report"


def test_task_board_tracks_multiple_independent_assignments_without_context():
    board = TaskCoordinationBoard()
    board.create_task({"task_id": "root", "goal": "Choose a fix"})
    board.delegate(
        "root",
        {
            "assignment_id": "debug",
            "region": "debugging",
            "question": "Find the failure mechanism",
        },
    )
    board.delegate(
        "root",
        {
            "assignment_id": "architecture",
            "region": "review",
            "question": "Assess the ownership boundary",
        },
    )
    board.set_assignment_status("root", "debug", "working")

    status = board.status("root")

    assert status["assignment_count"] == 2
    assert status["assignments"][0]["status"] == "working"
    assert status["contains_context_content"] is False
    assert "ContextBlock" not in json.dumps(status)


def test_task_board_rejects_unknown_tasks_duplicates_and_bad_status():
    board = TaskCoordinationBoard()
    with pytest.raises(ValueError, match="unknown task"):
        board.delegate(
            "missing",
            {"assignment_id": "a", "region": "debugging", "question": "q"},
        )
    board.create_task({"task_id": "root", "goal": "g"})
    with pytest.raises(ValueError, match="already exists"):
        board.create_task({"task_id": "root", "goal": "g"})
    board.delegate(
        "root",
        {"assignment_id": "a", "region": "debugging", "question": "q"},
    )
    with pytest.raises(ValueError, match="already exists"):
        board.delegate(
            "root",
            {"assignment_id": "a", "region": "review", "question": "q"},
        )
    with pytest.raises(ValueError, match="must be one of"):
        board.set_assignment_status("root", "a", "sleeping")


def test_evidence_wakes_are_assignment_scoped_and_age_only_on_matching_reads():
    board = TaskCoordinationBoard()
    board.create_task({"task_id": "root", "goal": "g"})
    board.delegate(
        "root",
        {"assignment_id": "parser", "region": "debugging", "question": "q1"},
    )
    board.delegate(
        "root",
        {"assignment_id": "architecture", "region": "review", "question": "q2"},
    )
    parser_wake = board.request_evidence_wake(
        "root",
        "parser",
        reason="expert_request",
        source="region_expert",
        ttl_reads=2,
    )
    board.request_evidence_wake(
        "root",
        "architecture",
        reason="task_focus_change",
        source="runtime_policy",
        ttl_reads=1,
    )

    architecture = board.consume_evidence_wakes("root", "architecture")
    status = board.status("root")

    assert architecture["count"] == 1
    assert architecture["deliveries"][0]["region"] == "review"
    assert architecture["deliveries"][0]["remaining_reads"] == 0
    assert status["evidence_wake_count"] == 1
    assert status["evidence_wakes"][0]["request_id"] == parser_wake["request_id"]
    assert status["evidence_wakes"][0]["remaining_reads"] == 2

    first_parser = board.consume_evidence_wakes("root", "parser")
    second_parser = board.consume_evidence_wakes("root", "parser")

    assert first_parser["deliveries"][0]["remaining_reads"] == 1
    assert second_parser["deliveries"][0]["remaining_reads"] == 0
    assert board.status("root")["evidence_wake_count"] == 0


def test_evidence_wake_contract_rejects_unowned_or_unbounded_requests():
    board = TaskCoordinationBoard()
    board.create_task({"task_id": "root", "goal": "g"})
    board.delegate(
        "root",
        {"assignment_id": "a", "region": "debugging", "question": "q"},
    )

    with pytest.raises(ValueError, match="unknown assignment"):
        board.request_evidence_wake(
            "root",
            "missing",
            reason="expert_request",
            source="region_expert",
        )
    with pytest.raises(ValueError, match="reason must be one of"):
        board.request_evidence_wake(
            "root", "a", reason="free form task text", source="region_expert"
        )
    with pytest.raises(ValueError, match="source must be one of"):
        board.request_evidence_wake(
            "root", "a", reason="expert_request", source="unknown_model"
        )
    with pytest.raises(ValueError, match="cannot exceed 32"):
        board.request_evidence_wake(
            "root",
            "a",
            reason="expert_request",
            source="region_expert",
            ttl_reads=33,
        )


def test_memory_request_is_metadata_only_and_fail_fast():
    request = MemoryRequest.from_dict(None, default_region="security")

    assert request.target_region == "security"
    assert request.query == ""
    with pytest.raises(ValueError, match="unknown field"):
        MemoryRequest.from_dict({"raw_context": "must not be stored"}, default_region="debugging")
    with pytest.raises(ValueError, match="positive integer"):
        MemoryRequest.from_dict({"top_k": 0}, default_region="debugging")
    with pytest.raises(ValueError, match="unknown field"):
        TaskSpec.from_dict(
            {"task_id": "root", "goal": "g", "context": "must not be stored"}
        )
    with pytest.raises(ValueError, match="unknown field"):
        ExpertAssignment.from_dict(
            "root",
            {
                "assignment_id": "a",
                "region": "debugging",
                "question": "q",
                "chain_of_thought": "must not be stored",
            },
        )


def test_task_clear_removes_only_selected_task():
    board = TaskCoordinationBoard()
    board.create_task({"task_id": "a", "goal": "A"})
    board.create_task({"task_id": "b", "goal": "B"})

    cleared = board.clear("a")

    assert cleared == {
        "task_id": "a",
        "removed_task": True,
        "removed_assignments": 0,
        "removed_evidence_wakes": 0,
    }
    assert board.status("b")["task"]["goal"] == "B"
