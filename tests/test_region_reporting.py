"""Region context receipts, structured reports, and deterministic escalation tests."""

from __future__ import annotations

import pytest

from brainregion.core.activation import ActivationPlan
from brainregion.core.context import ContextBlock
from brainregion.core.context_loader import ActivatedContext, ContextLoadRecord
from brainregion.core.region_reporting import (
    EscalationPolicy,
    RegionContextReceipt,
    RegionCoordinationBoard,
    RegionReport,
)


def _activated(
    *,
    blocks: tuple[ContextBlock, ...] | None = None,
    loads: tuple[ContextLoadRecord, ...] | None = None,
) -> ActivatedContext:
    default_blocks = (
        ContextBlock(
            source="memory",
            title="Parser history",
            content="Fallback parser was selected after repeated failures.",
            metadata={"id": "exp-parser"},
        ),
    )
    default_loads = (
        ContextLoadRecord(
            skill_id="memory-recall",
            region="memory",
            status="loaded",
            provider="memory",
            selectors=("failure_lessons", "evidence_anchors"),
            estimated_tokens=20,
            blocks_loaded=1,
            provider_meta={"removed_expired": 1},
        ),
    )
    return ActivatedContext(
        activation=ActivationPlan(
            decisions=(),
            woken_regions=("memory", "debugging"),
            context_requests=(),
            trace={"models_called": False},
        ),
        blocks=default_blocks if blocks is None else blocks,
        loads=default_loads if loads is None else loads,
        trace={"models_called": False, "estimated_tokens": 20},
    )


def _report(**overrides):
    data = {
        "region": "debugging",
        "state": "working",
        "summary": "Parser evidence is sufficient for the next reversible test.",
        "implication": "No global decision is needed yet.",
        "recommended_action": "Run the parser fallback test.",
        "evidence_refs": ["memory:id:exp-parser"],
        "context_state": "ready",
        "decision_scope": "routine",
        "risk": "low",
        "memory_impact": "supporting",
        "reversible": True,
    }
    data.update(overrides)
    return RegionReport.from_dict("task-report", data)


def test_context_receipt_reports_observable_coverage_without_claiming_understanding():
    receipt = RegionContextReceipt.from_activated(
        _activated(),
        task_id="task-receipt",
        region="debugging",
        evidence_refs=("memory:id:exp-parser",),
    ).to_dict()

    assert receipt["state"] == "ready"
    assert receipt["requested_selectors"] == ["failure_lessons", "evidence_anchors"]
    assert receipt["confirmed_selectors"] == []
    assert receipt["selector_coverage"] == "unverified"
    assert receipt["warnings"] == [
        "stale_candidates_removed",
        "selector_coverage_unverified",
    ]
    assert receipt["blocks_loaded"] == 1
    assert receipt["evidence_refs"] == ["memory:id:exp-parser"]


def test_context_receipt_distinguishes_partial_failure_conflict_and_insufficient():
    partial = RegionContextReceipt.from_activated(
        _activated(
            loads=(
                *_activated().loads,
                ContextLoadRecord(
                    skill_id="git-recall",
                    region="review",
                    status="failed",
                    reason="git unavailable",
                ),
            )
        ),
        task_id="partial",
        region="debugging",
    )
    conflicted = RegionContextReceipt.from_activated(
        _activated(
            loads=(
                ContextLoadRecord(
                    skill_id="memory-recall",
                    region="memory",
                    status="loaded",
                    blocks_loaded=1,
                    provider_meta={"conflicts": 2},
                ),
            )
        ),
        task_id="conflict",
        region="debugging",
    )
    insufficient = RegionContextReceipt.from_activated(
        _activated(blocks=(), loads=()),
        task_id="empty",
        region="debugging",
    )

    assert partial.state == "partial"
    assert "provider_failure_isolated" in partial.warnings
    assert conflicted.state == "conflicted" and conflicted.conflicts == 2
    assert insufficient.state == "insufficient"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"state": "needs_decision"}, "region_needs_decision"),
        ({"memory_impact": "decision_changing"}, "memory_decision_changing"),
        ({"memory_impact": "contradictory"}, "memory_contradictory"),
        ({"context_state": "conflicted"}, "context_conflicted"),
        ({"decision_scope": "architecture"}, "scope_architecture"),
        ({"risk": "high"}, "risk_high"),
        ({"reversible": False}, "irreversible_action"),
        ({"repeated_failure": True}, "repeated_failure"),
        ({"requires_user_choice": True}, "requires_user_choice"),
    ],
)
def test_escalation_policy_notifies_main_only_for_explicit_hard_signals(overrides, reason):
    decision = EscalationPolicy.evaluate(_report(**overrides))

    assert decision.action == "notify_main"
    assert reason in decision.reasons


def test_escalation_policy_keeps_routine_work_autonomous():
    decision = EscalationPolicy.evaluate(_report())

    assert decision.action == "continue"
    assert decision.reasons == ("within_delegated_scope",)


def test_escalation_policy_requests_context_without_interrupting_main():
    decision = EscalationPolicy.evaluate(_report(context_state="insufficient", needs_more_context=True))

    assert decision.action == "request_context"
    assert set(decision.reasons) == {
        "context_insufficient",
        "expert_requested_more_context",
    }


def test_coordination_board_status_is_visible_but_only_escalations_enter_inbox():
    board = RegionCoordinationBoard()
    receipt = RegionContextReceipt.from_activated(
        _activated(),
        task_id="task-board",
        region="debugging",
    )
    board.record_receipt(receipt)
    routine_data = _report().to_dict()
    routine_data.pop("report_id")
    routine_data.pop("task_id")
    routine = board.publish("task-board", routine_data)
    important_data = _report(
        state="needs_decision",
        memory_impact="decision_changing",
        summary="Historical decision conflicts with the proposed architecture.",
    ).to_dict()
    important_data.pop("report_id")
    important_data.pop("task_id")
    important = board.publish("task-board", important_data)

    assert routine["decision"]["action"] == "continue"
    assert important["decision"]["action"] == "notify_main"
    status = board.status("task-board")
    assert status["contains_private_context"] is False
    assert status["context_receipts"][0]["state"] == "ready"
    assert status["region_statuses"][0]["needs_main_attention"] is True
    inbox = board.inbox("task-board")
    assert inbox["count"] == 1
    assert inbox["reports"][0]["report"]["summary"].startswith("Historical decision")


def test_coordination_board_context_request_stays_out_of_main_inbox():
    board = RegionCoordinationBoard()
    report_data = _report(
        context_state="insufficient",
        needs_more_context=True,
    ).to_dict()
    report_data.pop("report_id")
    report_data.pop("task_id")

    published = board.publish("task-context", report_data)

    assert published["decision"]["action"] == "request_context"
    assert board.inbox("task-context")["count"] == 0
    assert board.status("task-context")["region_statuses"][0]["decision"]["action"] == ("request_context")


def test_report_validation_and_board_capacity_are_fail_fast():
    with pytest.raises(ValueError, match="unknown field"):
        RegionReport.from_dict(
            "task",
            {"region": "debugging", "summary": "x", "secret_thought": "do not store"},
        )
    with pytest.raises(ValueError, match="must be a boolean"):
        RegionReport.from_dict(
            "task",
            {"region": "debugging", "summary": "x", "reversible": "yes"},
        )

    board = RegionCoordinationBoard(max_reports=1)
    report_data = _report().to_dict()
    report_data.pop("report_id")
    report_data.pop("task_id")
    board.publish("task", report_data)
    with pytest.raises(RuntimeError, match="capacity"):
        board.publish("task", report_data)
