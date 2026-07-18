"""Main-brain intent compilation and ownership contract tests."""

from __future__ import annotations

import pytest

from brainregion.core.intent import CapabilityRoute, CognitiveIntent, IntentCompiler


def test_intent_compiler_routes_capability_without_model_reasoning():
    compiled = IntentCompiler().compile(
        {
            "intent_id": "iss-015",
            "objective": "Find grounded evidence for the regression",
            "required_capabilities": ["code_evidence"],
            "success_criteria": ["identify relevant source and test evidence"],
            "constraints": ["read only"],
            "resource_hints": ["brainregion/sandbox/loop.py"],
            "search_queries": ["parse_tool_call_diagnostic"],
            "risk": "low",
            "autonomy": "read_only",
        }
    )

    assignment = compiled.assignment_for("code_evidence")

    assert assignment is not None
    assert assignment.assignment_id == "iss-015:code_evidence"
    assert assignment.region == "evidence"
    assert assignment.allowed_actions == ("read_text", "search_text")
    assert assignment.output_contract == "evidence_packet"
    assert compiled.action_owners() == {
        "read_text": "evidence",
        "search_text": "evidence",
    }
    assert compiled.to_dict()["models_called"] is False
    assert compiled.to_dict()["contains_reasoning"] is False


def test_intent_contract_rejects_hidden_context_and_unroutable_capability():
    with pytest.raises(ValueError, match="unknown field"):
        CognitiveIntent.from_dict(
            {
                "intent_id": "task",
                "objective": "Inspect the project",
                "required_capabilities": ["code_evidence"],
                "chain_of_thought": "must not cross the protocol",
            }
        )

    with pytest.raises(ValueError, match="cannot exceed 500 characters"):
        CognitiveIntent.from_dict(
            {
                "intent_id": "task",
                "objective": "Inspect the project",
                "required_capabilities": ["code_evidence"],
                "search_queries": ["x" * 501],
            }
        )

    with pytest.raises(ValueError, match="unroutable capability"):
        IntentCompiler().compile(
            {
                "intent_id": "task",
                "objective": "Implement a fix",
                "required_capabilities": ["code_implementation"],
            }
        )


def test_read_only_intent_cannot_compile_write_ownership():
    compiler = IntentCompiler(
        routes=(
            CapabilityRoute(
                capability="code_implementation",
                region="implementation",
                allowed_actions=("apply_text_patch",),
                output_contract="change_set",
            ),
        )
    )

    with pytest.raises(ValueError, match="read_only intent cannot own write action"):
        compiler.compile(
            {
                "intent_id": "task",
                "objective": "Implement a fix",
                "required_capabilities": ["code_implementation"],
                "autonomy": "read_only",
            }
        )


def test_compiler_rejects_conflicting_action_owners():
    compiler = IntentCompiler(
        routes=(
            CapabilityRoute(
                capability="evidence_a",
                region="evidence_a",
                allowed_actions=("read_text",),
                output_contract="evidence_packet",
            ),
            CapabilityRoute(
                capability="evidence_b",
                region="evidence_b",
                allowed_actions=("read_text",),
                output_contract="evidence_packet",
            ),
        )
    )

    with pytest.raises(ValueError, match="conflicting owners"):
        compiler.compile(
            {
                "intent_id": "task",
                "objective": "Collect independent evidence",
                "required_capabilities": ["evidence_a", "evidence_b"],
                "autonomy": "read_only",
            }
        )
