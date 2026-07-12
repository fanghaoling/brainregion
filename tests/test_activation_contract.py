"""Structured brain-region activation contracts and hard-gate planning."""
from __future__ import annotations

from pathlib import Path

import pytest

from brainregion.core.activation import (
    ActivationContract,
    ActivationSignal,
    evaluate_activation,
    plan_activation,
)
from brainregion.core.skills import SKILLS_DIR, SkillManifest, SkillRegistry, load_skill, load_skills


def _contract(skill_id="motor", region="motor", **overrides):
    data = {
        "capabilities": ["continuous_control"],
        "wake_when": {
            "task_intents": ["move", "navigate"],
            "events": ["movement_requested", "blocked"],
            "required_capabilities": ["environment.act"],
        },
        "do_not_wake_when": {"task_intents": ["code_review"]},
        "context_selectors": ["current_pose", "recent_path", "action_model"],
        "max_context_tokens": 900,
        "activation_mode": "action",
        "cost_tier": "low",
        "confidence_threshold": 0.3,
        "cooldown_steps": 2,
    }
    data.update(overrides)
    return ActivationContract.from_dict(skill_id=skill_id, region=region, data=data)


def test_motor_contract_wakes_and_requests_only_selected_context():
    contract = _contract()
    signal = ActivationSignal.from_dict({
        "task_intents": ["navigate"],
        "events": ["blocked"],
        "available_capabilities": ["environment.act"],
    })

    decision = evaluate_activation(contract, signal)

    assert decision.action == "wake" and decision.score == 1.0
    assert decision.context_request is not None
    assert decision.context_request.selectors == ("current_pose", "recent_path", "action_model")
    assert decision.context_request.max_tokens == 900
    assert decision.context_request.activation_mode == "action"
    assert decision.context_request.cost_tier == "low"
    assert decision.context_request.cooldown_steps == 2


def test_active_cooldown_blocks_an_otherwise_valid_wake():
    signal = ActivationSignal.from_dict({
        "task_intents": ["navigate"],
        "available_capabilities": ["environment.act"],
        "cooldowns": {"motor": 2},
    })

    decision = evaluate_activation(_contract(), signal)

    assert decision.action == "skip"
    assert decision.reasons == ("cooldown_active",)
    assert decision.cooldown_remaining == 2


def test_hard_gate_blocks_missing_capability_before_relevance():
    decision = evaluate_activation(
        _contract(),
        ActivationSignal.from_dict({"task_intents": ["navigate"], "events": ["blocked"]}),
    )

    assert decision.action == "skip"
    assert decision.reasons == ("missing_requirement",)
    assert decision.missing_requirements == ("capability:environment.act",)


def test_negative_condition_has_priority_over_positive_match():
    decision = evaluate_activation(
        _contract(),
        ActivationSignal.from_dict({
            "task_intents": ["navigate", "code_review"],
            "available_capabilities": ["environment.act"],
        }),
    )

    assert decision.action == "skip"
    assert decision.reasons == ("forbidden_signal",)
    assert decision.matched_signals["task_intents"] == ["code_review"]


def test_partial_match_defers_to_future_cheap_semantic_gate():
    contract = _contract(min_signal_groups=2, confidence_threshold=0.8)
    signal = ActivationSignal.from_dict({
        "task_intents": ["navigate"],
        "available_capabilities": ["environment.act"],
    })

    decision = evaluate_activation(contract, signal)

    assert decision.action == "defer"
    assert decision.score == 0.5
    assert decision.reasons == ("ambiguous_match",)


def test_software_skill_requires_target_app_to_be_running():
    contract = ActivationContract.from_dict(
        skill_id="blender-operation",
        region="software_operation",
        data={
            "wake_when": {
                "task_intents": ["edit_scene"],
                "target_apps": ["blender"],
                "required_tools": ["blender_mcp"],
            },
            "do_not_wake_when": {"app_not_running": True},
            "context_selectors": ["current_scene", "selected_objects"],
            "activation_mode": "action",
        },
    )
    stopped = ActivationSignal.from_dict({
        "task_intents": ["edit_scene"],
        "target_apps": ["blender"],
        "available_tools": ["blender_mcp"],
    })
    running = ActivationSignal.from_dict({
        **stopped.to_dict(),
        "running_apps": ["blender"],
    })

    assert evaluate_activation(contract, stopped).missing_requirements == ("running_app",)
    assert evaluate_activation(contract, running).action == "wake"


def test_activation_plan_enforces_region_and_context_budgets():
    motor = _contract()
    vision = ActivationContract.from_dict(
        skill_id="vision",
        region="vision",
        data={
            "wake_when": {"events": ["frame_available"]},
            "context_selectors": ["latest_frame"],
            "max_context_tokens": 700,
            "cost_tier": "medium",
        },
    )
    signal = ActivationSignal.from_dict({
        "task_intents": ["navigate"],
        "events": ["blocked", "frame_available"],
        "available_capabilities": ["environment.act"],
    })

    plan = plan_activation([motor, vision], signal, max_regions=1, max_context_tokens=500)
    out = plan.to_dict()

    assert out["woken_regions"] == ["motor"]
    assert out["context_requests"][0]["max_tokens"] == 500
    assert out["trace"]["models_called"] is False
    assert out["trace"]["allocated_context_tokens"] == 500
    by_skill = {item["skill_id"]: item for item in out["decisions"]}
    assert by_skill["vision"]["action"] == "skip"
    assert by_skill["vision"]["reasons"] == ["region_budget_exceeded"]


def test_skill_registry_discovers_contracts_without_second_registry():
    registry = SkillRegistry()
    for manifest in load_skills(
        SKILLS_DIR,
        region_exists=lambda _region: True,
        provider_exists=lambda _provider: True,
    ):
        registry.register(manifest)

    contracts = registry.activation_contracts()
    assert {contract.skill_id for contract in contracts} == {
        "debugger", "git-recall", "memory-recall"
    }

    plan = registry.plan_activation(
        ActivationSignal.from_dict({"events": ["test_failed"]}),
        max_regions=2,
    ).to_dict()
    assert plan["woken_regions"] == ["debugging"]
    assert plan["context_requests"][0]["selectors"] == [
        "failing_tests", "recent_errors", "attempted_fixes", "relevant_symbols"
    ]


def test_mcp_plan_region_activation_exposes_auditable_hard_gate():
    from brainregion.server import plan_region_activation

    out = plan_region_activation(events=["test_failed"], max_regions=2, max_context_tokens=600)

    assert out["woken_regions"] == ["debugging"]
    assert out["trace"]["models_called"] is False
    assert out["trace"]["allocated_context_tokens"] == 600
    request = out["context_requests"][0]
    assert request["skill_id"] == "debugger"
    assert request["max_tokens"] == 600


def test_manifest_without_activation_metadata_remains_inert():
    registry = SkillRegistry()
    registry.register(SkillManifest(id="legacy", name="Legacy", region="review", kind="role"))

    assert registry.activation_contracts() == []
    assert registry.plan_activation(ActivationSignal()).to_dict()["decisions"] == []


def test_contract_rejects_unknown_fields_and_impossible_signal_count():
    with pytest.raises(ValueError, match="unknown field"):
        ActivationContract.from_dict(
            skill_id="bad",
            region="debugging",
            data={"wake_when": {"eventz": ["test_failed"]}},
        )
    with pytest.raises(ValueError, match="exceeds configured signal groups"):
        ActivationContract.from_dict(
            skill_id="bad",
            region="debugging",
            data={"wake_when": {"events": ["test_failed"]}, "min_signal_groups": 2},
        )


def test_activation_plan_rejects_duplicate_skill_contracts():
    contract = _contract()
    with pytest.raises(ValueError, match="duplicate activation skill_id"):
        plan_activation([contract, contract], ActivationSignal())


def test_skill_loader_rejects_invalid_activation_contract(tmp_path: Path):
    (tmp_path / "bad.yaml").write_text(
        """id: bad
name: Bad
region: debugging
kind: consultant
metadata:
  activation:
    wake_when:
      events: [test_failed]
    max_context_tokens: 0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_context_tokens"):
        load_skill("bad", tmp_path, region_exists=lambda _region: True)
