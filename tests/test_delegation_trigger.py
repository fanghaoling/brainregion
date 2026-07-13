from __future__ import annotations

import pytest

from brainregion.sandbox.delegation_trigger import DelegationTriggerPolicy
from brainregion.sandbox.loop import AdvisoryTriggerState


def _state(**updates) -> AdvisoryTriggerState:
    values = {
        "next_step": 0,
        "completed_steps": 0,
        "remaining_steps": 4,
        "workspace_effects": 0,
        "steps_since_workspace_effect": 0,
        "verification_runs": 0,
        "last_verification_passed": None,
        "recent_tools": (),
        "recent_paths": (),
        "recent_errors": 0,
        "remaining_cost_usd": 0.5,
    }
    return AdvisoryTriggerState(**{**values, **updates})


def test_policy_wakes_after_two_steps_without_workspace_effects():
    decision = DelegationTriggerPolicy().evaluate(
        _state(
            next_step=2,
            completed_steps=2,
            remaining_steps=2,
            steps_since_workspace_effect=2,
            recent_tools=("search_text", "read_text"),
        )
    )

    assert decision.activate is True
    assert decision.reason == "no_workspace_effect"
    assert decision.signals == ("no_workspace_effect",)


def test_policy_uses_repeated_and_failed_verification_signals_without_private_text():
    decision = DelegationTriggerPolicy().evaluate(
        _state(
            completed_steps=2,
            steps_since_workspace_effect=0,
            last_verification_passed=False,
            recent_tools=("read_text", "read_text"),
            recent_paths=("service.py", "service.py"),
        )
    )

    assert decision.activate is True
    assert decision.reason == "verification_failed"
    assert decision.signals == (
        "verification_failed",
        "repeated_tool",
        "repeated_path",
    )


def test_policy_stays_asleep_while_progressing_or_too_late_to_use_advice():
    policy = DelegationTriggerPolicy()

    assert policy.evaluate(_state(completed_steps=1, steps_since_workspace_effect=1)).activate is False
    assert (
        policy.evaluate(
            _state(
                completed_steps=3,
                steps_since_workspace_effect=3,
                remaining_steps=1,
            )
        ).activate
        is False
    )
    assert (
        policy.evaluate(
            _state(
                completed_steps=3,
                steps_since_workspace_effect=3,
                remaining_cost_usd=0.0,
            )
        ).activate
        is False
    )


@pytest.mark.parametrize("field", ["min_steps_without_effect", "min_remaining_steps"])
def test_policy_rejects_nonpositive_thresholds(field):
    with pytest.raises(ValueError, match=field):
        DelegationTriggerPolicy(**{field: 0})
