from __future__ import annotations

import pytest

from brainregion.sandbox.option_runtime import (
    ActivationRecord,
    CognitiveScheduler,
    OptionRegion,
    OptionResult,
    select_region_observation,
)
from brainregion.sandbox.regions import GroundedNavigationRegion, NavigationRegion
from brainregion.sandbox.loop import Trajectory


def test_navigation_regions_implement_option_protocol():
    assert isinstance(NavigationRegion(), OptionRegion)
    assert isinstance(GroundedNavigationRegion(), OptionRegion)


def test_select_region_observation_enforces_access_mode():
    public = "???\n?@?\n???"
    privileged = object()
    assert select_region_observation(
        GroundedNavigationRegion(), public_observation=public, privileged_observation=privileged,
    ) is public
    assert select_region_observation(
        NavigationRegion(), public_observation=public, privileged_observation=privileged,
    ) is privileged

    class Unknown:
        name = "unknown"
        access_mode = "unbounded"

    with pytest.raises(ValueError, match="unsupported option access_mode"):
        select_region_observation(
            Unknown(), public_observation=public, privileged_observation=privileged,
        )


def test_scheduler_initial_activation_guards():
    scheduler = CognitiveScheduler(continuous=True)
    assert scheduler.initial(region_available=True, action_budget=8).trigger == "initial"
    assert not scheduler.initial(region_available=False, action_budget=8).activate
    assert not scheduler.initial(region_available=True, action_budget=0).activate


def test_scheduler_reactivates_once_per_new_main_environment_action():
    scheduler = CognitiveScheduler(continuous=True)
    scheduler.mark_activated(action_clock=2)

    decision = scheduler.after_environment_change(
        action_clock=3, last_actor="main", solved=False,
        region_available=True, remaining_actions=5,
    )
    assert decision.activate and decision.trigger == "after_main_action"
    scheduler.mark_activated(action_clock=3)

    duplicate = scheduler.after_environment_change(
        action_clock=3, last_actor="main", solved=False,
        region_available=True, remaining_actions=5,
    )
    assert not duplicate.activate and duplicate.reason == "no_new_environment_action"


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"last_actor": "navigation_region"}, "last_actor_not_main"),
        ({"solved": True}, "already_solved"),
        ({"region_available": False}, "region_unavailable"),
        ({"remaining_actions": 0}, "no_action_budget"),
    ],
)
def test_scheduler_reactivation_guards(kwargs, reason):
    scheduler = CognitiveScheduler(continuous=True)
    base = {
        "action_clock": 1,
        "last_actor": "main",
        "solved": False,
        "region_available": True,
        "remaining_actions": 5,
    }
    base.update(kwargs)
    decision = scheduler.after_environment_change(**base)
    assert not decision.activate and decision.reason == reason


def test_scheduler_continuous_disabled():
    decision = CognitiveScheduler(continuous=False).after_environment_change(
        action_clock=1, last_actor="main", solved=False,
        region_available=True, remaining_actions=None,
    )
    assert not decision.activate and decision.reason == "continuous_disabled"


def test_activation_record_preserves_option_evidence():
    result = OptionResult(
        region="navigation", actor="navigation_region", access_mode="grounded",
        executed_actions=2, stop_reason="decision_boundary:junction", solved=False,
        trace=[{"action": "right"}, {"action": "down"}],
        region_state={"confidence": 0.5, "last_decision": "junction_boundary"},
    )
    record = ActivationRecord.from_result(result, trigger="after_main_action")
    assert record.to_dict() == {
        "trigger": "after_main_action",
        "region": "navigation",
        "access_mode": "grounded",
        "executed_actions": 2,
        "actions": ["right", "down"],
        "stop_reason": "decision_boundary:junction",
        "solved": False,
        "confidence": 0.5,
        "last_decision": "junction_boundary",
    }


def test_trajectory_generic_option_aliases_preserve_navigation_artifacts():
    traj = Trajectory(task_id="t", arm="none")
    traj.navigation_delegations = 2
    traj.option_activations.append({"trigger": "initial"})
    payload = traj.to_dict()
    assert traj.option_delegations == 2
    assert payload["option_delegations"] == payload["navigation_delegations"] == 2
    assert payload["option_activations"] == payload["navigation_options"] == [{"trigger": "initial"}]
