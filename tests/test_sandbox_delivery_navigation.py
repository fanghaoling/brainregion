"""配送 grounded 导航脑区及主脑/脑区控制交接测试。"""
from __future__ import annotations

import asyncio
import json

from brainregion.providers.base import ModelResponse
from brainregion.sandbox import cleanup_run_dir, make_run_dir
from brainregion.sandbox.envs import UrbanDeliveryEnv, build_env_system_prompt, generate_urban_delivery_scenario
from brainregion.sandbox.loop import run_agent, scoped_env
from brainregion.sandbox.option_runtime import OptionRegion
from brainregion.sandbox.regions import DeliveryNavigationInterfaceRegion, DeliveryNavigationRegion
from brainregion.sandbox.regions.delivery_navigation_region import parse_delivery_observation
from brainregion.sandbox.task import SandboxTask


def _status(info):
    return "blocked" if info.get("blocked") else "moved"


def _drive_region(env, region, *, limit=100):
    actions = []
    for _ in range(limit):
        action = region.next_action(env.observation())
        if action is None:
            break
        observation, _, _, info = env.step(action)
        actions.append(action)
        region.observe_transition(action=action, observation=observation, status=_status(info))
        if region.option_boundary(observation, actions_executed=len(actions)):
            break
    return actions


def test_delivery_navigation_region_implements_option_protocol():
    assert isinstance(DeliveryNavigationRegion(), OptionRegion)
    assert isinstance(DeliveryNavigationInterfaceRegion(), OptionRegion)


def test_interface_control_ingests_public_observation_but_never_emits_actions():
    scenario = generate_urban_delivery_scenario(seed=0, width=9, height=9, order_count=1, vehicle_count=1)
    env = UrbanDeliveryEnv(scenario)
    env.step("pickup")
    region = DeliveryNavigationInterfaceRegion()

    assert region.next_action(env.observation()) is None
    state = region.snapshot()
    assert state["policy"] == "matched_interface_no_action"
    assert state["control"] == "interface_only"
    assert state["known_roads"] > 0


def test_parse_delivery_observation_tracks_main_interaction_state():
    scenario = generate_urban_delivery_scenario(seed=0, width=9, height=9, order_count=1, vehicle_count=1)
    env = UrbanDeliveryEnv(scenario)
    initial = parse_delivery_observation(env.observation())
    assert initial.task == "pickup" and initial.target_unit == scenario.orders[0].unit_id

    env.step("pickup")
    carrying = parse_delivery_observation(env.observation())
    assert carrying.task == "deliver" and carrying.target_unit == scenario.orders[0].unit_id

    region = DeliveryNavigationRegion()
    _drive_region(env, region)
    env.step("deliver")
    returning = parse_delivery_observation(env.observation())
    assert returning.task == "return" and returning.target_unit is None


def test_region_completes_movement_but_never_interactions():
    scenario = generate_urban_delivery_scenario(seed=0, width=13, height=13, order_count=3, vehicle_count=2)
    env = UrbanDeliveryEnv(scenario)
    region = DeliveryNavigationRegion()
    region_actions = []
    for order in scenario.orders:
        env.step("pickup")
        region_actions.extend(_drive_region(env, region))
        assert env._agent == scenario.unit_positions[order.unit_id]
        env.step("deliver")
        region_actions.extend(_drive_region(env, region))
        assert env._agent == scenario.shop

    assert env.solved is True
    assert region_actions
    assert set(region_actions) <= {"up", "down", "left", "right"}
    assert region.snapshot()["known_vehicles"] == len(scenario.vehicles)
    assert region.snapshot()["replans"] >= 1


class _InteractionBackend:
    """主脑只做 pickup/deliver/done，移动全部等待自动脑区完成。"""

    def __init__(self):
        self.calls = 0
        self.messages = []

    async def complete_messages(self, messages, **kwargs):
        self.messages.append([dict(message) for message in messages])
        script = [
            {"thought": "取货", "tool": "act", "args": {"action": "pickup"}},
            {"thought": "签收", "tool": "act", "args": {"action": "deliver"}},
            {"thought": "返店完成", "done": True, "answer": "配送完成"},
        ]
        content = json.dumps(script[min(self.calls, len(script) - 1)], ensure_ascii=False)
        self.calls += 1
        return ModelResponse(model=kwargs.get("model", "mock"), content=content, usage={}, cost_usd=0.0)


def test_run_agent_sleeps_region_until_pickup_then_unloads_all_movement():
    scenario = generate_urban_delivery_scenario(seed=0, width=9, height=9, order_count=1, vehicle_count=1)
    env = UrbanDeliveryEnv(scenario)
    region = DeliveryNavigationRegion()
    backend = _InteractionBackend()
    task = SandboxTask(id="delivery-option", goal="完成一单并返店")

    def verify(t, run_dir, *, python_exe=None):
        return {
            "tests_green": env.solved,
            "solve_status": "solved" if env.solved else "tests_fail",
            "pytest": None,
            "gold_diff": t.gold_diff,
        }

    run_dir = make_run_dir()
    try:
        with scoped_env(env):
            trajectory = asyncio.run(run_agent(
                backend,
                "mock",
                task,
                run_dir=run_dir,
                arm="none",
                max_steps=8,
                max_env_actions=60,
                system_prompt=build_env_system_prompt(env, task.goal, navigation=True),
                verify_fn=verify,
                option_region=region,
                option_autorun_actions=16,
                option_continuous=True,
                option_initial_activation=False,
            ))
    finally:
        cleanup_run_dir(run_dir)

    assert env.solved is True and trajectory.tests_green is True
    assert backend.calls == 3
    assert trajectory.interaction_actions == 2
    assert trajectory.delegated_actions == trajectory.successful_moves + trajectory.blocked_actions
    assert trajectory.delegated_actions > 0
    assert all(item["actor"] == "main" for item in trajectory.env_action_trace if item["status"] == "interacted")
    assert all(record["trigger"] != "initial" for record in trajectory.option_activations)
    assert [record["trigger"] for record in trajectory.option_activations] == [
        "after_main_action", "after_main_action",
    ]
    assert "导航执行脑区" in backend.messages[0][0]["content"]


def test_automatic_navigation_activation_respects_runtime_cap():
    scenario = generate_urban_delivery_scenario(seed=0, width=9, height=9, order_count=1, vehicle_count=1)
    env = UrbanDeliveryEnv(scenario)
    region = DeliveryNavigationRegion()
    backend = _InteractionBackend()
    task = SandboxTask(id="delivery-option-cap", goal="完成一单并返店")

    def verify(t, run_dir, *, python_exe=None):
        return {
            "tests_green": env.solved,
            "solve_status": "solved" if env.solved else "tests_fail",
            "pytest": None,
            "gold_diff": t.gold_diff,
        }

    run_dir = make_run_dir()
    try:
        with scoped_env(env):
            trajectory = asyncio.run(run_agent(
                backend,
                "mock",
                task,
                run_dir=run_dir,
                arm="none",
                max_steps=6,
                max_env_actions=60,
                system_prompt=build_env_system_prompt(env, task.goal, navigation=True),
                verify_fn=verify,
                option_region=region,
                option_autorun_actions=16,
                option_continuous=True,
                option_initial_activation=False,
                max_option_activations=1,
            ))
    finally:
        cleanup_run_dir(run_dir)

    assert trajectory.automatic_region_activations == 1
    assert len(trajectory.option_activations) == 1
    assert env.solved is False


def test_navigation_prompt_exposes_delegate_without_oracle_truth():
    scenario = generate_urban_delivery_scenario(seed=4, width=9, height=9, order_count=1, vehicle_count=1)
    env = UrbanDeliveryEnv(scenario)
    prompt = build_env_system_prompt(env, "完成配送", navigation=True)
    assert "delegate_navigation" in prompt and "只读取公开地图" in prompt
    assert "oracle" not in prompt.lower()
    assert str(sorted(scenario.vehicles)) not in prompt
