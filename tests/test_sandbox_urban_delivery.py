"""城区配送环境：生成性质、隐藏 oracle、状态机和现有 agent loop 接入。"""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json

import pytest

from brainregion.providers.base import ModelResponse
from brainregion.sandbox import cleanup_run_dir, make_run_dir
from brainregion.sandbox.envs import (
    UrbanDeliveryEnv,
    build_delivery_oracle,
    build_env_system_prompt,
    generate_urban_delivery_scenario,
    shortest_path,
    validate_urban_delivery_scenario,
)
from brainregion.sandbox.loop import run_agent, scoped_env
from brainregion.sandbox.task import SandboxTask

_ACTION_BY_DELTA = {
    (0, -1): "up",
    (0, 1): "down",
    (-1, 0): "left",
    (1, 0): "right",
}


def _path_actions(path):
    return [
        _ACTION_BY_DELTA[(after[0] - before[0], after[1] - before[1])]
        for before, after in zip(path, path[1:])
    ]


def _optimal_actions(scenario):
    actions = []
    units = scenario.unit_positions
    for order in scenario.orders:
        outbound = shortest_path(scenario.roads, scenario.vehicles, scenario.shop, units[order.unit_id])
        assert outbound is not None
        actions.extend(["pickup", *_path_actions(outbound), "deliver", *_path_actions(tuple(reversed(outbound)))])
    return actions


def test_generator_is_deterministic_for_same_seed():
    first = generate_urban_delivery_scenario(seed=17)
    second = generate_urban_delivery_scenario(seed=17)
    assert first == second
    assert build_delivery_oracle(first) == build_delivery_oracle(second)


def test_generated_scenarios_are_reachable_and_obstacles_are_effectful():
    for seed in range(40):
        scenario = generate_urban_delivery_scenario(seed=seed)
        validation = validate_urban_delivery_scenario(scenario)
        oracle = build_delivery_oracle(scenario)
        assert validation.valid, (seed, validation.reasons)
        assert len(validation.blocked_distances) == len(scenario.orders)
        assert oracle.obstacle_delay > 0
        assert any(route.blocked_distance > route.baseline_distance for route in oracle.routes)


def test_small_single_order_scenarios_support_two_vehicles():
    for seed in range(20):
        scenario = generate_urban_delivery_scenario(
            seed=seed, width=9, height=9, order_count=1, vehicle_count=2,
        )
        assert validate_urban_delivery_scenario(scenario).valid


def test_vehicle_free_scenario_is_valid_with_zero_delay():
    scenario = generate_urban_delivery_scenario(seed=2, vehicle_count=0)
    validation = validate_urban_delivery_scenario(scenario)
    oracle = build_delivery_oracle(scenario)
    assert validation.valid
    assert oracle.obstacle_delay == 0
    assert oracle.optimal_total_time == oracle.baseline_total_time


def test_validator_rejects_vehicle_on_unit():
    scenario = generate_urban_delivery_scenario(seed=4, vehicle_count=0)
    broken = replace(scenario, vehicles=frozenset({scenario.units[0][1]}))
    validation = validate_urban_delivery_scenario(broken)
    assert validation.valid is False
    assert any("车辆必须" in reason for reason in validation.reasons)
    with pytest.raises(ValueError, match="无效城区配送场景"):
        build_delivery_oracle(broken)


def test_initial_observation_hides_vehicle_truth_and_oracle():
    scenario = generate_urban_delivery_scenario(seed=5, vehicle_count=2)
    env = UrbanDeliveryEnv(scenario, visibility_radius=1)
    model_view = env.observation()
    admin_view = env.render_admin()
    assert "V" not in model_view
    assert admin_view.count("V") == len(scenario.vehicles)
    assert "oracle" not in model_view.lower()
    assert "optimal" not in model_view.lower()
    assert str(sorted(scenario.vehicles)) not in model_view


def test_attempting_vehicle_cell_discovers_it_without_moving():
    scenario = generate_urban_delivery_scenario(seed=3, vehicle_count=1)
    env = UrbanDeliveryEnv(scenario, visibility_radius=0)
    vehicle = next(iter(scenario.vehicles))
    neighbor = next(
        cell
        for dx, dy in _ACTION_BY_DELTA
        if (cell := (vehicle[0] - dx, vehicle[1] - dy)) in scenario.roads
        and cell not in scenario.vehicles
        and shortest_path(scenario.roads, scenario.vehicles, scenario.shop, cell) is not None
    )
    path = shortest_path(scenario.roads, scenario.vehicles, scenario.shop, neighbor)
    assert path is not None
    for action in _path_actions(path):
        env.step(action)
    before = env._agent
    action = _ACTION_BY_DELTA[(vehicle[0] - neighbor[0], vehicle[1] - neighbor[1])]
    _, reward, terminated, info = env.step(action)
    assert env._agent == before
    assert reward == 0.0 and terminated is False
    assert info == {"blocked": True, "reason": "vehicle"}
    assert "V" in env.render()
    assert env.blocked_attempts == 1


def test_pickup_deliver_and_final_return_are_required():
    scenario = generate_urban_delivery_scenario(seed=7, width=9, height=9, order_count=1, vehicle_count=1)
    env = UrbanDeliveryEnv(scenario)
    order = scenario.orders[0]
    route = shortest_path(scenario.roads, scenario.vehicles, scenario.shop, scenario.unit_positions[order.unit_id])
    assert route is not None

    _, _, _, pickup_info = env.step("pickup")
    for action in _path_actions(route):
        env.step(action)
    _, reward, terminated, delivery_info = env.step("deliver")
    assert pickup_info == {"interaction": True, "pickup": order.id}
    assert delivery_info == {"interaction": True, "delivered": order.id}
    assert reward == 1.0 and terminated is False and env.solved is False

    reverse_actions = _path_actions(tuple(reversed(route)))
    for action in reverse_actions[:-1]:
        env.step(action)
    _, reward, terminated, return_info = env.step(reverse_actions[-1])
    assert reward == 0.0 and terminated is True
    assert return_info == {"returned": order.id, "goal": True}
    assert env.solved is True


def test_oracle_route_scores_one_at_optimal_execution():
    scenario = generate_urban_delivery_scenario(seed=11)
    env = UrbanDeliveryEnv(scenario)
    for action in _optimal_actions(scenario):
        env.step(action)
    metrics = env.metrics()
    assert env.solved is True
    assert metrics["elapsed_time"] == metrics["oracle"]["optimal_total_time"]
    assert metrics["efficiency"] == 1.0
    assert metrics["delivered_orders"] == len(scenario.orders)
    assert metrics["returned_orders"] == len(scenario.orders)
    assert all(record["efficiency"] == 1.0 for record in metrics["orders"])


def test_invalid_interactions_do_not_advance_order_state():
    scenario = generate_urban_delivery_scenario(seed=1, order_count=1, vehicle_count=0)
    env = UrbanDeliveryEnv(scenario)
    _, _, _, info = env.step("deliver")
    assert info == {"interaction": True, "invalid": "nothing_to_deliver"}
    assert env.current_order == scenario.orders[0]
    assert env.total_reward == 0.0
    assert env.interaction_actions == 1


def test_delivery_prompt_uses_custom_rules_without_truth_leak():
    scenario = generate_urban_delivery_scenario(seed=8)
    env = UrbanDeliveryEnv(scenario)
    prompt = build_env_system_prompt(env, "完成全部订单")
    assert "pickup" in prompt and "deliver" in prompt and "最后一单也必须返回" in prompt
    assert "oracle" not in prompt.lower() and "optimal" not in prompt.lower()
    assert str(sorted(scenario.vehicles)) not in prompt
    navigation_prompt = build_env_system_prompt(env, "完成", navigation=True)
    assert "delegate_navigation" in navigation_prompt
    with pytest.raises(ValueError, match="尚未接入"):
        build_env_system_prompt(env, "完成", memory=True)


class _ScriptedBackend:
    def __init__(self, script):
        self.script = script
        self.index = 0

    async def complete_messages(self, messages, **kwargs):
        content = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        return ModelResponse(model=kwargs.get("model", "mock"), content=content, usage={}, cost_usd=0.0)


def _verify_env(env):
    def verify(task, run_dir, *, python_exe=None):
        return {
            "tests_green": env.solved,
            "solve_status": "solved" if env.solved else "tests_fail",
            "pytest": None,
            "gold_diff": task.gold_diff,
        }
    return verify


def test_existing_run_agent_completes_delivery_round_trip():
    scenario = generate_urban_delivery_scenario(seed=2, width=9, height=9, order_count=1, vehicle_count=1)
    env = UrbanDeliveryEnv(scenario)
    actions = _optimal_actions(scenario)
    script = [
        json.dumps({"thought": action, "tool": "act", "args": {"action": action}})
        for action in actions
    ]
    script.append(json.dumps({"thought": "完成", "done": True, "answer": "配送完成"}))
    task = SandboxTask(id="urban-delivery-smoke", goal="完成一单配送并返店")
    run_dir = make_run_dir()
    try:
        with scoped_env(env):
            trajectory = asyncio.run(run_agent(
                _ScriptedBackend(script),
                "mock",
                task,
                run_dir=run_dir,
                arm="none",
                max_steps=len(script) + 1,
                max_env_actions=len(actions),
                system_prompt=build_env_system_prompt(env, task.goal),
                verify_fn=_verify_env(env),
            ))
    finally:
        cleanup_run_dir(run_dir)

    assert trajectory.tests_green is True and trajectory.solve_status == "solved"
    assert trajectory.env_actions == len(actions)
    assert trajectory.interaction_actions == 2
    assert trajectory.successful_moves == len(actions) - 2
    assert trajectory.blocked_actions == 0
    delivery_steps = [step for step in trajectory.env_action_trace if step["action"] == "deliver"]
    assert delivery_steps[0]["status"] == "interacted" and delivery_steps[0]["reward"] == 1.0


def test_sandbox_env_cli_runs_urban_delivery_without_external_model(monkeypatch, tmp_path):
    from brainregion.cli import build_parser
    from brainregion.sandbox import cli as sandbox_cli

    scenario = generate_urban_delivery_scenario(
        seed=6, width=9, height=9, order_count=1, vehicle_count=1,
    )
    actions = _optimal_actions(scenario)
    script = [
        json.dumps({"thought": action, "tool": "act", "args": {"action": action}})
        for action in actions
    ]
    script.append(json.dumps({"thought": "完成", "done": True, "answer": "配送完成"}))
    backend = _ScriptedBackend(script)
    monkeypatch.setattr(
        sandbox_cli._defaults_mod,
        "apply",
        lambda: {"sandbox_main_brain": "mock", "sandbox_max_steps": 10},
    )
    monkeypatch.setattr(sandbox_cli, "_build_backend", lambda *args, **kwargs: (backend, {}))
    monkeypatch.setattr(sandbox_cli, "_resolve_main_brain", lambda *args, **kwargs: ("mock", None))
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args([
        "sandbox", "env", "--env", "urban-delivery", "--size", "9",
        "--orders", "1", "--vehicles", "1", "--seed", "6",
        "--main-brain", "mock", "--max-steps", str(len(script) + 1),
    ])

    result = asyncio.run(sandbox_cli.run_env(args))

    assert result["solved"] is True and result["tests_green"] is True
    assert result["delivery_metrics"]["efficiency"] == 1.0
    assert result["delivery_metrics"]["oracle"]["obstacle_delay"] > 0
    assert (tmp_path / result["replay"]).is_file()
