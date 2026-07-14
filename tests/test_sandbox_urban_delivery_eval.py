"""城区配送正式成对 A/B harness 测试。"""
from __future__ import annotations

import asyncio
import csv
import json

from brainregion.providers.base import ModelResponse
from brainregion.sandbox.envs import generate_urban_delivery_scenario, shortest_path
from brainregion.sandbox.urban_delivery_eval import (
    DeliveryEvalConfig,
    aggregate_delivery_report,
    build_delivery_env,
    render_delivery_summary,
    run_delivery_eval,
    write_delivery_report,
)


def _run(config, arm, **overrides):
    base = {
        "config": config.label,
        "arm": arm,
        "solved": arm == "navigation_region",
        "completion_fraction": 1.0 if arm == "navigation_region" else 0.5,
        "efficiency": 0.9 if arm == "navigation_region" else None,
        "elapsed_time": 40.0 if arm == "navigation_region" else 60.0,
        "main_turns": 8 if arm == "navigation_region" else 20,
        "env_actions": 36,
        "main_env_actions": 4 if arm == "navigation_region" else 36,
        "delegated_actions": 32 if arm == "navigation_region" else 0,
        "delegated_action_share": 32 / 36 if arm == "navigation_region" else 0.0,
        "blocked_actions": 1,
        "navigation_replans": 1 if arm == "navigation_region" else 0,
        "main_input_tokens": 800,
        "input_tokens": 800,
        "output_tokens": 100,
        "cost": 0.02,
    }
    base.update(overrides)
    return base


def _aggregate(configs, runs, **overrides):
    kwargs = {
        "run_id": "delivery-test",
        "model": "mock",
        "configs": configs,
        "repeats": 1,
        "runs": runs,
        "orphan_runs": [],
        "cost_total": sum(run["cost"] for run in runs),
        "cost_capped": False,
        "temperature": 0.0,
        "thinking": False,
        "effort": None,
        "endpoint_id": None,
        "max_tokens": 100,
    }
    kwargs.update(overrides)
    return aggregate_delivery_report(**kwargs)


def test_delivery_config_build_is_deterministic_and_effectful():
    config = DeliveryEvalConfig(size=9, seed=3, orders=2, vehicles=2)
    first = build_delivery_env(config)
    second = build_delivery_env(config)
    assert first.scenario == second.scenario
    assert first.oracle.obstacle_delay > 0


def test_aggregate_uses_config_level_paired_deltas():
    configs = [DeliveryEvalConfig(seed=0), DeliveryEvalConfig(seed=1)]
    runs = []
    for index, config in enumerate(configs):
        runs.append(_run(config, "main_only", solved=bool(index), completion_fraction=0.5 + 0.5 * index))
        runs.append(_run(
            config, "navigation_interface", solved=bool(index), completion_fraction=0.5 + 0.5 * index,
        ))
        runs.append(_run(config, "navigation_region", solved=True, completion_fraction=1.0))
    report = _aggregate(configs, runs)
    pair = report["pairwise"]["main_only_vs_navigation_region"]
    interface = report["pairwise"]["main_only_vs_navigation_interface"]
    execution = report["pairwise"]["navigation_interface_vs_navigation_region"]
    assert report["n_complete_configs"] == 2
    assert pair["solve_rate_delta"]["point"] == 0.5
    assert interface["solve_rate_delta"]["point"] == 0.0
    assert execution["solve_rate_delta"]["point"] == 0.5
    assert pair["main_turns_delta"]["point"] == -12.0
    assert pair["main_env_actions_delta"]["point"] == -32.0
    assert report["per_arm"]["navigation_region"]["mean_delegated_action_share"] > 0.8


def test_incomplete_cost_capped_pair_is_orphaned_not_aggregated(monkeypatch):
    config = DeliveryEvalConfig(seed=0)

    async def fake_episode(backend, model, cfg, arm, **kwargs):
        return _run(cfg, arm.name, cost=0.8)

    monkeypatch.setattr(
        "brainregion.sandbox.urban_delivery_eval._run_delivery_episode",
        fake_episode,
    )
    report = asyncio.run(run_delivery_eval(
        object(), "mock", [config], repeats=1, max_cost_usd=0.7, log_progress=False,
    ))
    assert report["cost_capped"] is True
    assert report["runs"] == []
    assert len(report["orphan_runs"]) == 1
    assert report["incomplete_pairs"] is True
    assert report["per_arm"]["main_only"]["n_runs"] == 0


def test_arm_order_rotates_across_repeats(monkeypatch):
    config = DeliveryEvalConfig(seed=0)

    async def fake_episode(backend, model, cfg, arm, **kwargs):
        return _run(cfg, arm.name, cost=0.0)

    monkeypatch.setattr(
        "brainregion.sandbox.urban_delivery_eval._run_delivery_episode",
        fake_episode,
    )
    report = asyncio.run(run_delivery_eval(
        object(), "mock", [config], repeats=2, max_cost_usd=1.0, log_progress=False,
    ))
    assert [run["arm"] for run in report["runs"]] == [
        "main_only", "navigation_interface", "navigation_region",
        "navigation_interface", "navigation_region", "main_only",
    ]
    assert report["runs"][0]["arm_order"] == [
        "main_only", "navigation_interface", "navigation_region",
    ]
    assert report["runs"][3]["arm_order"] == [
        "navigation_interface", "navigation_region", "main_only",
    ]


def test_report_writes_json_csv_and_markdown(tmp_path):
    configs = [DeliveryEvalConfig(seed=0), DeliveryEvalConfig(seed=1)]
    arms = ("main_only", "navigation_interface", "navigation_region")
    runs = [_run(config, arm) for config in configs for arm in arms]
    report = _aggregate(configs, runs)
    json_path, csv_path = write_delivery_report(report, tmp_path)
    summary = render_delivery_summary(report)
    assert json_path.is_file() and csv_path.is_file()
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6 and {row["arm"] for row in rows} == set(arms)
    assert "navigation_interface - main_only" in summary
    assert "navigation_region - navigation_interface" in summary
    assert "navigation_region - main_only" in summary


def test_delivery_eval_cli_argparse():
    from brainregion.cli import build_parser

    args = build_parser().parse_args([
        "sandbox", "delivery-eval", "--main-brain", "sonnet", "--sizes", "9,13",
        "--seeds", "1,2", "--orders", "3", "--vehicles", "2", "--repeats", "4",
        "--max-env-actions", "160", "--option-actions", "12",
    ])
    assert args.sandbox_command == "delivery-eval"
    assert args.sizes == "9,13" and args.seeds == "1,2"
    assert args.orders == 3 and args.vehicles == 2 and args.repeats == 4
    assert args.max_env_actions == 160 and args.option_actions == 12


class _PairedScriptBackend:
    def __init__(self, main_script, interface_script, navigation_script):
        self.scripts = {
            "main": main_script,
            "interface": interface_script,
            "navigation": navigation_script,
        }
        self.indices = {arm: 0 for arm in self.scripts}
        self.current_arm = "main"
        self.navigation_episodes = 0

    async def complete_messages(self, messages, **kwargs):
        if not any(message.get("role") == "assistant" for message in messages):
            if "导航执行脑区" in messages[0]["content"]:
                self.current_arm = ("interface", "navigation")[self.navigation_episodes]
                self.navigation_episodes += 1
            else:
                self.current_arm = "main"
        index = self.indices[self.current_arm]
        self.indices[self.current_arm] += 1
        script = self.scripts[self.current_arm]
        content = script[min(index, len(script) - 1)]
        return ModelResponse(model="mock", content=content, usage={}, cost_usd=0.0)


def _tool(action):
    return json.dumps({"thought": action, "tool": "act", "args": {"action": action}})


def _path_actions(path):
    by_delta = {(0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}
    return [
        by_delta[(after[0] - before[0], after[1] - before[1])]
        for before, after in zip(path, path[1:])
    ]


def test_real_harness_pair_separates_main_and_region_action_ownership():
    config = DeliveryEvalConfig(size=9, seed=0, orders=1, vehicles=1, max_env_actions=80)
    scenario = generate_urban_delivery_scenario(
        seed=config.seed,
        width=config.size,
        height=config.size,
        order_count=config.orders,
        vehicle_count=config.vehicles,
    )
    order = scenario.orders[0]
    route = shortest_path(scenario.roads, scenario.vehicles, scenario.shop, scenario.unit_positions[order.unit_id])
    assert route is not None
    main_actions = ["pickup", *_path_actions(route), "deliver", *_path_actions(tuple(reversed(route)))]
    done = json.dumps({"thought": "完成", "done": True, "answer": "配送完成"})
    backend = _PairedScriptBackend(
        [_tool(action) for action in main_actions] + [done],
        [_tool(action) for action in main_actions] + [done],
        [_tool("pickup"), _tool("deliver"), done],
    )

    report = asyncio.run(run_delivery_eval(
        backend,
        "mock",
        [config],
        repeats=1,
        max_cost_usd=1.0,
        max_tokens=100,
        log_progress=False,
    ))

    by_arm = {run["arm"]: run for run in report["runs"]}
    assert by_arm["main_only"]["solved"] is True
    assert by_arm["navigation_interface"]["solved"] is True
    assert by_arm["navigation_region"]["solved"] is True
    assert by_arm["main_only"]["delegated_actions"] == 0
    assert by_arm["navigation_interface"]["delegated_actions"] == 0
    assert by_arm["navigation_region"]["delegated_actions"] > 0
    assert by_arm["navigation_interface"]["main_movement_actions"] > 0
    assert by_arm["navigation_region"]["main_movement_actions"] == 0
    assert by_arm["navigation_region"]["region_interaction_actions"] == 0
    assert by_arm["navigation_interface"]["automatic_region_activations"] == 2
    assert by_arm["navigation_region"]["automatic_region_activations"] == 2
    assert by_arm["navigation_region"]["main_turns"] < by_arm["main_only"]["main_turns"]
