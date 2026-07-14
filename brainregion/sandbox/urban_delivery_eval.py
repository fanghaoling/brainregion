"""城区配送主脑、匹配界面对照与 grounded 导航执行脑区的成对评测。"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any

from brainregion.eval import stats as eval_stats
from brainregion.runtime import merge_usage, normalize_usage

from .envs import UrbanDeliveryEnv, build_env_system_prompt, generate_urban_delivery_scenario
from .isolation import cleanup_run_dir, make_run_dir
from .loop import run_agent, scoped_env
from .regions import DeliveryNavigationInterfaceRegion, DeliveryNavigationRegion
from .task import SandboxTask

logger = logging.getLogger("brainregion.sandbox.urban_delivery_eval")


@dataclass(frozen=True)
class DeliveryEvalConfig:
    size: int = 9
    seed: int = 0
    orders: int = 2
    vehicles: int = 2
    visibility_radius: int = 1
    max_env_actions: int = 120
    max_main_turns: int | None = None

    @property
    def label(self) -> str:
        return (
            f"{self.size}x{self.size}_seed{self.seed}_"
            f"orders{self.orders}_vehicles{self.vehicles}_vis{self.visibility_radius}"
        )


@dataclass(frozen=True)
class DeliveryEvalArm:
    name: str
    navigation_region: bool = False
    interface_control: bool = False


DELIVERY_EVAL_ARMS: tuple[DeliveryEvalArm, ...] = (
    DeliveryEvalArm("main_only"),
    DeliveryEvalArm("navigation_interface", interface_control=True),
    DeliveryEvalArm("navigation_region", navigation_region=True),
)

DELIVERY_EVAL_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("main_only_vs_navigation_interface", "main_only", "navigation_interface"),
    ("navigation_interface_vs_navigation_region", "navigation_interface", "navigation_region"),
    ("main_only_vs_navigation_region", "main_only", "navigation_region"),
)

_DELIVERY_DELTA_METRICS: dict[str, str] = {
    "solve_rate_delta": "solve_rate",
    "completion_fraction_delta": "mean_completion_fraction",
    "efficiency_delta": "mean_efficiency",
    "elapsed_time_delta": "mean_elapsed_time",
    "main_turns_delta": "mean_main_turns",
    "main_env_actions_delta": "mean_main_env_actions",
    "delegated_action_share_delta": "mean_delegated_action_share",
    "blocked_actions_delta": "mean_blocked_actions",
    "automatic_region_activations_delta": "mean_automatic_region_activations",
    "explicit_navigation_calls_delta": "mean_explicit_navigation_calls",
    "input_tokens_delta": "mean_input_tokens",
    "cost_delta": "mean_cost",
}


def build_delivery_env(config: DeliveryEvalConfig) -> UrbanDeliveryEnv:
    scenario = generate_urban_delivery_scenario(
        seed=config.seed,
        width=config.size,
        height=config.size,
        order_count=config.orders,
        vehicle_count=config.vehicles,
    )
    return UrbanDeliveryEnv(scenario, visibility_radius=config.visibility_radius)


async def _run_delivery_episode(
    backend: Any,
    model: str,
    config: DeliveryEvalConfig,
    arm: DeliveryEvalArm,
    *,
    max_cost_usd: float,
    temperature: float,
    max_tokens: int,
    endpoint_id: str | None,
    thinking: bool | None,
    effort: str | None,
    option_actions: int,
) -> dict[str, Any]:
    env = build_delivery_env(config)
    if arm.navigation_region:
        navigation_region = DeliveryNavigationRegion()
    elif arm.interface_control:
        navigation_region = DeliveryNavigationInterfaceRegion()
    else:
        navigation_region = None
    goal = "按顺序完成全部配送订单，并在最后一单后返回商铺 S"
    task = SandboxTask(id=f"delivery-{config.label}", goal=goal)
    main_turn_cap = (
        int(config.max_main_turns)
        if config.max_main_turns is not None
        else max(config.max_env_actions * 2, config.max_env_actions + 20)
    )

    def verify(t, run_dir, *, python_exe=None):
        return {
            "tests_green": bool(env.solved),
            "solve_status": "solved" if env.solved else "tests_fail",
            "pytest": None,
            "gold_diff": getattr(t, "gold_diff", ""),
        }

    run_dir = make_run_dir()
    try:
        with scoped_env(env):
            trajectory = await run_agent(
                backend,
                model,
                task,
                run_dir=run_dir,
                arm="none",
                max_steps=main_turn_cap,
                max_env_actions=config.max_env_actions,
                max_cost_usd=max_cost_usd,
                temperature=temperature,
                max_tokens=max_tokens,
                endpoint_id=endpoint_id,
                thinking=thinking,
                effort=effort,
                system_prompt=build_env_system_prompt(env, goal, navigation=navigation_region is not None),
                verify_fn=verify,
                option_region=navigation_region,
                option_autorun_actions=(option_actions if navigation_region else 0),
                option_continuous=bool(navigation_region),
                option_initial_activation=False,
                option_reactivation_statuses={"interacted"},
                max_option_activations=max(10, config.orders * 2 + 2),
            )
    finally:
        cleanup_run_dir(run_dir)

    metrics = env.metrics()
    main_usage = normalize_usage(trajectory.total_main_usage)
    region_usage = normalize_usage(trajectory.total_arm_usage)
    total_usage = merge_usage(main_usage, region_usage)
    movement_actions = [
        item for item in trajectory.env_action_trace
        if item.get("action") in {"up", "down", "left", "right"}
    ]
    main_movement = sum(1 for item in movement_actions if item.get("actor") == "main")
    region_movement = sum(1 for item in movement_actions if item.get("actor") == "navigation_region")
    interactions = [item for item in trajectory.env_action_trace if item.get("status") == "interacted"]
    region_state = navigation_region.snapshot() if navigation_region is not None else {}
    efficiency = metrics.get("efficiency")
    return {
        "config": config.label,
        "arm": arm.name,
        "solved": bool(trajectory.tests_green),
        "termination": trajectory.termination_reason,
        "main_turns": trajectory.n_steps,
        "main_turn_cap": main_turn_cap,
        "env_actions": trajectory.env_actions,
        "env_action_budget": config.max_env_actions,
        "main_env_actions": sum(1 for item in trajectory.env_action_trace if item.get("actor") == "main"),
        "delegated_actions": trajectory.delegated_actions,
        "delegated_action_share": (
            trajectory.delegated_actions / trajectory.env_actions if trajectory.env_actions else 0.0
        ),
        "main_movement_actions": main_movement,
        "region_movement_actions": region_movement,
        "interaction_actions": len(interactions),
        "region_interaction_actions": sum(
            1 for item in interactions if item.get("actor") == "navigation_region"
        ),
        "blocked_actions": trajectory.blocked_actions,
        "navigation_delegations": trajectory.navigation_delegations,
        "automatic_region_activations": trajectory.automatic_region_activations,
        "explicit_navigation_calls": sum(
            1 for step in trajectory.steps if step.tool == "delegate_navigation"
        ),
        "navigation_replans": region_state.get("replans", 0),
        "navigation_known_vehicles": region_state.get("known_vehicles", 0),
        "navigation_policy": region_state.get("policy"),
        "delivered_orders": metrics["delivered_orders"],
        "returned_orders": metrics["returned_orders"],
        "completion_fraction": metrics["returned_orders"] / config.orders,
        "elapsed_time": metrics["elapsed_time"],
        "oracle_optimal_time": metrics["oracle"]["optimal_total_time"],
        "obstacle_delay": metrics["oracle"]["obstacle_delay"],
        "efficiency": efficiency,
        "main_usage": main_usage,
        "region_usage": region_usage,
        "total_usage": total_usage,
        "main_input_tokens": main_usage["input_tokens"],
        "region_input_tokens": region_usage["input_tokens"],
        "input_tokens": total_usage["input_tokens"],
        "output_tokens": total_usage["output_tokens"],
        "cost": round(trajectory.total_main_cost_usd + trajectory.total_arm_cost_usd, 6),
        "main_cost": round(trajectory.total_main_cost_usd, 6),
        "region_cost": round(trajectory.total_arm_cost_usd, 6),
    }


def _mean(runs: list[dict[str, Any]], key: str) -> float | None:
    values = [float(run[key]) for run in runs if run.get(key) is not None]
    return sum(values) / len(values) if values else None


def _aggregate_arm(runs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(runs)
    return {
        "n_runs": n,
        "n_solved": sum(1 for run in runs if run["solved"]),
        "solve_rate": sum(1 for run in runs if run["solved"]) / n if n else 0.0,
        "mean_completion_fraction": _mean(runs, "completion_fraction"),
        "mean_efficiency": _mean(runs, "efficiency"),
        "mean_elapsed_time": _mean(runs, "elapsed_time"),
        "mean_main_turns": _mean(runs, "main_turns"),
        "mean_env_actions": _mean(runs, "env_actions"),
        "mean_main_env_actions": _mean(runs, "main_env_actions"),
        "mean_delegated_actions": _mean(runs, "delegated_actions"),
        "mean_delegated_action_share": _mean(runs, "delegated_action_share"),
        "mean_blocked_actions": _mean(runs, "blocked_actions"),
        "mean_automatic_region_activations": _mean(runs, "automatic_region_activations"),
        "mean_explicit_navigation_calls": _mean(runs, "explicit_navigation_calls"),
        "mean_navigation_replans": _mean(runs, "navigation_replans"),
        "mean_main_input_tokens": _mean(runs, "main_input_tokens"),
        "mean_input_tokens": _mean(runs, "input_tokens"),
        "mean_output_tokens": _mean(runs, "output_tokens"),
        "mean_cost": _mean(runs, "cost"),
    }


def _paired_delta(
    rows: list[dict[str, Any]],
    key: str,
    *,
    control_arm: str,
    treatment_arm: str,
) -> float | None:
    deltas = []
    for row in rows:
        control = row[control_arm].get(key)
        treatment = row[treatment_arm].get(key)
        if control is not None and treatment is not None:
            deltas.append(float(treatment) - float(control))
    return sum(deltas) / len(deltas) if deltas else None


def _bootstrap_deltas(
    rows: list[dict[str, Any]],
    run_id: str,
    *,
    comparison: str,
    control_arm: str,
    treatment_arm: str,
) -> dict[str, dict[str, Any]]:
    return {
        metric: eval_stats.bootstrap_statistic(
            rows,
            lambda sampled, aggregate_key=aggregate_key: _paired_delta(
                sampled,
                aggregate_key,
                control_arm=control_arm,
                treatment_arm=treatment_arm,
            ),
            seed=eval_stats.seed_for(run_id, f"delivery|{comparison}|{metric}"),
        )
        for metric, aggregate_key in _DELIVERY_DELTA_METRICS.items()
    }


def aggregate_delivery_report(
    *,
    run_id: str,
    model: str,
    configs: list[DeliveryEvalConfig],
    repeats: int,
    runs: list[dict[str, Any]],
    orphan_runs: list[dict[str, Any]],
    cost_total: float,
    max_cost_usd: float,
    cost_capped: bool,
    temperature: float,
    thinking: bool | None,
    effort: str | None,
    endpoint_id: str | None,
    max_tokens: int,
) -> dict[str, Any]:
    per_config: dict[str, dict[str, dict[str, Any]]] = {}
    for config in configs:
        per_config[config.label] = {
            arm.name: _aggregate_arm([
                run for run in runs if run["config"] == config.label and run["arm"] == arm.name
            ])
            for arm in DELIVERY_EVAL_ARMS
        }
    per_arm = {
        arm.name: _aggregate_arm([run for run in runs if run["arm"] == arm.name])
        for arm in DELIVERY_EVAL_ARMS
    }
    rows = [
        per_config[config.label]
        for config in configs
        if all(per_config[config.label][arm.name]["n_runs"] > 0 for arm in DELIVERY_EVAL_ARMS)
    ]
    pairwise = {
        comparison: _bootstrap_deltas(
            rows,
            run_id,
            comparison=comparison,
            control_arm=control_arm,
            treatment_arm=treatment_arm,
        )
        for comparison, control_arm, treatment_arm in DELIVERY_EVAL_COMPARISONS
    }
    descriptive_deltas = {
        comparison: {
            metric: _paired_delta(
                rows,
                aggregate_key,
                control_arm=control_arm,
                treatment_arm=treatment_arm,
            )
            for metric, aggregate_key in _DELIVERY_DELTA_METRICS.items()
        }
        for comparison, control_arm, treatment_arm in DELIVERY_EVAL_COMPARISONS
    }
    if runs:
        signal_regime = (
            "all_solve" if all(run["solved"] for run in runs)
            else "all_fail" if all(not run["solved"] for run in runs)
            else "ok"
        )
    else:
        signal_regime = "no_runs"
    return {
        "run_id": run_id,
        "model": model,
        "arms": [arm.name for arm in DELIVERY_EVAL_ARMS],
        "configs": [config.label for config in configs],
        "repeats": repeats,
        "temperature": temperature,
        "thinking": thinking,
        "effort": effort,
        "endpoint_id": endpoint_id,
        "max_tokens": max_tokens,
        "treatment_contract": (
            "navigation_region reads only public observation and owns movement actions; "
            "main brain retains pickup, deliver, and done decisions"
        ),
        "interface_control_contract": (
            "navigation_interface receives the same public observation, prompt/tool contract, and "
            "interaction-triggered activations, but its matched control policy never emits or executes an action"
        ),
        "interpretation_limit": (
            "navigation_interface - main_only estimates prompt/tool/no-op activation exposure; "
            "navigation_region - navigation_interface estimates the grounded execution-policy increment. "
            "Explicit delegate calls and post-activation transcripts may still diverge and remain observable mediators."
        ),
        "primary_comparison": "navigation_interface_vs_navigation_region",
        "primary_metric": "solve_rate_delta",
        "secondary_metrics": ["efficiency_delta", "main_turns_delta", "main_env_actions_delta"],
        "signal_regime": signal_regime,
        "n_complete_configs": len(rows),
        "per_arm": per_arm,
        "per_config": [{label: values} for label, values in per_config.items()],
        "pairwise": pairwise,
        "descriptive_deltas": descriptive_deltas,
        "cost_budget_usd": round(max_cost_usd, 6),
        "cost_total": round(cost_total, 6),
        "budget_semantics": "call_boundary_soft_cap",
        "budget_overrun_usd": round(max(0.0, cost_total - max_cost_usd), 6),
        "within_budget": cost_total <= max_cost_usd,
        "cost_capped": cost_capped,
        "incomplete_pairs": bool(orphan_runs),
        "orphan_runs": orphan_runs,
        "runs": runs,
    }


async def run_delivery_eval(
    backend: Any,
    model: str,
    configs: list[DeliveryEvalConfig],
    *,
    repeats: int = 2,
    max_cost_usd: float = 2.0,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    endpoint_id: str | None = None,
    thinking: bool | None = None,
    effort: str | None = None,
    option_actions: int = 16,
    log_progress: bool = True,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not (1 <= option_actions <= 16):
        raise ValueError("option_actions must be in 1..16")
    run_id = f"delivery-eval-{int(time.time() * 1000)}"
    runs: list[dict[str, Any]] = []
    orphan_runs: list[dict[str, Any]] = []
    cost_total = 0.0
    cost_capped = False

    for config_index, config in enumerate(configs):
        if cost_capped:
            break
        for repeat in range(repeats):
            if cost_total >= max_cost_usd:
                cost_capped = True
                break
            offset = (config_index + repeat) % len(DELIVERY_EVAL_ARMS)
            ordered_arms = DELIVERY_EVAL_ARMS[offset:] + DELIVERY_EVAL_ARMS[:offset]
            pair_runs: list[dict[str, Any]] = []
            for arm in ordered_arms:
                if cost_total >= max_cost_usd:
                    cost_capped = True
                    break
                summary = await _run_delivery_episode(
                    backend,
                    model,
                    config,
                    arm,
                    max_cost_usd=max(0.0, max_cost_usd - cost_total),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    endpoint_id=endpoint_id,
                    thinking=thinking,
                    effort=effort,
                    option_actions=option_actions,
                )
                summary["repeat"] = repeat
                summary["arm_order"] = [candidate.name for candidate in ordered_arms]
                pair_runs.append(summary)
                cost_total += summary["cost"]
                if log_progress:
                    logger.info(
                        "[delivery-eval] %s r%d %s solved=%s efficiency=%s main_turns=%d delegated=%d cost=%.4f",
                        config.label,
                        repeat,
                        arm.name,
                        summary["solved"],
                        summary["efficiency"],
                        summary["main_turns"],
                        summary["delegated_actions"],
                        summary["cost"],
                    )
            if len(pair_runs) == len(DELIVERY_EVAL_ARMS):
                runs.extend(pair_runs)
            else:
                orphan_runs.extend(pair_runs)
                break

    cost_capped = cost_capped or cost_total >= max_cost_usd
    return aggregate_delivery_report(
        run_id=run_id,
        model=model,
        configs=configs,
        repeats=repeats,
        runs=runs,
        orphan_runs=orphan_runs,
        cost_total=cost_total,
        max_cost_usd=max_cost_usd,
        cost_capped=cost_capped,
        temperature=temperature,
        thinking=thinking,
        effort=effort,
        endpoint_id=endpoint_id,
        max_tokens=max_tokens,
    )


def write_delivery_report(report: dict[str, Any], out_dir: str | Path | None = None) -> tuple[Path, Path]:
    out = Path(out_dir) if out_dir else Path(".brain-region") / "sandbox"
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{report['run_id']}.json"
    csv_path = out / f"{report['run_id']}.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = [
        "config", "repeat", "arm", "solved", "termination", "main_turns", "env_actions",
        "main_env_actions", "delegated_actions", "delegated_action_share", "blocked_actions",
        "automatic_region_activations", "explicit_navigation_calls", "navigation_replans",
        "delivered_orders", "returned_orders", "completion_fraction",
        "elapsed_time", "oracle_optimal_time", "efficiency", "input_tokens", "output_tokens", "cost",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for run in report["runs"]:
            writer.writerow([run.get(column) for column in columns])
    return json_path, csv_path


def render_delivery_summary(report: dict[str, Any]) -> str:
    lines = [
        f"### delivery-eval {report['run_id']}",
        "",
        "| arm | solve | efficiency | main turns | main actions | delegated share | auto wakes | cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in report["arms"]:
        summary = report["per_arm"][arm]
        lines.append(
            f"| {arm} | {summary['solve_rate']:.3f} | {_fmt(summary['mean_efficiency'])} | "
            f"{_fmt(summary['mean_main_turns'])} | {_fmt(summary['mean_main_env_actions'])} | "
            f"{_fmt(summary['mean_delegated_action_share'])} | "
            f"{_fmt(summary['mean_automatic_region_activations'])} | {_fmt(summary['mean_cost'])} |"
        )
    comparison_labels = {
        "main_only_vs_navigation_interface": "navigation_interface - main_only",
        "navigation_interface_vs_navigation_region": "navigation_region - navigation_interface",
        "main_only_vs_navigation_region": "navigation_region - main_only",
    }
    for comparison, label in comparison_labels.items():
        pair = report["pairwise"][comparison]
        raw = report["descriptive_deltas"][comparison]
        lines.extend(["", f"**{label}:**"])
        for metric in (
            "solve_rate_delta", "efficiency_delta", "main_turns_delta",
            "main_env_actions_delta", "input_tokens_delta", "cost_delta",
        ):
            value = pair[metric]
            lines.append(
                f"- {metric}: raw={_fmt(raw.get(metric))}, bootstrap={_fmt(value.get('point'))}, "
                f"CI=[{_fmt(value.get('low'))}, {_fmt(value.get('high'))}]"
            )
    lines.append(
        f"\n完整 config={report['n_complete_configs']}, signal={report['signal_regime']}, "
        f"cost=${report['cost_total']:.4f}/${report['cost_budget_usd']:.4f}, "
        f"overrun=${report['budget_overrun_usd']:.4f}, incomplete_pairs={report['incomplete_pairs']}"
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "null" if value is None else f"{float(value):.3f}"


__all__ = [
    "DELIVERY_EVAL_ARMS",
    "DELIVERY_EVAL_COMPARISONS",
    "DeliveryEvalArm",
    "DeliveryEvalConfig",
    "aggregate_delivery_report",
    "build_delivery_env",
    "render_delivery_summary",
    "run_delivery_eval",
    "write_delivery_report",
]
