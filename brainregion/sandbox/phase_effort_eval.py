"""Matched evaluation for fixed-off versus phase-active inference controls."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brainregion.eval.stats import bootstrap_statistic, seed_for
from brainregion.runtime import normalize_usage

from .cognitive_eval import classify_thinking_control
from .isolation import cleanup_run_dir, make_run_dir, materialize_fixture
from .loop import run_agent
from .phase_control import assess_task_difficulty
from .task import SandboxTask

ARM_FIXED_OFF = "fixed_off"
ARM_PHASE_ACTIVE = "phase_active"


@dataclass(frozen=True)
class PhaseEffortEvalArm:
    name: str
    active: bool


PHASE_EFFORT_EVAL_ARMS: tuple[PhaseEffortEvalArm, ...] = (
    PhaseEffortEvalArm(ARM_FIXED_OFF, active=False),
    PhaseEffortEvalArm(ARM_PHASE_ACTIVE, active=True),
)
_ARM_BY_NAME = {arm.name: arm for arm in PHASE_EFFORT_EVAL_ARMS}
_METRICS = (
    "solved",
    "protocol_completed",
    "steps",
    "main_input_tokens",
    "main_output_tokens",
    "main_total_tokens",
    "reasoning_tokens",
    "cost_usd",
    "step_budget_fraction",
    "cost_budget_fraction",
    "wall_time_s",
    "workspace_effects",
    "main_check_calls",
    "repeated_target_rate",
    "recovery_entries",
    "resolved_recovery_rate",
    "mean_recovery_steps",
    "strong_calls",
    "effective_thinking_calls",
)


def _rotated_arms(offset: int) -> tuple[PhaseEffortEvalArm, ...]:
    pivot = offset % len(PHASE_EFFORT_EVAL_ARMS)
    return (*PHASE_EFFORT_EVAL_ARMS[pivot:], *PHASE_EFFORT_EVAL_ARMS[:pivot])


def _recovery_metrics(transitions: list[dict[str, Any]]) -> dict[str, Any]:
    spans: list[int] = []
    entries = 0
    for index, transition in enumerate(transitions):
        if transition.get("to") != "recover":
            continue
        entries += 1
        entered_at = int(transition.get("step") or 0)
        for candidate in transitions[index + 1 :]:
            if candidate.get("from") == "recover" and candidate.get("to") != "recover":
                spans.append(max(0, int(candidate.get("step") or 0) - entered_at))
                break
    return {
        "recovery_entries": entries,
        "resolved_recoveries": len(spans),
        "resolved_recovery_rate": len(spans) / entries if entries else None,
        "mean_recovery_steps": sum(spans) / len(spans) if spans else None,
    }


def _failure_metrics(transitions: list[dict[str, Any]]) -> dict[str, int]:
    reasons = [str(transition.get("reason") or "") for transition in transitions]
    return {
        "model_error_events": reasons.count("model_error"),
        "parse_error_events": reasons.count("parse_error"),
    }


def classify_empirical_difficulty(
    *,
    solve_rate: float | None,
    protocol_completion_rate: float | None,
    mean_workspace_effects: float | None,
    mean_main_check_calls: float | None,
    mean_step_budget_fraction: float | None = None,
    mean_cost_budget_fraction: float | None = None,
) -> str:
    """Classify task difficulty from the fixed-off arm only."""

    if solve_rate is None or protocol_completion_rate is None:
        return "insufficient"
    if solve_rate >= 0.8:
        budget_pressure = max(
            float(mean_step_budget_fraction or 0.0),
            float(mean_cost_budget_fraction or 0.0),
        )
        return (
            "easy"
            if protocol_completion_rate >= 0.8 and budget_pressure < 0.8
            else "costly_success"
        )
    if solve_rate >= 0.2:
        return "sweet_spot"
    if (mean_workspace_effects or 0.0) > 0 or (mean_main_check_calls or 0.0) > 0:
        return "hard"
    return "blocked"


async def run_phase_effort_eval(
    backend: Any,
    model: str,
    tasks: list[SandboxTask],
    *,
    endpoint_id: str | None = None,
    repeats: int = 1,
    max_steps: int = 10,
    max_cost_usd: float = 0.5,
    max_total_cost_usd: float | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    transcript_token_cap: int = 24000,
    consecutive_error_limit: int = 3,
    tool_result_lifecycle: str = "full",
    tool_result_live_reads: int = 3,
    run_id: str = "",
    bootstrap_samples: int | None = None,
) -> dict[str, Any]:
    """Run matched fixed-off/phase-active pairs in fresh fixture sandboxes."""

    if not tasks:
        raise ValueError("phase effort eval tasks cannot be empty")
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("phase effort eval task ids must be unique")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    if not math.isfinite(float(max_cost_usd)) or max_cost_usd <= 0:
        raise ValueError("max_cost_usd must be positive and finite")
    if max_total_cost_usd is not None and (
        not math.isfinite(float(max_total_cost_usd)) or max_total_cost_usd <= 0
    ):
        raise ValueError("max_total_cost_usd must be positive and finite")
    if tool_result_lifecycle not in {"full", "compact"}:
        raise ValueError(f"unknown tool result lifecycle mode: {tool_result_lifecycle!r}")
    if (
        isinstance(tool_result_live_reads, bool)
        or not isinstance(tool_result_live_reads, int)
        or tool_result_live_reads < 0
    ):
        raise ValueError("tool_result_live_reads must be a non-negative integer")

    run_id = run_id or f"phase-effort-{int(time.time() * 1000)}"
    cases: list[dict[str, Any]] = []
    actual_model_calls = 0
    actual_cost_usd = 0.0
    arm_orders: dict[str, int] = {}
    cost_capped = False
    cost_capped_at: str | None = None
    planned_pairs = len(tasks) * repeats
    completed_pairs = 0

    for task_index, task in enumerate(tasks):
        if cost_capped:
            break
        structural_difficulty = assess_task_difficulty(task).to_dict()
        for repeat in range(repeats):
            if max_total_cost_usd is not None:
                remaining = max_total_cost_usd - actual_cost_usd
                if remaining <= 0:
                    cost_capped = True
                    cost_capped_at = f"{task.id} repeat{repeat} before_pair"
                    break
                pair_run_cap = min(max_cost_usd, remaining / len(PHASE_EFFORT_EVAL_ARMS))
            else:
                pair_run_cap = max_cost_usd

            ordered_arms = _rotated_arms(task_index + repeat)
            order_key = "->".join(arm.name for arm in ordered_arms)
            arm_orders[order_key] = arm_orders.get(order_key, 0) + 1
            for arm in ordered_arms:
                run_dir = make_run_dir(prefix="brainregion-phase-effort-eval-")
                materialize_fixture(task, Path(run_dir))
                started = time.perf_counter()
                try:
                    trajectory = await run_agent(
                        backend,
                        model,
                        task,
                        run_dir=run_dir,
                        arm="none",
                        max_steps=max_steps,
                        max_cost_usd=pair_run_cap,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        transcript_token_cap=transcript_token_cap,
                        consecutive_error_limit=consecutive_error_limit,
                        endpoint_id=endpoint_id,
                        thinking=False,
                        effort=None,
                        effort_routing_shadow=not arm.active,
                        effort_routing_active=arm.active,
                        tool_result_lifecycle=tool_result_lifecycle,
                        tool_result_live_reads=tool_result_live_reads,
                    )
                    wall_time_s = time.perf_counter() - started
                    actual_model_calls += trajectory.n_steps
                    actual_cost_usd += float(trajectory.total_main_cost_usd)
                    usage = normalize_usage(trajectory.total_main_usage)
                    target_steps = [
                        step for step in trajectory.progress_trace if step["target_fingerprint"]
                    ]
                    repeated_targets = sum(not step["target_is_new"] for step in target_steps)
                    operation_counts: dict[str, int] = {}
                    for progress in trajectory.progress_trace:
                        operation = str(progress.get("operation") or "model_turn")
                        operation_counts[operation] = operation_counts.get(operation, 0) + 1
                    routing = (
                        trajectory.effort_routing_shadow.snapshot()
                        if trajectory.effort_routing_shadow
                        else {}
                    )
                    phase_control = (
                        trajectory.phase_controller.snapshot()
                        if trajectory.phase_controller
                        else {"transitions": []}
                    )
                    transitions = list(phase_control.get("transitions") or [])
                    recovery = _recovery_metrics(transitions)
                    failures = _failure_metrics(transitions)
                    decisions = list(routing.get("decisions") or [])
                    cases.append(
                        {
                            "task_id": task.id,
                            "repeat": repeat,
                            "arm": arm.name,
                            "routing_mode": "active" if arm.active else "shadow",
                            "structural_difficulty": structural_difficulty,
                            "solved": trajectory.tests_green,
                            "protocol_completed": trajectory.done,
                            "termination_reason": trajectory.termination_reason,
                            "infrastructure_error": (
                                trajectory.termination_reason == "model_error"
                                or failures["model_error_events"] > 0
                            ),
                            "infrastructure_degraded": failures["model_error_events"] > 0,
                            "steps": trajectory.n_steps,
                            "workspace_effects": trajectory.workspace_effects,
                            "main_check_calls": operation_counts.get("workspace_run_check", 0),
                            "automatic_verification_runs": trajectory.verification_runs,
                            "repeated_target_rate": (
                                repeated_targets / len(target_steps) if target_steps else None
                            ),
                            "main_input_tokens": usage["input_tokens"],
                            "main_output_tokens": usage["output_tokens"],
                            "main_total_tokens": usage["total_tokens"],
                            "reasoning_tokens": usage["reasoning_tokens"],
                            "cost_usd": float(trajectory.total_main_cost_usd),
                            "step_budget_fraction": (
                                trajectory.n_steps / max_steps if max_steps > 0 else None
                            ),
                            "cost_budget_fraction": (
                                float(trajectory.total_main_cost_usd) / pair_run_cap
                                if pair_run_cap > 0
                                else None
                            ),
                            "wall_time_s": wall_time_s,
                            **recovery,
                            **failures,
                            "recommended_thinking_calls": int(
                                routing.get("recommended_thinking_calls") or 0
                            ),
                            "effective_thinking_calls": int(
                                routing.get("actual_thinking_calls") or 0
                            ),
                            "applied_change_calls": int(
                                routing.get("applied_change_calls") or 0
                            ),
                            "strong_calls": sum(
                                decision.get("recommended_tier") == "strong"
                                for decision in decisions
                            ),
                            "phase_calls": {
                                phase: int((values or {}).get("calls") or 0)
                                for phase, values in (routing.get("by_phase") or {}).items()
                            },
                            "operation_counts": operation_counts,
                            "progress_trace": trajectory.progress_trace,
                            "effort_routing": routing,
                            "phase_control": phase_control,
                            "contains_trajectories": False,
                            "contains_context_content": False,
                            "contains_reasoning": False,
                            "contains_tool_results": False,
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - preserve the matched pair
                    cases.append(
                        _runner_error_case(
                            task.id,
                            repeat,
                            arm,
                            structural_difficulty=structural_difficulty,
                            wall_time_s=time.perf_counter() - started,
                            error=exc,
                        )
                    )
                finally:
                    cleanup_run_dir(run_dir)

            completed_pairs += 1
            if max_total_cost_usd is not None and actual_cost_usd >= max_total_cost_usd:
                cost_capped = completed_pairs < planned_pairs
                if cost_capped:
                    cost_capped_at = f"{task.id} repeat{repeat} after_pair"
                break

    report = summarize_phase_effort_records(
        cases,
        run_id=run_id,
        bootstrap_samples=bootstrap_samples,
    )
    report["cases"] = cases
    report["execution"] = {
        "model": model,
        "endpoint_id": endpoint_id,
        "configured_thinking": False,
        "configured_effort": None,
        "thinking_control": classify_thinking_control(model),
        "tool_result_lifecycle": tool_result_lifecycle,
        "tool_result_live_reads": tool_result_live_reads,
        "max_steps": max_steps,
        "per_run_cost_cap_usd": max_cost_usd,
        "max_total_cost_usd": max_total_cost_usd,
        "cost_cap_policy": "equal_per_arm_budget_within_matched_pair",
        "cost_capped": cost_capped,
        "cost_capped_at": cost_capped_at,
        "planned_pairs": planned_pairs,
        "completed_pairs": completed_pairs,
        "arm_order_counts": arm_orders,
        "actual_model_calls": actual_model_calls,
        "actual_cost_usd": actual_cost_usd,
        "contains_trajectories": False,
        "contains_context_content": False,
        "contains_reasoning": False,
        "contains_tool_results": False,
    }
    status_reasons: list[str] = []
    if int((report.get("effect") or {}).get("n_tasks") or 0) < 2:
        status_reasons.append("insufficient_independent_task_units")
    if report.get("thinking_telemetry_status") != "telemetry_confirmed":
        status_reasons.append("provider_thinking_not_confirmed")
    if cost_capped:
        status_reasons.append("planned_matrix_cost_capped")
    if int(report.get("n_matched_pairs") or 0) < completed_pairs:
        status_reasons.append("incomplete_or_failed_matched_pairs")
    report["experiment_status"] = "INCONCLUSIVE" if status_reasons else "COMPLETE"
    report["status_reasons"] = status_reasons
    return report


def summarize_phase_effort_records(
    records: list[dict[str, Any]],
    *,
    run_id: str = "",
    bootstrap_samples: int | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("phase effort records cannot be empty")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        arm = str(record.get("arm") or "")
        if arm not in _ARM_BY_NAME:
            raise ValueError(f"unknown phase effort eval arm: {arm!r}")
        grouped.setdefault(arm, []).append(record)

    per_arm: dict[str, Any] = {}
    for arm in _ARM_BY_NAME:
        arm_records = grouped.get(arm, [])
        valid = [record for record in arm_records if not record.get("infrastructure_error")]
        per_arm[arm] = {
            "n_runs": len(arm_records),
            "n_valid_runs": len(valid),
            "solve_rate": _mean(valid, "solved"),
            "protocol_completion_rate": _mean(valid, "protocol_completed"),
            "mean_steps": _mean(valid, "steps"),
            "mean_main_input_tokens": _mean(valid, "main_input_tokens"),
            "mean_main_output_tokens": _mean(valid, "main_output_tokens"),
            "mean_main_total_tokens": _mean(valid, "main_total_tokens"),
            "mean_reasoning_tokens": _mean(valid, "reasoning_tokens"),
            "mean_cost_usd": _mean(valid, "cost_usd"),
            "mean_step_budget_fraction": _mean(valid, "step_budget_fraction"),
            "mean_cost_budget_fraction": _mean(valid, "cost_budget_fraction"),
            "mean_wall_time_s": _mean(valid, "wall_time_s"),
            "mean_workspace_effects": _mean(valid, "workspace_effects"),
            "mean_main_check_calls": _mean(valid, "main_check_calls"),
            "mean_repeated_target_rate": _mean(valid, "repeated_target_rate"),
            "mean_recovery_entries": _mean(valid, "recovery_entries"),
            "mean_resolved_recovery_rate": _mean(valid, "resolved_recovery_rate"),
            "mean_recovery_steps": _mean(valid, "mean_recovery_steps"),
            "mean_strong_calls": _mean(valid, "strong_calls"),
            "mean_effective_thinking_calls": _mean(valid, "effective_thinking_calls"),
            "infrastructure_failures": len(arm_records) - len(valid),
        }

    task_rows = _paired_task_rows(records)
    raw_deltas = {
        metric: _paired_delta(task_rows, metric)
        for metric in _METRICS
    }
    run_id = run_id or f"phase-effort-summary-{int(time.time() * 1000)}"
    bootstrap_deltas = {
        metric: bootstrap_statistic(
            task_rows,
            lambda sample, metric=metric: _paired_delta(sample, metric),
            B=bootstrap_samples,
            seed=seed_for(run_id, f"phase-effort:{metric}"),
        )
        for metric in _METRICS
    }
    matched_pairs = _matched_pairs(records)
    task_difficulty = _summarize_task_difficulty(records)
    difficulty_mix: dict[str, int] = {}
    for row in task_difficulty:
        band = str(row["empirical_band"])
        difficulty_mix[band] = difficulty_mix.get(band, 0) + 1
    active_records = [
        pair[ARM_PHASE_ACTIVE]
        for pair in matched_pairs
    ]
    control_records = [
        pair[ARM_FIXED_OFF]
        for pair in matched_pairs
    ]
    active_reasoning = any(int(record.get("reasoning_tokens") or 0) > 0 for record in active_records)
    control_reasoning = any(int(record.get("reasoning_tokens") or 0) > 0 for record in control_records)
    return {
        "run_id": run_id,
        "arms": [ARM_FIXED_OFF, ARM_PHASE_ACTIVE],
        "n_tasks": len({record["task_id"] for record in records}),
        "n_runs": len(records),
        "n_matched_pairs": len(matched_pairs),
        "bootstrap_unit": "task",
        "delta_direction": "phase_active_minus_fixed_off",
        "per_arm": per_arm,
        "effect": {
            "control": ARM_FIXED_OFF,
            "treatment": ARM_PHASE_ACTIVE,
            "n_tasks": len(task_rows),
            "n_matched_repeats": sum(row["matched_repeats"] for row in task_rows),
            "raw_deltas": raw_deltas,
            "bootstrap_deltas": bootstrap_deltas,
        },
        "pair_outcomes": {
            "solve": _pair_outcomes(matched_pairs, "solved"),
            "protocol_completed": _pair_outcomes(matched_pairs, "protocol_completed"),
        },
        "task_difficulty": task_difficulty,
        "difficulty_mix": difficulty_mix,
        "recommended_task_ids": [
            row["task_id"]
            for row in task_difficulty
            if row["recommended_for_next_eval"]
        ],
        "active_thinking_requested": any(
            int(record.get("effective_thinking_calls") or 0) > 0
            for record in active_records
        ),
        "active_thinking_observed": active_reasoning,
        "control_reasoning_observed": control_reasoning,
        "thinking_telemetry_status": (
            "contaminated_control"
            if control_reasoning
            else "telemetry_confirmed"
            if active_reasoning
            else "request_only"
        ),
        "contains_trajectories": False,
        "contains_context_content": False,
        "contains_reasoning": False,
        "contains_tool_results": False,
    }


def render_phase_effort_eval_summary(report: dict[str, Any]) -> str:
    execution = report.get("execution") or {}
    lines = [
        f"### phase effort A/B {report['run_id']} "
        f"(tasks={report['n_tasks']}, pairs={report['n_matched_pairs']})",
        f"model={execution.get('model', '')} telemetry={report.get('thinking_telemetry_status')} "
        f"cost=${float(execution.get('actual_cost_usd') or 0):.4f} "
        f"cost_capped={execution.get('cost_capped')} status={report.get('experiment_status')}",
    ]
    for arm, summary in (report.get("per_arm") or {}).items():
        lines.append(
            f"  {arm}: solve={summary.get('solve_rate')} "
            f"completed={summary.get('protocol_completion_rate')} "
            f"steps={summary.get('mean_steps')} tokens={summary.get('mean_main_total_tokens')} "
            f"checks={summary.get('mean_main_check_calls')} "
            f"thinking_calls={summary.get('mean_effective_thinking_calls')} "
            f"recover_steps={summary.get('mean_recovery_steps')} "
            f"cost=${float(summary.get('mean_cost_usd') or 0):.4f}"
        )
    deltas = (report.get("effect") or {}).get("raw_deltas") or {}
    lines.append(
        "  delta(active-off): "
        f"solve={deltas.get('solved')} completed={deltas.get('protocol_completed')} "
        f"steps={deltas.get('steps')} tokens={deltas.get('main_total_tokens')} "
        f"cost={deltas.get('cost_usd')}"
    )
    if report.get("status_reasons"):
        lines.append(f"  inconclusive_reasons={','.join(report['status_reasons'])}")
    for row in report.get("task_difficulty") or []:
        lines.append(
            f"  difficulty/{row['task_id']}: empirical={row['empirical_band']} "
            f"structural={row['structural_band']} evidence={row['evidence_status']}"
        )
    return "\n".join(lines)


def _matched_pairs(records: list[dict[str, Any]]) -> list[dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for record in records:
        key = (str(record["task_id"]), int(record["repeat"]))
        grouped.setdefault(key, {})[str(record["arm"])] = record
    required = set(_ARM_BY_NAME)
    return [
        arms
        for arms in grouped.values()
        if required.issubset(arms)
        and not any(arms[name].get("infrastructure_error") for name in required)
    ]


def _paired_task_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, dict[str, Any]]]] = {}
    for pair in _matched_pairs(records):
        task_id = str(pair[ARM_FIXED_OFF]["task_id"])
        grouped.setdefault(task_id, []).append(pair)
    rows: list[dict[str, Any]] = []
    for task_id, pairs in grouped.items():
        row: dict[str, Any] = {"task_id": task_id, "matched_repeats": len(pairs)}
        for arm in _ARM_BY_NAME:
            row[arm] = {
                metric: _mean([pair[arm] for pair in pairs], metric)
                for metric in _METRICS
            }
        rows.append(row)
    return rows


def _summarize_task_difficulty(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("arm") != ARM_FIXED_OFF or record.get("infrastructure_error"):
            continue
        grouped.setdefault(str(record["task_id"]), []).append(record)

    rows: list[dict[str, Any]] = []
    for task_id, control_records in grouped.items():
        solve_rate = _mean(control_records, "solved")
        protocol_rate = _mean(control_records, "protocol_completed")
        workspace_effects = _mean(control_records, "workspace_effects")
        check_calls = _mean(control_records, "main_check_calls")
        step_budget_fraction = _mean(control_records, "step_budget_fraction")
        cost_budget_fraction = _mean(control_records, "cost_budget_fraction")
        structural = dict(control_records[0].get("structural_difficulty") or {})
        structural_score = structural.get("score")
        if structural_score is None:
            structural_band = "unknown"
        elif float(structural_score) < 0.25:
            structural_band = "low"
        elif float(structural_score) < 0.50:
            structural_band = "medium"
        else:
            structural_band = "high"
        empirical_band = classify_empirical_difficulty(
            solve_rate=solve_rate,
            protocol_completion_rate=protocol_rate,
            mean_workspace_effects=workspace_effects,
            mean_main_check_calls=check_calls,
            mean_step_budget_fraction=step_budget_fraction,
            mean_cost_budget_fraction=cost_budget_fraction,
        )
        observations = len(control_records)
        rows.append(
            {
                "task_id": task_id,
                "control_observations": observations,
                "evidence_status": "calibrated" if observations >= 2 else "pilot_only",
                "structural_band": structural_band,
                "structural_difficulty": structural,
                "empirical_band": empirical_band,
                "control_solve_rate": solve_rate,
                "control_protocol_completion_rate": protocol_rate,
                "control_mean_steps": _mean(control_records, "steps"),
                "control_mean_step_budget_fraction": step_budget_fraction,
                "control_mean_cost_budget_fraction": cost_budget_fraction,
                "control_mean_workspace_effects": workspace_effects,
                "control_mean_main_check_calls": check_calls,
                "recommended_for_next_eval": (
                    observations >= 2
                    and empirical_band in {"sweet_spot", "costly_success"}
                ),
                "difficulty_depends_on_treatment": False,
            }
        )
    return rows


def _paired_delta(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [
        row[ARM_PHASE_ACTIVE][metric] - row[ARM_FIXED_OFF][metric]
        for row in rows
        if row[ARM_PHASE_ACTIVE][metric] is not None
        and row[ARM_FIXED_OFF][metric] is not None
    ]
    return sum(values) / len(values) if values else None


def _pair_outcomes(
    pairs: list[dict[str, dict[str, Any]]], metric: str
) -> dict[str, int]:
    treatment_wins = 0
    control_wins = 0
    ties = 0
    for pair in pairs:
        treatment = float(pair[ARM_PHASE_ACTIVE].get(metric) or 0)
        control = float(pair[ARM_FIXED_OFF].get(metric) or 0)
        if treatment > control:
            treatment_wins += 1
        elif control > treatment:
            control_wins += 1
        else:
            ties += 1
    return {
        "treatment_wins": treatment_wins,
        "control_wins": control_wins,
        "ties": ties,
    }


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def _runner_error_case(
    task_id: str,
    repeat: int,
    arm: PhaseEffortEvalArm,
    *,
    structural_difficulty: dict[str, Any],
    wall_time_s: float,
    error: Exception,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "repeat": repeat,
        "arm": arm.name,
        "routing_mode": "active" if arm.active else "shadow",
        "structural_difficulty": structural_difficulty,
        "solved": False,
        "protocol_completed": False,
        "termination_reason": "runner_error",
        "infrastructure_error": True,
        "infrastructure_degraded": True,
        "steps": 0,
        "workspace_effects": 0,
        "main_check_calls": 0,
        "automatic_verification_runs": 0,
        "repeated_target_rate": None,
        "main_input_tokens": 0,
        "main_output_tokens": 0,
        "main_total_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
        "step_budget_fraction": None,
        "cost_budget_fraction": None,
        "wall_time_s": wall_time_s,
        "recovery_entries": 0,
        "resolved_recoveries": 0,
        "resolved_recovery_rate": None,
        "mean_recovery_steps": None,
        "model_error_events": 0,
        "parse_error_events": 0,
        "recommended_thinking_calls": 0,
        "effective_thinking_calls": 0,
        "applied_change_calls": 0,
        "strong_calls": 0,
        "phase_calls": {},
        "operation_counts": {},
        "progress_trace": [],
        "effort_routing": {},
        "phase_control": {},
        "error": f"runner_error: {type(error).__name__}: {error}"[:500],
        "contains_trajectories": False,
        "contains_context_content": False,
        "contains_reasoning": False,
        "contains_tool_results": False,
    }


__all__ = [
    "ARM_FIXED_OFF",
    "ARM_PHASE_ACTIVE",
    "PHASE_EFFORT_EVAL_ARMS",
    "PhaseEffortEvalArm",
    "classify_empirical_difficulty",
    "render_phase_effort_eval_summary",
    "run_phase_effort_eval",
    "summarize_phase_effort_records",
]
