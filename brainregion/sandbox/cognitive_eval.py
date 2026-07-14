"""Matched 2x2 evaluation for provider-native thinking and external cognitive state."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brainregion.eval.stats import bootstrap_statistic, seed_for
from brainregion.runtime import normalize_usage

from .isolation import cleanup_run_dir, make_run_dir, materialize_fixture
from .input_attribution import merge_input_attributions
from .loop import run_agent
from .task import SandboxTask

ARM_PLAIN = "plain"
ARM_NATIVE_THINKING = "native_thinking"
ARM_EXTERNAL_SCAFFOLD = "external_scaffold"
ARM_COMBINED = "combined"


@dataclass(frozen=True)
class CognitiveEvalArm:
    name: str
    native_thinking: bool
    external_scaffold: bool


COGNITIVE_EVAL_ARMS: tuple[CognitiveEvalArm, ...] = (
    CognitiveEvalArm(ARM_PLAIN, native_thinking=False, external_scaffold=False),
    CognitiveEvalArm(ARM_NATIVE_THINKING, native_thinking=True, external_scaffold=False),
    CognitiveEvalArm(ARM_EXTERNAL_SCAFFOLD, native_thinking=False, external_scaffold=True),
    CognitiveEvalArm(ARM_COMBINED, native_thinking=True, external_scaffold=True),
)
_ARM_BY_NAME = {arm.name: arm for arm in COGNITIVE_EVAL_ARMS}
_SCAFFOLD_MODES = frozenset({"runtime_checkpoint", "model_managed"})
_METRICS = (
    "solved",
    "protocol_completed",
    "steps",
    "main_input_tokens",
    "main_total_tokens",
    "reasoning_tokens",
    "cost_usd",
    "repeated_target_rate",
)


def classify_thinking_control(model: str) -> dict[str, Any]:
    """Describe whether this adapter can create a meaningful thinking on/off contrast."""
    normalized = str(model or "").casefold()
    if "deepseek" in normalized:
        return {
            "mode": "explicit_enabled_vs_disabled",
            "adapter_verified": True,
        }
    if "claude" in normalized:
        return {
            "mode": "adaptive_opt_in_vs_omitted",
            "adapter_verified": True,
        }
    return {
        "mode": "provider_default_or_adapter_noop",
        "adapter_verified": False,
    }


def _selected_arms(names: list[str] | tuple[str, ...] | None) -> tuple[CognitiveEvalArm, ...]:
    selected_names = tuple(names or _ARM_BY_NAME)
    if not selected_names:
        raise ValueError("cognitive eval arms cannot be empty")
    unknown = [name for name in selected_names if name not in _ARM_BY_NAME]
    if unknown:
        raise ValueError(f"unknown cognitive eval arm(s): {unknown}")
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("cognitive eval arms cannot contain duplicates")
    return tuple(_ARM_BY_NAME[name] for name in selected_names)


async def run_cognitive_scaffold_eval(
    backend: Any,
    model: str,
    tasks: list[SandboxTask],
    *,
    endpoint_id: str | None = None,
    repeats: int = 1,
    arms: list[str] | tuple[str, ...] | None = None,
    max_steps: int = 10,
    max_cost_usd: float = 0.5,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    transcript_token_cap: int = 24000,
    consecutive_error_limit: int = 3,
    effort: str | None = None,
    scaffold_mode: str = "runtime_checkpoint",
    checkpoint_period: int = 3,
    tool_result_lifecycle: str = "full",
    tool_result_live_reads: int = 3,
    run_id: str = "",
    bootstrap_samples: int | None = None,
) -> dict[str, Any]:
    if not tasks:
        raise ValueError("cognitive eval tasks cannot be empty")
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("cognitive eval task ids must be unique")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    if scaffold_mode not in _SCAFFOLD_MODES:
        raise ValueError(f"unknown cognitive scaffold mode: {scaffold_mode!r}")
    if (
        isinstance(checkpoint_period, bool)
        or not isinstance(checkpoint_period, int)
        or checkpoint_period <= 0
    ):
        raise ValueError("checkpoint_period must be a positive integer")
    if tool_result_lifecycle not in {"full", "compact"}:
        raise ValueError(f"unknown tool result lifecycle mode: {tool_result_lifecycle!r}")
    if (
        isinstance(tool_result_live_reads, bool)
        or not isinstance(tool_result_live_reads, int)
        or tool_result_live_reads < 0
    ):
        raise ValueError("tool_result_live_reads must be a non-negative integer")
    selected_arms = _selected_arms(arms)
    run_id = run_id or f"cognitive-{int(time.time() * 1000)}"
    cases: list[dict[str, Any]] = []
    actual_model_calls = 0
    actual_cost_usd = 0.0

    for task in tasks:
        for repeat in range(repeats):
            for arm in selected_arms:
                run_dir = make_run_dir(prefix="brainregion-cognitive-eval-")
                materialize_fixture(task, Path(run_dir))
                try:
                    trajectory = await run_agent(
                        backend,
                        model,
                        task,
                        run_dir=run_dir,
                        arm="none",
                        max_steps=max_steps,
                        max_cost_usd=max_cost_usd,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        transcript_token_cap=transcript_token_cap,
                        consecutive_error_limit=consecutive_error_limit,
                        endpoint_id=endpoint_id,
                        thinking=arm.native_thinking,
                        effort=effort if arm.native_thinking else None,
                        cognitive_scaffold=arm.external_scaffold,
                        cognitive_scaffold_mode=scaffold_mode,
                        cognitive_checkpoint_period=checkpoint_period,
                        tool_result_lifecycle=tool_result_lifecycle,
                        tool_result_live_reads=tool_result_live_reads,
                    )
                    actual_model_calls += trajectory.n_steps
                    actual_cost_usd += float(trajectory.total_main_cost_usd)
                    usage = normalize_usage(trajectory.total_main_usage)
                    target_steps = [
                        step for step in trajectory.progress_trace if step["target_fingerprint"]
                    ]
                    repeated_targets = sum(not step["target_is_new"] for step in target_steps)
                    scaffold_metrics = (
                        trajectory.cognitive_state.public_metrics()
                        if trajectory.cognitive_state
                        else _disabled_scaffold_metrics()
                    )
                    lifecycle_metrics = trajectory.tool_result_lifecycle
                    cases.append(
                        {
                            "task_id": task.id,
                            "repeat": repeat,
                            "arm": arm.name,
                            "native_thinking": arm.native_thinking,
                            "external_scaffold": arm.external_scaffold,
                            "scaffold_mode": scaffold_mode if arm.external_scaffold else None,
                            "solved": trajectory.tests_green,
                            "protocol_completed": trajectory.done,
                            "termination_reason": trajectory.termination_reason,
                            "infrastructure_error": trajectory.termination_reason == "model_error",
                            "steps": trajectory.n_steps,
                            "workspace_effects": trajectory.workspace_effects,
                            "repeated_target_rate": (
                                repeated_targets / len(target_steps) if target_steps else None
                            ),
                            "main_input_tokens": usage["input_tokens"],
                            "main_input_attribution": trajectory.main_input_attribution,
                            "main_total_tokens": usage["total_tokens"],
                            "reasoning_tokens": usage["reasoning_tokens"],
                            "cost_usd": float(trajectory.total_main_cost_usd),
                            "cognitive_scaffold": scaffold_metrics,
                            "tool_result_lifecycle": lifecycle_metrics,
                            "progress_trace": trajectory.progress_trace,
                            "contains_state_content": False,
                            "contains_reasoning": False,
                            "contains_tool_results": False,
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - preserve matched matrix
                    cases.append(
                        {
                            "task_id": task.id,
                            "repeat": repeat,
                            "arm": arm.name,
                            "native_thinking": arm.native_thinking,
                            "external_scaffold": arm.external_scaffold,
                            "scaffold_mode": scaffold_mode if arm.external_scaffold else None,
                            "solved": False,
                            "protocol_completed": False,
                            "termination_reason": "runner_error",
                            "infrastructure_error": True,
                            "steps": 0,
                            "workspace_effects": 0,
                            "repeated_target_rate": None,
                            "main_input_tokens": 0,
                            "main_input_attribution": merge_input_attributions([]),
                            "main_total_tokens": 0,
                            "reasoning_tokens": 0,
                            "cost_usd": 0.0,
                            "cognitive_scaffold": _disabled_scaffold_metrics(),
                            "tool_result_lifecycle": _disabled_lifecycle_metrics(
                                tool_result_lifecycle,
                                tool_result_live_reads,
                            ),
                            "progress_trace": [],
                            "error": f"runner_error: {exc}"[:500],
                            "contains_state_content": False,
                            "contains_reasoning": False,
                            "contains_tool_results": False,
                        }
                    )
                finally:
                    cleanup_run_dir(run_dir)

    summary = summarize_cognitive_records(
        cases,
        run_id=run_id,
        bootstrap_samples=bootstrap_samples,
    )
    summary["cases"] = cases
    summary["execution"] = {
        "model": model,
        "endpoint_id": endpoint_id,
        "effort": effort,
        "scaffold_mode": scaffold_mode,
        "checkpoint_period": checkpoint_period,
        "tool_result_lifecycle": tool_result_lifecycle,
        "tool_result_live_reads": tool_result_live_reads,
        "thinking_control": classify_thinking_control(model),
        "max_steps": max_steps,
        "actual_model_calls": actual_model_calls,
        "actual_cost_usd": actual_cost_usd,
        "contains_trajectories": False,
        "contains_state_content": False,
        "contains_reasoning": False,
    }
    return summary


def summarize_cognitive_records(
    records: list[dict[str, Any]],
    *,
    run_id: str = "",
    bootstrap_samples: int | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("cognitive records cannot be empty")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        arm = str(record.get("arm") or "")
        if arm not in _ARM_BY_NAME:
            raise ValueError(f"unknown cognitive eval arm: {arm!r}")
        grouped.setdefault(arm, []).append(record)
    per_arm: dict[str, Any] = {}
    for arm, arm_records in grouped.items():
        valid = [record for record in arm_records if not record.get("infrastructure_error")]
        updates = sum(
            int((record.get("cognitive_scaffold") or {}).get("update_attempts") or 0)
            for record in valid
        )
        failures = sum(
            int((record.get("cognitive_scaffold") or {}).get("update_failures") or 0)
            for record in valid
        )
        runtime_checkpoint_counts = [
            int((record.get("cognitive_scaffold") or {}).get("checkpoint_count") or 0)
            for record in valid
            if (record.get("cognitive_scaffold") or {}).get("mode") == "runtime_checkpoint"
        ]
        input_attribution = _summarize_input_attribution(valid)
        compacted_results = [
            int((record.get("tool_result_lifecycle") or {}).get("compacted_results") or 0)
            for record in valid
        ]
        estimated_removed = [
            int(
                (record.get("tool_result_lifecycle") or {}).get(
                    "estimated_input_tokens_avoided"
                )
                or 0
            )
            for record in valid
        ]
        per_arm[arm] = {
            "n_runs": len(arm_records),
            "n_valid_runs": len(valid),
            "solve_rate": _mean(valid, "solved"),
            "protocol_completion_rate": _mean(valid, "protocol_completed"),
            "mean_steps": _mean(valid, "steps"),
            "mean_main_input_tokens": _mean(valid, "main_input_tokens"),
            "mean_main_total_tokens": _mean(valid, "main_total_tokens"),
            "mean_reasoning_tokens": _mean(valid, "reasoning_tokens"),
            "mean_cost_usd": _mean(valid, "cost_usd"),
            "mean_repeated_target_rate": _mean(valid, "repeated_target_rate"),
            "scaffold_update_success_rate": (updates - failures) / updates if updates else None,
            "mean_checkpoint_count": (
                sum(runtime_checkpoint_counts) / len(runtime_checkpoint_counts)
                if runtime_checkpoint_counts
                else None
            ),
            "input_attribution": input_attribution,
            "mean_compacted_tool_results": (
                sum(compacted_results) / len(compacted_results) if compacted_results else None
            ),
            "mean_estimated_tool_input_tokens_avoided": (
                sum(estimated_removed) / len(estimated_removed) if estimated_removed else None
            ),
            "infrastructure_failures": len(arm_records) - len(valid),
        }

    run_id = run_id or f"cognitive-summary-{int(time.time() * 1000)}"
    task_rows = _complete_task_rows(records)
    contrasts = {
        "native_without_scaffold": (ARM_PLAIN, ARM_NATIVE_THINKING),
        "scaffold_without_native": (ARM_PLAIN, ARM_EXTERNAL_SCAFFOLD),
        "scaffold_with_native": (ARM_NATIVE_THINKING, ARM_COMBINED),
        "native_with_scaffold": (ARM_EXTERNAL_SCAFFOLD, ARM_COMBINED),
    }
    effects: dict[str, Any] = {}
    for name, (control, treatment) in contrasts.items():
        effects[name] = _contrast_summary(
            task_rows,
            control,
            treatment,
            run_id=run_id,
            name=name,
            bootstrap_samples=bootstrap_samples,
        )
    interactions: dict[str, Any] = {}
    for metric in _METRICS:
        interactions[metric] = bootstrap_statistic(
            task_rows,
            lambda sample, metric=metric: _interaction(sample, metric),
            B=bootstrap_samples,
            seed=seed_for(run_id, f"cognitive-interaction:{metric}"),
        )
    native_records = [
        record
        for record in records
        if record.get("native_thinking") and not record.get("infrastructure_error")
    ]
    control_records = [
        record
        for record in records
        if not record.get("native_thinking") and not record.get("infrastructure_error")
    ]
    native_observed = any(int(record.get("reasoning_tokens") or 0) > 0 for record in native_records)
    control_observed = any(int(record.get("reasoning_tokens") or 0) > 0 for record in control_records)
    return {
        "run_id": run_id,
        "arms": [arm.name for arm in COGNITIVE_EVAL_ARMS if arm.name in grouped],
        "n_tasks": len({record["task_id"] for record in records}),
        "n_runs": len(records),
        "bootstrap_unit": "task",
        "per_arm": per_arm,
        "effects": effects,
        "interaction": interactions,
        "native_thinking_requested": bool(native_records),
        "native_thinking_observed": native_observed,
        "control_reasoning_observed": control_observed,
        "thinking_telemetry_status": (
            "contaminated_control"
            if control_observed
            else "telemetry_confirmed"
            if native_observed
            else "request_only"
        ),
        "native_thinking_observation_note": (
            "reasoning_tokens>0 observed"
            if native_observed
            else "backend received thinking=True, but reasoning token telemetry did not confirm provider-side thinking"
        ),
        "contains_state_content": False,
        "contains_reasoning": False,
        "contains_tool_results": False,
    }


def render_cognitive_eval_summary(report: dict[str, Any]) -> str:
    lines = [
        f"### cognitive 2x2 {report['run_id']} (tasks={report['n_tasks']}, runs={report['n_runs']})",
        f"native_thinking_observed={report.get('native_thinking_observed')} "
        f"telemetry={report.get('thinking_telemetry_status')} "
        f"control={((report.get('execution') or {}).get('thinking_control') or {}).get('mode')} "
        f"scaffold={((report.get('execution') or {}).get('scaffold_mode'))} "
        f"period={((report.get('execution') or {}).get('checkpoint_period'))} "
        f"tool_results={((report.get('execution') or {}).get('tool_result_lifecycle'))} "
        f"actual_cost=${float((report.get('execution') or {}).get('actual_cost_usd') or 0):.4f}",
    ]
    for arm, summary in (report.get("per_arm") or {}).items():
        input_categories = (summary.get("input_attribution") or {}).get("categories") or {}
        lines.append(
            f"  {arm}: solve={summary.get('solve_rate')} completed={summary.get('protocol_completion_rate')} "
            f"steps={summary.get('mean_steps')} tokens={summary.get('mean_main_total_tokens')} "
            f"reasoning={summary.get('mean_reasoning_tokens')} cost=${float(summary.get('mean_cost_usd') or 0):.4f} "
            f"scaffold_updates={summary.get('scaffold_update_success_rate')} "
            f"checkpoints={summary.get('mean_checkpoint_count')} "
            f"input_mix(tool={_mean_category_tokens(input_categories, 'tool_transcript')},"
            f"checkpoint={_mean_category_tokens(input_categories, 'checkpoint')},"
            f"model={_mean_category_tokens(input_categories, 'model_transcript')}) "
            f"receipts={summary.get('mean_compacted_tool_results')} "
            f"estimated_avoided={summary.get('mean_estimated_tool_input_tokens_avoided')}"
        )
    for name, effect in (report.get("effects") or {}).items():
        point = ((effect.get("deltas") or {}).get("solved") or {}).get("point")
        lines.append(f"  effect/{name}: solve_delta={point} n_tasks={effect.get('n_tasks')}")
    interaction = ((report.get("interaction") or {}).get("solved") or {}).get("point")
    lines.append(f"  interaction/solved={interaction}")
    return "\n".join(lines)


def _disabled_scaffold_metrics() -> dict[str, Any]:
    return {
        "enabled": False,
        "contains_state_content": False,
        "contains_reasoning": False,
    }


def _disabled_lifecycle_metrics(mode: str, live_read_results: int) -> dict[str, Any]:
    return {
        "mode": mode,
        "enabled": mode == "compact",
        "policy": "evidence_pinned_receipt_v1",
        "live_read_results": live_read_results,
        "tool_results_observed": 0,
        "compaction_passes": 0,
        "compacted_results": 0,
        "active_receipts": 0,
        "body_characters_removed": 0,
        "body_estimated_tokens_removed": 0,
        "estimated_input_tokens_avoided": 0,
        "first_compaction_step": None,
        "compacted_by_tool": {},
        "contains_result_content": False,
        "contains_reasoning": False,
    }


def _summarize_input_attribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    merged = merge_input_attributions(
        record.get("main_input_attribution") or {} for record in records
    )
    n_runs = len(records)
    categories = {}
    for category, values in (merged.get("categories") or {}).items():
        categories[category] = {
            **values,
            "mean_actual_input_tokens": (
                values["actual_input_tokens"] / n_runs if n_runs else None
            ),
            "mean_estimated_input_tokens": (
                values["estimated_tokens"] / n_runs if n_runs else None
            ),
        }
    return {
        **merged,
        "n_runs": n_runs,
        "mean_actual_input_tokens": (
            merged["actual_input_tokens"] / n_runs if n_runs else None
        ),
        "categories": categories,
    }


def _mean_category_tokens(categories: dict[str, Any], category: str) -> float | None:
    values = categories.get(category) or {}
    value = values.get("mean_actual_input_tokens")
    return round(float(value), 1) if value is not None else None


def _complete_task_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for record in records:
        grouped.setdefault(record["task_id"], {}).setdefault(record["repeat"], {})[
            record["arm"]
        ] = record
    rows = []
    required = set(_ARM_BY_NAME)
    for task_id, repeats in grouped.items():
        complete = [
            arms
            for arms in repeats.values()
            if required.issubset(arms)
            and not any(arms[name].get("infrastructure_error") for name in required)
        ]
        if not complete:
            continue
        row: dict[str, Any] = {"task_id": task_id, "matched_repeats": len(complete)}
        for arm in required:
            row[arm] = {
                metric: _mean([arms[arm] for arms in complete], metric) for metric in _METRICS
            }
        rows.append(row)
    return rows


def _contrast_summary(
    rows: list[dict[str, Any]],
    control: str,
    treatment: str,
    *,
    run_id: str,
    name: str,
    bootstrap_samples: int | None,
) -> dict[str, Any]:
    deltas = {}
    for metric in _METRICS:
        deltas[metric] = bootstrap_statistic(
            rows,
            lambda sample, metric=metric: _paired_delta(sample, control, treatment, metric),
            B=bootstrap_samples,
            seed=seed_for(run_id, f"cognitive:{name}:{metric}"),
        )
    return {
        "control": control,
        "treatment": treatment,
        "n_tasks": len(rows),
        "n_matched_repeats": sum(row["matched_repeats"] for row in rows),
        "deltas": deltas,
    }


def _paired_delta(
    rows: list[dict[str, Any]], control: str, treatment: str, metric: str
) -> float | None:
    values = [
        row[treatment][metric] - row[control][metric]
        for row in rows
        if row[treatment][metric] is not None and row[control][metric] is not None
    ]
    return sum(values) / len(values) if values else None


def _interaction(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = []
    for row in rows:
        points = [row[arm][metric] for arm in _ARM_BY_NAME]
        if any(point is None for point in points):
            continue
        values.append(
            (row[ARM_COMBINED][metric] - row[ARM_NATIVE_THINKING][metric])
            - (row[ARM_EXTERNAL_SCAFFOLD][metric] - row[ARM_PLAIN][metric])
        )
    return sum(values) / len(values) if values else None


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


__all__ = [
    "ARM_COMBINED",
    "ARM_EXTERNAL_SCAFFOLD",
    "ARM_NATIVE_THINKING",
    "ARM_PLAIN",
    "COGNITIVE_EVAL_ARMS",
    "CognitiveEvalArm",
    "classify_thinking_control",
    "render_cognitive_eval_summary",
    "run_cognitive_scaffold_eval",
    "summarize_cognitive_records",
]
