"""Matched evaluation for passive context and functional Region collaboration."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brainregion.core.context import ContextBlock
from brainregion.eval.stats import bootstrap_statistic, seed_for
from brainregion.runtime import normalize_usage
from brainregion.workspace import read_text
from brainregion.workspace.files import scoped_workspace_root

from .input_attribution import merge_input_attributions
from .isolation import cleanup_run_dir, make_run_dir, materialize_fixture
from .loop import run_agent
from .regions import EvidenceRegion, VerificationOptionRegion
from .task import SandboxTask

ARM_MAIN_ONLY = "main_only"
ARM_PASSIVE_CONTEXT = "passive_context"
ARM_EVIDENCE_REGION = "evidence_region"
ARM_EVIDENCE_VERIFICATION = "evidence_verification_regions"


@dataclass(frozen=True)
class FunctionalRegionEvalArm:
    name: str
    passive_context: bool = False
    evidence_region: bool = False
    verification_region: bool = False


FUNCTIONAL_REGION_EVAL_ARMS: tuple[FunctionalRegionEvalArm, ...] = (
    FunctionalRegionEvalArm(ARM_MAIN_ONLY),
    FunctionalRegionEvalArm(ARM_PASSIVE_CONTEXT, passive_context=True),
    FunctionalRegionEvalArm(ARM_EVIDENCE_REGION, evidence_region=True),
    FunctionalRegionEvalArm(
        ARM_EVIDENCE_VERIFICATION,
        evidence_region=True,
        verification_region=True,
    ),
)
_ARM_BY_NAME = {arm.name: arm for arm in FUNCTIONAL_REGION_EVAL_ARMS}
_CONTRASTS = {
    "context_value": (ARM_MAIN_ONLY, ARM_PASSIVE_CONTEXT),
    "evidence_ownership": (ARM_PASSIVE_CONTEXT, ARM_EVIDENCE_REGION),
    "verification_delegation": (ARM_EVIDENCE_REGION, ARM_EVIDENCE_VERIFICATION),
    "end_to_end": (ARM_MAIN_ONLY, ARM_EVIDENCE_VERIFICATION),
}
_METRICS = (
    "solved",
    "protocol_completed",
    "steps",
    "main_input_tokens",
    "main_total_tokens",
    "total_cost_usd",
    "main_tool_calls",
    "region_tool_calls",
    "context_preparation_tool_calls",
    "total_tool_calls",
    "main_read_calls",
    "main_check_calls",
    "verification_runs",
    "repeated_target_rate",
)


def _selected_arms(
    names: list[str] | tuple[str, ...] | None,
) -> tuple[FunctionalRegionEvalArm, ...]:
    selected_names = tuple(names or _ARM_BY_NAME)
    if not selected_names:
        raise ValueError("functional Region eval arms cannot be empty")
    unknown = [name for name in selected_names if name not in _ARM_BY_NAME]
    if unknown:
        raise ValueError(f"unknown functional Region eval arm(s): {unknown}")
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("functional Region eval arms cannot contain duplicates")
    return tuple(_ARM_BY_NAME[name] for name in selected_names)


def _rotated_arms(
    arms: tuple[FunctionalRegionEvalArm, ...], offset: int
) -> tuple[FunctionalRegionEvalArm, ...]:
    pivot = offset % len(arms)
    return (*arms[pivot:], *arms[:pivot])


def _prepare_passive_context(
    task: SandboxTask,
) -> tuple[tuple[ContextBlock, ...], int, int]:
    """Materialize the same snapshots as EvidenceRegion without activating it."""
    region = EvidenceRegion()
    requests = region.requests(task)
    failures = 0
    for request in requests:
        try:
            result = read_text(request.path, max_bytes=request.max_bytes)
        except Exception as exc:  # noqa: BLE001 - preserve a matched passive arm
            failures += 1
            region.observe(request, error=f"{type(exc).__name__}: {exc}")
        else:
            region.observe(request, result=result)
    return region.blocks(), len(requests), failures


async def run_functional_region_eval(
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
    thinking: bool = False,
    effort: str | None = None,
    tool_result_lifecycle: str = "full",
    tool_result_live_reads: int = 3,
    run_id: str = "",
    bootstrap_samples: int | None = None,
) -> dict[str, Any]:
    if not tasks:
        raise ValueError("functional Region eval tasks cannot be empty")
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("functional Region eval task ids must be unique")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    if tool_result_lifecycle not in {"full", "compact"}:
        raise ValueError(f"unknown tool result lifecycle mode: {tool_result_lifecycle!r}")
    if (
        isinstance(tool_result_live_reads, bool)
        or not isinstance(tool_result_live_reads, int)
        or tool_result_live_reads < 0
    ):
        raise ValueError("tool_result_live_reads must be a non-negative integer")
    selected_arms = _selected_arms(arms)
    run_id = run_id or f"functional-region-{int(time.time() * 1000)}"
    cases: list[dict[str, Any]] = []
    arm_orders: dict[str, int] = {}
    actual_model_calls = 0
    actual_cost_usd = 0.0
    actual_tool_calls = 0

    for task_index, task in enumerate(tasks):
        for repeat in range(repeats):
            ordered_arms = _rotated_arms(selected_arms, task_index + repeat)
            order_key = "->".join(arm.name for arm in ordered_arms)
            arm_orders[order_key] = arm_orders.get(order_key, 0) + 1
            for arm in ordered_arms:
                run_dir = make_run_dir(prefix="brainregion-functional-region-eval-")
                materialize_fixture(task, Path(run_dir))
                passive_blocks: tuple[ContextBlock, ...] | None = None
                preparation_calls = 0
                preparation_failures = 0
                try:
                    if arm.passive_context:
                        with scoped_workspace_root(run_dir):
                            (
                                passive_blocks,
                                preparation_calls,
                                preparation_failures,
                            ) = _prepare_passive_context(task)
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
                        thinking=thinking,
                        effort=effort if thinking else None,
                        tool_result_lifecycle=tool_result_lifecycle,
                        tool_result_live_reads=tool_result_live_reads,
                        passive_evidence_blocks=passive_blocks,
                        evidence_region=EvidenceRegion() if arm.evidence_region else None,
                        option_region=(
                            VerificationOptionRegion() if arm.verification_region else None
                        ),
                        option_continuous=arm.verification_region,
                        max_option_activations=max(1, max_steps),
                    )
                    main_tool_steps = [step for step in trajectory.steps if step.tool]
                    main_tool_calls = len(main_tool_steps)
                    region_tool_calls = int(trajectory.region_tool_calls)
                    total_tool_calls = main_tool_calls + region_tool_calls + preparation_calls
                    usage = normalize_usage(trajectory.total_main_usage)
                    target_steps = [
                        step for step in trajectory.progress_trace if step["target_fingerprint"]
                    ]
                    repeated_targets = sum(not step["target_is_new"] for step in target_steps)
                    actual_model_calls += trajectory.n_steps + trajectory.region_model_calls
                    actual_cost_usd += (
                        float(trajectory.total_main_cost_usd) + float(trajectory.total_arm_cost_usd)
                    )
                    actual_tool_calls += total_tool_calls
                    workbench = dict(trajectory.region_workbench)
                    evidence_blocks = int((workbench.get("by_region") or {}).get("evidence") or 0)
                    region_evidence_attempts = max(
                        0, region_tool_calls - trajectory.verification_runs
                    )
                    evidence_failures = (
                        preparation_failures
                        if arm.passive_context
                        else max(0, region_evidence_attempts - evidence_blocks)
                    )
                    cases.append(
                        {
                            "task_id": task.id,
                            "repeat": repeat,
                            "arm": arm.name,
                            "passive_context": arm.passive_context,
                            "evidence_region": arm.evidence_region,
                            "verification_region": arm.verification_region,
                            "solved": trajectory.tests_green,
                            "protocol_completed": trajectory.done,
                            "termination_reason": trajectory.termination_reason,
                            "infrastructure_error": trajectory.termination_reason == "model_error",
                            "steps": trajectory.n_steps,
                            "workspace_effects": trajectory.workspace_effects,
                            "verification_runs": trajectory.verification_runs,
                            "automatic_region_activations": trajectory.automatic_region_activations,
                            "main_tool_calls": main_tool_calls,
                            "region_tool_calls": region_tool_calls,
                            "region_model_calls": trajectory.region_model_calls,
                            "context_preparation_tool_calls": preparation_calls,
                            "context_preparation_failures": evidence_failures,
                            "total_tool_calls": total_tool_calls,
                            "main_read_calls": sum(
                                step.tool == "read_text" for step in main_tool_steps
                            ),
                            "main_check_calls": sum(
                                step.tool == "workspace_run_check" for step in main_tool_steps
                            ),
                            "repeated_target_rate": (
                                repeated_targets / len(target_steps) if target_steps else None
                            ),
                            "main_input_tokens": usage["input_tokens"],
                            "main_input_attribution": trajectory.main_input_attribution,
                            "main_total_tokens": usage["total_tokens"],
                            "main_cost_usd": float(trajectory.total_main_cost_usd),
                            "region_cost_usd": float(trajectory.total_arm_cost_usd),
                            "total_cost_usd": float(trajectory.total_main_cost_usd)
                            + float(trajectory.total_arm_cost_usd),
                            "region_workbench": workbench,
                            "progress_trace": trajectory.progress_trace,
                            "contains_context_content": False,
                            "contains_reasoning": False,
                            "contains_tool_results": False,
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - preserve the matched matrix
                    actual_tool_calls += preparation_calls
                    cases.append(
                        _runner_error_case(
                            task.id,
                            repeat,
                            arm,
                            preparation_calls=preparation_calls,
                            preparation_failures=preparation_failures,
                            error=exc,
                        )
                    )
                finally:
                    cleanup_run_dir(run_dir)

    report = summarize_functional_region_records(
        cases,
        run_id=run_id,
        bootstrap_samples=bootstrap_samples,
    )
    report["cases"] = cases
    report["execution"] = {
        "model": model,
        "endpoint_id": endpoint_id,
        "thinking": thinking,
        "effort": effort if thinking else None,
        "tool_result_lifecycle": tool_result_lifecycle,
        "tool_result_live_reads": tool_result_live_reads,
        "max_steps": max_steps,
        "arm_order_counts": arm_orders,
        "actual_model_calls": actual_model_calls,
        "actual_tool_calls": actual_tool_calls,
        "actual_cost_usd": actual_cost_usd,
        "passive_region_input_contract": "model_visible_context_equivalent_v1",
        "contains_trajectories": False,
        "contains_context_content": False,
        "contains_reasoning": False,
        "contains_tool_results": False,
    }
    return report


def summarize_functional_region_records(
    records: list[dict[str, Any]],
    *,
    run_id: str = "",
    bootstrap_samples: int | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("functional Region records cannot be empty")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        arm = str(record.get("arm") or "")
        if arm not in _ARM_BY_NAME:
            raise ValueError(f"unknown functional Region eval arm: {arm!r}")
        grouped.setdefault(arm, []).append(record)

    per_arm: dict[str, Any] = {}
    for arm, arm_records in grouped.items():
        valid = [record for record in arm_records if not record.get("infrastructure_error")]
        per_arm[arm] = {
            "n_runs": len(arm_records),
            "n_valid_runs": len(valid),
            "solve_rate": _mean(valid, "solved"),
            "protocol_completion_rate": _mean(valid, "protocol_completed"),
            "mean_steps": _mean(valid, "steps"),
            "mean_main_input_tokens": _mean(valid, "main_input_tokens"),
            "mean_main_total_tokens": _mean(valid, "main_total_tokens"),
            "mean_main_cost_usd": _mean(valid, "main_cost_usd"),
            "mean_region_cost_usd": _mean(valid, "region_cost_usd"),
            "mean_total_cost_usd": _mean(valid, "total_cost_usd"),
            "mean_main_tool_calls": _mean(valid, "main_tool_calls"),
            "mean_region_tool_calls": _mean(valid, "region_tool_calls"),
            "mean_context_preparation_tool_calls": _mean(
                valid, "context_preparation_tool_calls"
            ),
            "mean_total_tool_calls": _mean(valid, "total_tool_calls"),
            "mean_main_read_calls": _mean(valid, "main_read_calls"),
            "mean_main_check_calls": _mean(valid, "main_check_calls"),
            "mean_verification_runs": _mean(valid, "verification_runs"),
            "mean_region_activations": _mean(valid, "automatic_region_activations"),
            "mean_repeated_target_rate": _mean(valid, "repeated_target_rate"),
            "mean_workbench_tokens": _mean_workbench_tokens(valid),
            "input_attribution": _summarize_input_attribution(valid),
            "context_preparation_failures": sum(
                int(record.get("context_preparation_failures") or 0) for record in valid
            ),
            "infrastructure_failures": len(arm_records) - len(valid),
        }

    run_id = run_id or f"functional-region-summary-{int(time.time() * 1000)}"
    effects: dict[str, Any] = {}
    for name, (control, treatment) in _CONTRASTS.items():
        if control not in grouped or treatment not in grouped:
            continue
        rows = _paired_task_rows(records, control, treatment)
        raw_deltas = {metric: _paired_delta(rows, metric) for metric in _METRICS}
        bootstrap_deltas = {
            metric: bootstrap_statistic(
                rows,
                lambda sample, metric=metric: _paired_delta(sample, metric),
                B=bootstrap_samples,
                seed=seed_for(run_id, f"functional-region:{name}:{metric}"),
            )
            for metric in _METRICS
        }
        effects[name] = {
            "control": control,
            "treatment": treatment,
            "n_tasks": len(rows),
            "n_matched_repeats": sum(row["matched_repeats"] for row in rows),
            "raw_deltas": raw_deltas,
            "bootstrap_deltas": bootstrap_deltas,
        }
    return {
        "run_id": run_id,
        "arms": [arm.name for arm in FUNCTIONAL_REGION_EVAL_ARMS if arm.name in grouped],
        "n_tasks": len({record["task_id"] for record in records}),
        "n_runs": len(records),
        "bootstrap_unit": "task",
        "delta_direction": "treatment_minus_control",
        "per_arm": per_arm,
        "effects": effects,
        "interpretation": {
            "context_value": "additional grounded context, without Region activation",
            "evidence_ownership": "Region ownership with model-visible context held equivalent",
            "verification_delegation": "host-controlled objective verification after a main effect",
            "end_to_end": "complete functional Region pipeline versus main brain only",
        },
        "contains_context_content": False,
        "contains_reasoning": False,
        "contains_tool_results": False,
    }


def render_functional_region_eval_summary(report: dict[str, Any]) -> str:
    execution = report.get("execution") or {}
    lines = [
        f"### functional Region eval {report['run_id']} "
        f"(tasks={report['n_tasks']}, runs={report['n_runs']})",
        f"model={execution.get('model', '')} thinking={execution.get('thinking')} "
        f"tool_results={execution.get('tool_result_lifecycle')} "
        f"actual_calls(model={execution.get('actual_model_calls')},tool={execution.get('actual_tool_calls')}) "
        f"cost=${float(execution.get('actual_cost_usd') or 0):.4f}",
    ]
    for arm, summary in (report.get("per_arm") or {}).items():
        lines.append(
            f"  {arm}: solve={summary.get('solve_rate')} "
            f"completed={summary.get('protocol_completion_rate')} steps={summary.get('mean_steps')} "
            f"tools(main={summary.get('mean_main_tool_calls')},"
            f"region={summary.get('mean_region_tool_calls')},"
            f"prep={summary.get('mean_context_preparation_tool_calls')},"
            f"total={summary.get('mean_total_tool_calls')}) "
            f"input={summary.get('mean_main_input_tokens')} "
            f"cost=${float(summary.get('mean_total_cost_usd') or 0):.4f}"
        )
    for name, effect in (report.get("effects") or {}).items():
        solved = (effect.get("raw_deltas") or {}).get("solved")
        tools = (effect.get("raw_deltas") or {}).get("total_tool_calls")
        lines.append(
            f"  effect/{name}: solve_delta={solved} total_tool_delta={tools} "
            f"n_tasks={effect.get('n_tasks')}"
        )
    return "\n".join(lines)


def _runner_error_case(
    task_id: str,
    repeat: int,
    arm: FunctionalRegionEvalArm,
    *,
    preparation_calls: int,
    preparation_failures: int,
    error: Exception,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "repeat": repeat,
        "arm": arm.name,
        "passive_context": arm.passive_context,
        "evidence_region": arm.evidence_region,
        "verification_region": arm.verification_region,
        "solved": False,
        "protocol_completed": False,
        "termination_reason": "runner_error",
        "infrastructure_error": True,
        "steps": 0,
        "workspace_effects": 0,
        "verification_runs": 0,
        "automatic_region_activations": 0,
        "main_tool_calls": 0,
        "region_tool_calls": 0,
        "region_model_calls": 0,
        "context_preparation_tool_calls": preparation_calls,
        "context_preparation_failures": preparation_failures,
        "total_tool_calls": preparation_calls,
        "main_read_calls": 0,
        "main_check_calls": 0,
        "repeated_target_rate": None,
        "main_input_tokens": 0,
        "main_input_attribution": merge_input_attributions([]),
        "main_total_tokens": 0,
        "main_cost_usd": 0.0,
        "region_cost_usd": 0.0,
        "total_cost_usd": 0.0,
        "region_workbench": {
            "enabled": arm.passive_context or arm.evidence_region,
            "delivery_mode": (
                "passive"
                if arm.passive_context
                else "region"
                if arm.evidence_region
                else "disabled"
            ),
            "contains_context_content": False,
        },
        "progress_trace": [],
        "error_type": type(error).__name__,
        "error_stage": "runner",
        "contains_context_content": False,
        "contains_reasoning": False,
        "contains_tool_results": False,
    }


def _paired_task_rows(
    records: list[dict[str, Any]], control: str, treatment: str
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for record in records:
        grouped.setdefault(record["task_id"], {}).setdefault(record["repeat"], {})[
            record["arm"]
        ] = record
    rows: list[dict[str, Any]] = []
    for task_id, repeats in grouped.items():
        matched = [
            arms
            for arms in repeats.values()
            if control in arms
            and treatment in arms
            and not arms[control].get("infrastructure_error")
            and not arms[treatment].get("infrastructure_error")
        ]
        if not matched:
            continue
        row: dict[str, Any] = {"task_id": task_id, "matched_repeats": len(matched)}
        for metric in _METRICS:
            values = [
                float(arms[treatment][metric]) - float(arms[control][metric])
                for arms in matched
                if arms[treatment].get(metric) is not None
                and arms[control].get(metric) is not None
            ]
            row[metric] = sum(values) / len(values) if values else None
        rows.append(row)
    return rows


def _paired_delta(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    return sum(values) / len(values) if values else None


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
        }
    return {
        **merged,
        "n_runs": n_runs,
        "mean_actual_input_tokens": (
            merged["actual_input_tokens"] / n_runs if n_runs else None
        ),
        "categories": categories,
    }


def _mean_workbench_tokens(records: list[dict[str, Any]]) -> float | None:
    values = [
        float((record.get("region_workbench") or {}).get("estimated_tokens") or 0)
        for record in records
        if (record.get("region_workbench") or {}).get("enabled")
    ]
    return sum(values) / len(values) if values else None


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


__all__ = [
    "ARM_EVIDENCE_REGION",
    "ARM_EVIDENCE_VERIFICATION",
    "ARM_MAIN_ONLY",
    "ARM_PASSIVE_CONTEXT",
    "FUNCTIONAL_REGION_EVAL_ARMS",
    "FunctionalRegionEvalArm",
    "render_functional_region_eval_summary",
    "run_functional_region_eval",
    "summarize_functional_region_records",
]
