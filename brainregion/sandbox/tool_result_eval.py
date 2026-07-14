"""Matched evaluation for full and compact tool-result transcript lifecycles."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from brainregion.eval.stats import bootstrap_statistic, seed_for
from brainregion.runtime import normalize_usage

from .input_attribution import merge_input_attributions
from .isolation import cleanup_run_dir, make_run_dir, materialize_fixture
from .loop import run_agent
from .prefix_replay import ModelPrefixTape, PrefixReplayBackend
from .task import SandboxTask

ARM_FULL = "full"
ARM_COMPACT = "compact"
TOOL_RESULT_EVAL_ARMS = (ARM_FULL, ARM_COMPACT)

_METRICS = (
    "solved",
    "protocol_completed",
    "steps",
    "workspace_effects",
    "main_input_tokens",
    "main_total_tokens",
    "tool_transcript_input_tokens",
    "reasoning_tokens",
    "cost_usd",
    "repeated_target_rate",
    "repeated_retrieval_rate",
    "repeated_read_rate",
)
_RETRIEVAL_TOOLS = frozenset({"search_text", "read_text", "inspect_file"})
_MAX_SAFE_SHARED_PREFIX_TURNS = 2


def _selected_arms(names: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    requested = tuple(names or TOOL_RESULT_EVAL_ARMS)
    if not requested:
        raise ValueError("tool-result eval arms cannot be empty")
    unknown = [name for name in requested if name not in TOOL_RESULT_EVAL_ARMS]
    if unknown:
        raise ValueError(f"unknown tool-result eval arm(s): {unknown}")
    if len(set(requested)) != len(requested):
        raise ValueError("tool-result eval arms cannot contain duplicates")
    return tuple(arm for arm in TOOL_RESULT_EVAL_ARMS if arm in requested)


async def run_tool_result_eval(
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
    cognitive_scaffold: bool = False,
    scaffold_mode: str = "runtime_checkpoint",
    checkpoint_period: int = 3,
    tool_result_live_reads: int = 3,
    shared_prefix_turns: int = _MAX_SAFE_SHARED_PREFIX_TURNS,
    run_id: str = "",
    bootstrap_samples: int | None = None,
) -> dict[str, Any]:
    """Run fresh, counterbalanced full/compact arms on each task and repeat."""
    if not tasks:
        raise ValueError("tool-result eval tasks cannot be empty")
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("tool-result eval task ids must be unique")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    if scaffold_mode not in {"runtime_checkpoint", "model_managed"}:
        raise ValueError(f"unknown cognitive scaffold mode: {scaffold_mode!r}")
    if (
        isinstance(checkpoint_period, bool)
        or not isinstance(checkpoint_period, int)
        or checkpoint_period <= 0
    ):
        raise ValueError("checkpoint_period must be a positive integer")
    if (
        isinstance(tool_result_live_reads, bool)
        or not isinstance(tool_result_live_reads, int)
        or tool_result_live_reads < 0
    ):
        raise ValueError("tool_result_live_reads must be a non-negative integer")
    if (
        isinstance(shared_prefix_turns, bool)
        or not isinstance(shared_prefix_turns, int)
        or not 0 <= shared_prefix_turns <= _MAX_SAFE_SHARED_PREFIX_TURNS
    ):
        raise ValueError(
            "shared_prefix_turns must be an integer between 0 and 2; "
            "later turns may already contain the treatment"
        )

    selected_arms = _selected_arms(arms)
    run_id = run_id or f"tool-result-{int(time.time() * 1000)}"
    cases: list[dict[str, Any]] = []
    actual_model_calls = 0
    accounted_model_calls = 0
    replayed_model_calls = 0
    actual_cost_usd = 0.0
    accounted_cost_usd = 0.0
    replayed_accounted_cost_usd = 0.0
    execution_order = 0
    arm_order_counts = {"full_first": 0, "compact_first": 0, "single_arm": 0}

    for task_index, task in enumerate(tasks):
        for repeat in range(repeats):
            ordered_arms = list(selected_arms)
            if len(ordered_arms) > 1 and (task_index + repeat) % 2:
                ordered_arms.reverse()
            if len(ordered_arms) == 1:
                arm_order_counts["single_arm"] += 1
            elif ordered_arms[0] == ARM_FULL:
                arm_order_counts["full_first"] += 1
            else:
                arm_order_counts["compact_first"] += 1
            pair_prefix_enabled = shared_prefix_turns > 0 and len(ordered_arms) > 1
            prefix_tape = ModelPrefixTape(
                turn_limit=shared_prefix_turns if pair_prefix_enabled else 0
            )
            for arm_index, lifecycle_mode in enumerate(ordered_arms):
                prefix_role = (
                    "capture"
                    if pair_prefix_enabled and arm_index == 0
                    else "replay"
                    if pair_prefix_enabled
                    else "disabled"
                )
                run_backend = PrefixReplayBackend(
                    backend,
                    prefix_tape,
                    role=prefix_role,
                )
                run_dir = make_run_dir(prefix="brainregion-tool-result-eval-")
                materialize_fixture(task, Path(run_dir))
                execution_order += 1
                try:
                    trajectory = await run_agent(
                        run_backend,
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
                        cognitive_scaffold=cognitive_scaffold,
                        cognitive_scaffold_mode=scaffold_mode,
                        cognitive_checkpoint_period=checkpoint_period,
                        tool_result_lifecycle=lifecycle_mode,
                        tool_result_live_reads=tool_result_live_reads,
                    )
                    cases.append(
                        _case_from_trajectory(
                            trajectory,
                            task_id=task.id,
                            repeat=repeat,
                            arm=lifecycle_mode,
                            execution_order=execution_order,
                            shared_prefix=run_backend.public_metrics(),
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - preserve the matched matrix
                    cases.append(
                        _error_case(
                            task_id=task.id,
                            repeat=repeat,
                            arm=lifecycle_mode,
                            execution_order=execution_order,
                            live_read_results=tool_result_live_reads,
                            shared_prefix=run_backend.public_metrics(),
                            error=exc,
                        )
                    )
                finally:
                    prefix_metrics = run_backend.public_metrics()
                    actual_model_calls += int(prefix_metrics["provider_calls"])
                    accounted_model_calls += int(prefix_metrics["accounted_calls"])
                    replayed_model_calls += int(prefix_metrics["replayed_calls"])
                    actual_cost_usd += float(prefix_metrics["provider_cost_usd"])
                    accounted_cost_usd += float(prefix_metrics["accounted_cost_usd"])
                    replayed_accounted_cost_usd += float(
                        prefix_metrics["replayed_accounted_cost_usd"]
                    )
                    cleanup_run_dir(run_dir)

    report = summarize_tool_result_records(
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
        "cognitive_scaffold": cognitive_scaffold,
        "scaffold_mode": scaffold_mode if cognitive_scaffold else None,
        "checkpoint_period": checkpoint_period if cognitive_scaffold else None,
        "tool_result_live_reads": tool_result_live_reads,
        "shared_prefix_turns": shared_prefix_turns,
        "shared_prefix_policy": "exact_response_replay_before_earliest_treatment_v1",
        "max_steps": max_steps,
        "arm_order_policy": "alternating_by_task_repeat",
        "arm_order_counts": arm_order_counts,
        "counterbalanced_order": (
            arm_order_counts["full_first"] > 0
            and arm_order_counts["compact_first"] > 0
        ),
        "actual_model_calls": actual_model_calls,
        "accounted_model_calls": accounted_model_calls,
        "replayed_model_calls": replayed_model_calls,
        "actual_cost_usd": actual_cost_usd,
        "accounted_cost_usd": accounted_cost_usd,
        "replayed_accounted_cost_usd": replayed_accounted_cost_usd,
        "contains_trajectories": False,
        "contains_result_content": False,
        "contains_reasoning": False,
    }
    return report


def _case_from_trajectory(
    trajectory: Any,
    *,
    task_id: str,
    repeat: int,
    arm: str,
    execution_order: int,
    shared_prefix: dict[str, Any],
) -> dict[str, Any]:
    usage = normalize_usage(trajectory.total_main_usage)
    trace = trajectory.progress_trace
    target_steps = [step for step in trace if step["target_fingerprint"]]
    retrieval_steps = [step for step in trace if step["operation"] in _RETRIEVAL_TOOLS]
    read_steps = [step for step in trace if step["operation"] == "read_text"]
    attribution = trajectory.main_input_attribution
    return {
        "task_id": task_id,
        "repeat": repeat,
        "arm": arm,
        "execution_order": execution_order,
        "solved": trajectory.tests_green,
        "protocol_completed": trajectory.done,
        "termination_reason": trajectory.termination_reason,
        "infrastructure_error": trajectory.termination_reason == "model_error",
        "steps": trajectory.n_steps,
        "workspace_effects": trajectory.workspace_effects,
        "main_input_tokens": usage["input_tokens"],
        "main_total_tokens": usage["total_tokens"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "tool_transcript_input_tokens": _category_tokens(attribution, "tool_transcript"),
        "cost_usd": float(trajectory.total_main_cost_usd),
        "repeated_target_rate": _repeat_rate(target_steps),
        "retrieval_calls": len(retrieval_steps),
        "repeated_retrieval_calls": _repeat_count(retrieval_steps),
        "repeated_retrieval_rate": _repeat_rate(retrieval_steps),
        "read_calls": len(read_steps),
        "repeated_read_calls": _repeat_count(read_steps),
        "repeated_read_rate": _repeat_rate(read_steps),
        "main_input_attribution": attribution,
        "tool_result_lifecycle": trajectory.tool_result_lifecycle,
        "shared_prefix": dict(shared_prefix),
        "progress_trace": trace,
        "contains_result_content": False,
        "contains_reasoning": False,
    }


def _error_case(
    *,
    task_id: str,
    repeat: int,
    arm: str,
    execution_order: int,
    live_read_results: int,
    shared_prefix: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "repeat": repeat,
        "arm": arm,
        "execution_order": execution_order,
        "solved": False,
        "protocol_completed": False,
        "termination_reason": "runner_error",
        "infrastructure_error": True,
        "steps": 0,
        "workspace_effects": 0,
        "main_input_tokens": 0,
        "main_total_tokens": 0,
        "reasoning_tokens": 0,
        "tool_transcript_input_tokens": 0,
        "cost_usd": 0.0,
        "repeated_target_rate": None,
        "retrieval_calls": 0,
        "repeated_retrieval_calls": 0,
        "repeated_retrieval_rate": None,
        "read_calls": 0,
        "repeated_read_calls": 0,
        "repeated_read_rate": None,
        "main_input_attribution": merge_input_attributions([]),
        "tool_result_lifecycle": _empty_lifecycle_metrics(arm, live_read_results),
        "shared_prefix": dict(shared_prefix),
        "progress_trace": [],
        "error": f"runner_error:{type(error).__name__}",
        "contains_result_content": False,
        "contains_reasoning": False,
    }


def summarize_tool_result_records(
    records: list[dict[str, Any]],
    *,
    run_id: str = "",
    bootstrap_samples: int | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("tool-result records cannot be empty")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        arm = str(record.get("arm") or "")
        if arm not in TOOL_RESULT_EVAL_ARMS:
            raise ValueError(f"unknown tool-result eval arm: {arm!r}")
        grouped.setdefault(arm, []).append(record)

    per_arm: dict[str, Any] = {}
    for arm in TOOL_RESULT_EVAL_ARMS:
        if arm not in grouped:
            continue
        arm_records = grouped[arm]
        valid = [record for record in arm_records if not record.get("infrastructure_error")]
        lifecycle = [record.get("tool_result_lifecycle") or {} for record in valid]
        shared_prefix = [record.get("shared_prefix") or {} for record in valid]
        input_attribution = _summarize_input_attribution(valid)
        per_arm[arm] = {
            "n_runs": len(arm_records),
            "n_valid_runs": len(valid),
            "solve_rate": _mean(valid, "solved"),
            "protocol_completion_rate": _mean(valid, "protocol_completed"),
            "mean_steps": _mean(valid, "steps"),
            "mean_workspace_effects": _mean(valid, "workspace_effects"),
            "mean_main_input_tokens": _mean(valid, "main_input_tokens"),
            "mean_main_total_tokens": _mean(valid, "main_total_tokens"),
            "mean_tool_transcript_input_tokens": _mean(
                valid, "tool_transcript_input_tokens"
            ),
            "mean_reasoning_tokens": _mean(valid, "reasoning_tokens"),
            "mean_cost_usd": _mean(valid, "cost_usd"),
            "mean_repeated_target_rate": _mean(valid, "repeated_target_rate"),
            "mean_retrieval_calls": _mean(valid, "retrieval_calls"),
            "mean_repeated_retrieval_calls": _mean(valid, "repeated_retrieval_calls"),
            "mean_repeated_retrieval_rate": _mean(valid, "repeated_retrieval_rate"),
            "mean_read_calls": _mean(valid, "read_calls"),
            "mean_repeated_read_calls": _mean(valid, "repeated_read_calls"),
            "mean_repeated_read_rate": _mean(valid, "repeated_read_rate"),
            "mean_tool_results_observed": _mean(lifecycle, "tool_results_observed"),
            "mean_compacted_results": _mean(lifecycle, "compacted_results"),
            "mean_active_receipts": _mean(lifecycle, "active_receipts"),
            "mean_body_estimated_tokens_removed": _mean(
                lifecycle, "body_estimated_tokens_removed"
            ),
            "mean_estimated_input_tokens_avoided": _mean(
                lifecycle, "estimated_input_tokens_avoided"
            ),
            "mean_provider_model_calls": _mean(shared_prefix, "provider_calls"),
            "mean_replayed_model_calls": _mean(shared_prefix, "replayed_calls"),
            "mean_actual_provider_cost_usd": _mean(
                shared_prefix, "provider_cost_usd"
            ),
            "input_attribution": input_attribution,
            "infrastructure_failures": len(arm_records) - len(valid),
        }

    run_id = run_id or f"tool-result-summary-{int(time.time() * 1000)}"
    task_rows = _complete_task_rows(records)
    pair_diagnostics = _pair_diagnostics(records)
    aligned_pairs = {
        (item["task_id"], item["repeat"])
        for item in pair_diagnostics
        if item["treatment_exposed"]
        and item["pre_exposure_trace_match"] is True
        and item["prefix_replay_valid"] is not False
    }
    aligned_rows = _complete_task_rows(records, allowed_pairs=aligned_pairs)
    return {
        "run_id": run_id,
        "arms": [arm for arm in TOOL_RESULT_EVAL_ARMS if arm in grouped],
        "n_tasks": len({record["task_id"] for record in records}),
        "n_runs": len(records),
        "bootstrap_unit": "task",
        "per_arm": per_arm,
        "matched_effect": _effect_summary(
            task_rows,
            run_id=run_id,
            label="all-matched",
            bootstrap_samples=bootstrap_samples,
        ),
        "exposure_aligned_effect": _effect_summary(
            aligned_rows,
            run_id=run_id,
            label="exposure-aligned",
            bootstrap_samples=bootstrap_samples,
        ),
        "pair_quality": _pair_quality(pair_diagnostics),
        "pair_diagnostics": pair_diagnostics,
        "contains_result_content": False,
        "contains_reasoning": False,
    }


def render_tool_result_eval_summary(report: dict[str, Any]) -> str:
    execution = report.get("execution") or {}
    lines = [
        f"### tool-result lifecycle {report['run_id']} "
        f"(tasks={report['n_tasks']}, runs={report['n_runs']})",
        f"model={execution.get('model', '')} counterbalanced={execution.get('counterbalanced_order')} "
        f"scaffold={execution.get('cognitive_scaffold')} "
        f"actual_cost=${float(execution.get('actual_cost_usd') or 0):.4f} "
        f"accounted_cost=${float(execution.get('accounted_cost_usd') or 0):.4f} "
        f"provider_calls={execution.get('actual_model_calls')} "
        f"replayed_calls={execution.get('replayed_model_calls')}",
    ]
    for arm, summary in (report.get("per_arm") or {}).items():
        lines.append(
            f"  {arm}: solve={summary.get('solve_rate')} "
            f"completed={summary.get('protocol_completion_rate')} "
            f"steps={summary.get('mean_steps')} "
            f"input={summary.get('mean_main_input_tokens')} "
            f"tool_input={summary.get('mean_tool_transcript_input_tokens')} "
            f"cost=${float(summary.get('mean_cost_usd') or 0):.4f} "
            f"repeat_retrieval={summary.get('mean_repeated_retrieval_rate')} "
            f"repeat_read={summary.get('mean_repeated_read_rate')} "
            f"receipts={summary.get('mean_compacted_results')}"
        )
    effect = report.get("matched_effect") or {}
    raw = effect.get("raw_deltas") or {}
    boot = effect.get("bootstrap_deltas") or {}
    lines.append(
        "  compact-full: "
        f"solve={raw.get('solved')} input={raw.get('main_input_tokens')} "
        f"tool_input={raw.get('tool_transcript_input_tokens')} "
        f"cost={raw.get('cost_usd')} repeat_read={raw.get('repeated_read_rate')} "
        f"n_tasks={effect.get('n_tasks')} matched_repeats={effect.get('n_matched_repeats')}"
    )
    input_ci = boot.get("main_input_tokens") or {}
    lines.append(
        f"  input_delta_ci=[{input_ci.get('low')}, {input_ci.get('high')}] "
        f"bootstrap_n={input_ci.get('n')}"
    )
    quality = report.get("pair_quality") or {}
    aligned = report.get("exposure_aligned_effect") or {}
    aligned_raw = aligned.get("raw_deltas") or {}
    lines.append(
        f"  pair_quality={quality.get('status')} exposed={quality.get('treatment_exposed_pairs')} "
        f"pre_aligned={quality.get('pre_exposure_aligned_pairs')} "
        f"pre_diverged={quality.get('pre_exposure_diverged_pairs')} "
        f"prefix_invalid={quality.get('prefix_replay_invalid_pairs')}"
    )
    lines.append(
        f"  exposure-aligned compact-full: solve={aligned_raw.get('solved')} "
        f"input={aligned_raw.get('main_input_tokens')} n_tasks={aligned.get('n_tasks')}"
    )
    return "\n".join(lines)


def _effect_summary(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    label: str,
    bootstrap_samples: int | None,
) -> dict[str, Any]:
    return {
        "control": ARM_FULL,
        "treatment": ARM_COMPACT,
        "delta_direction": "compact_minus_full",
        "n_tasks": len(rows),
        "n_matched_repeats": sum(row["matched_repeats"] for row in rows),
        "raw_deltas": {metric: _paired_delta(rows, metric) for metric in _METRICS},
        "bootstrap_deltas": {
            metric: bootstrap_statistic(
                rows,
                lambda sample, metric=metric: _paired_delta(sample, metric),
                B=bootstrap_samples,
                seed=seed_for(run_id, f"tool-result:{label}:{metric}"),
            )
            for metric in _METRICS
        },
    }


def _complete_task_rows(
    records: list[dict[str, Any]],
    *,
    allowed_pairs: set[tuple[str, int]] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for record in records:
        grouped.setdefault(record["task_id"], {}).setdefault(record["repeat"], {})[
            record["arm"]
        ] = record
    rows = []
    required = set(TOOL_RESULT_EVAL_ARMS)
    for task_id, repeats in grouped.items():
        complete = [
            arms
            for repeat, arms in repeats.items()
            if required.issubset(arms)
            and not any(arms[name].get("infrastructure_error") for name in required)
            and (allowed_pairs is None or (task_id, repeat) in allowed_pairs)
        ]
        if not complete:
            continue
        row: dict[str, Any] = {"task_id": task_id, "matched_repeats": len(complete)}
        for arm in TOOL_RESULT_EVAL_ARMS:
            row[arm] = {
                metric: _mean([pair[arm] for pair in complete], metric)
                for metric in _METRICS
            }
        rows.append(row)
    return rows


def _pair_diagnostics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["task_id"], record["repeat"]), {})[record["arm"]] = record
    diagnostics = []
    for (task_id, repeat), arms in sorted(grouped.items()):
        if not all(arm in arms for arm in TOOL_RESULT_EVAL_ARMS):
            continue
        full = arms[ARM_FULL]
        compact = arms[ARM_COMPACT]
        if full.get("infrastructure_error") or compact.get("infrastructure_error"):
            continue
        first_compaction_step = (compact.get("tool_result_lifecycle") or {}).get(
            "first_compaction_step"
        )
        full_prefix = full.get("shared_prefix") or {}
        compact_prefix = compact.get("shared_prefix") or {}
        prefix_runs = [full_prefix, compact_prefix]
        capture_prefix = next(
            (item for item in prefix_runs if item.get("role") == "capture"),
            None,
        )
        replay_prefix = next(
            (item for item in prefix_runs if item.get("role") == "replay"),
            None,
        )
        prefix_replay_valid = None
        common_prefix_turns = 0
        if replay_prefix is not None:
            common_prefix_turns = int(replay_prefix.get("replayed_calls") or 0)
            prefix_replay_valid = (
                int(replay_prefix.get("replay_mismatches") or 0) == 0
                and int(replay_prefix.get("replay_shortfalls") or 0) == 0
            )
        treatment_exposed = (
            isinstance(first_compaction_step, int)
            and not isinstance(first_compaction_step, bool)
        )
        full_trace = {step["step"]: step for step in full.get("progress_trace") or []}
        compact_trace = {
            step["step"]: step for step in compact.get("progress_trace") or []
        }
        trace_stop = max([*full_trace, *compact_trace], default=-1) + 1
        first_observable_divergence_step = _first_trace_divergence(
            full_trace,
            compact_trace,
            start=0,
            stop=trace_stop,
        )
        first_divergence_step = None
        first_post_exposure_divergence_step = None
        pre_exposure_trace_match = None
        if treatment_exposed:
            first_divergence_step = _first_trace_divergence(
                full_trace,
                compact_trace,
                start=0,
                stop=first_compaction_step,
            )
            pre_exposure_trace_match = first_divergence_step is None
            if pre_exposure_trace_match:
                first_post_exposure_divergence_step = _first_trace_divergence(
                    full_trace,
                    compact_trace,
                    start=first_compaction_step,
                    stop=trace_stop,
                )
        diagnostics.append(
            {
                "task_id": task_id,
                "repeat": repeat,
                "full_execution_order": full.get("execution_order"),
                "compact_execution_order": compact.get("execution_order"),
                "treatment_exposed": treatment_exposed,
                "first_compaction_step": first_compaction_step,
                "prefix_capture_arm": (
                    full["arm"]
                    if capture_prefix is full_prefix
                    else compact["arm"]
                    if capture_prefix is compact_prefix
                    else None
                ),
                "prefix_replay_arm": (
                    full["arm"]
                    if replay_prefix is full_prefix
                    else compact["arm"]
                    if replay_prefix is compact_prefix
                    else None
                ),
                "common_prefix_turns": common_prefix_turns,
                "prefix_replay_valid": prefix_replay_valid,
                "prefix_covers_pre_exposure": (
                    common_prefix_turns >= first_compaction_step
                    if treatment_exposed
                    else None
                ),
                "pre_exposure_trace_match": pre_exposure_trace_match,
                "first_divergence_step": first_divergence_step,
                "first_observable_divergence_step": first_observable_divergence_step,
                "first_post_exposure_divergence_step": (
                    first_post_exposure_divergence_step
                ),
                "contains_result_content": False,
                "contains_reasoning": False,
            }
        )
    return diagnostics


def _trace_signature(step: dict[str, Any] | None) -> tuple[Any, ...] | None:
    if step is None:
        return None
    return (
        step.get("operation"),
        step.get("target_kind"),
        step.get("target_fingerprint"),
        bool(step.get("workspace_effect")),
        step.get("verification_passed"),
        bool(step.get("error")),
    )


def _first_trace_divergence(
    full_trace: dict[int, dict[str, Any]],
    compact_trace: dict[int, dict[str, Any]],
    *,
    start: int,
    stop: int,
) -> int | None:
    for step_index in range(start, stop):
        if _trace_signature(full_trace.get(step_index)) != _trace_signature(
            compact_trace.get(step_index)
        ):
            return step_index
    return None


def _pair_quality(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    exposed = [item for item in diagnostics if item["treatment_exposed"]]
    aligned = [
        item
        for item in exposed
        if item["pre_exposure_trace_match"] is True
        and item["prefix_replay_valid"] is not False
    ]
    diverged = [item for item in exposed if item["pre_exposure_trace_match"] is False]
    prefix_invalid = [item for item in diagnostics if item["prefix_replay_valid"] is False]
    if not diagnostics:
        status = "no_matched_pairs"
    elif not exposed:
        status = "no_treatment_exposure"
    elif prefix_invalid:
        status = "prefix_replay_invalid"
    elif not aligned:
        status = "pre_exposure_diverged"
    elif diverged:
        status = "mixed_pre_exposure_alignment"
    else:
        status = "pre_exposure_aligned"
    return {
        "status": status,
        "matched_pairs": len(diagnostics),
        "treatment_exposed_pairs": len(exposed),
        "pre_exposure_aligned_pairs": len(aligned),
        "pre_exposure_diverged_pairs": len(diverged),
        "prefix_replayed_pairs": sum(item["common_prefix_turns"] > 0 for item in diagnostics),
        "prefix_replay_invalid_pairs": len(prefix_invalid),
        "prefix_covers_pre_exposure_pairs": sum(
            item["prefix_covers_pre_exposure"] is True for item in diagnostics
        ),
        "unexposed_pairs": len(diagnostics) - len(exposed),
        "alignment_uses_content_free_tool_trace": True,
    }


def _paired_delta(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [
        row[ARM_COMPACT][metric] - row[ARM_FULL][metric]
        for row in rows
        if row[ARM_COMPACT][metric] is not None and row[ARM_FULL][metric] is not None
    ]
    return sum(values) / len(values) if values else None


def _repeat_count(steps: list[dict[str, Any]]) -> int:
    return sum(not step["target_is_new"] for step in steps)


def _repeat_rate(steps: list[dict[str, Any]]) -> float | None:
    return _repeat_count(steps) / len(steps) if steps else None


def _category_tokens(attribution: dict[str, Any], category: str) -> int:
    categories = attribution.get("categories") or {}
    return int((categories.get(category) or {}).get("actual_input_tokens") or 0)


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


def _empty_lifecycle_metrics(mode: str, live_read_results: int) -> dict[str, Any]:
    return {
        "mode": mode,
        "enabled": mode == ARM_COMPACT,
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


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


__all__ = [
    "ARM_COMPACT",
    "ARM_FULL",
    "TOOL_RESULT_EVAL_ARMS",
    "render_tool_result_eval_summary",
    "run_tool_result_eval",
    "summarize_tool_result_records",
]
