"""Counterbalanced matched evaluation for epistemic transcript suppression."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

from brainregion.eval.stats import bootstrap_statistic, seed_for
from brainregion.runtime import normalize_usage

from .envs import RuleShiftEnv
from .epistemic_ledger import classify_epistemic_error
from .isolation import cleanup_run_dir, make_run_dir
from .loop import run_agent, scoped_env
from .prefix_replay import ModelPrefixTape, PrefixReplayBackend
from .task import SandboxTask

ARM_FULL = "full"
ARM_SUPPRESS = "suppress"
ARM_EVIDENCE = "evidence"
ARM_SELECTIVE = "selective"
RULE_SHIFT_ARMS = (ARM_FULL, ARM_SUPPRESS)
RULE_SHIFT_LIFECYCLE_ARMS = (
    ARM_FULL,
    ARM_SUPPRESS,
    ARM_EVIDENCE,
    ARM_SELECTIVE,
)
RULE_SHIFT_GOAL = (
    "Establish an evidence-supported action-effect rule, detect any later contradiction, "
    "and verify a replacement rule."
)

_METRICS = (
    "solved",
    "protocol_completed",
    "model_steps",
    "environment_actions",
    "environment_action_attempts",
    "distractor_actions",
    "error_steps",
    "input_tokens",
    "total_tokens",
    "cost_usd",
    "prediction_accuracy",
    "verified_insights",
    "supersessions",
)


async def run_rule_shift_case(
    backend: Any,
    model: str,
    *,
    endpoint_id: str | None = None,
    arm: str = ARM_FULL,
    shift_after: int = 3,
    distractor_steps: int = 0,
    max_steps: int = 10,
    max_cost_usd: float = 0.08,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    transcript_token_cap: int = 24000,
    consecutive_error_limit: int = 3,
    thinking: bool | None = False,
    effort: str | None = None,
    tool_result_lifecycle: str = "compact",
    tool_result_live_reads: int = 0,
    evidence_wake_live_reads: int = 2,
    evidence_max_selected_events: int = 4,
) -> dict[str, Any]:
    """Run one fresh rule-shift episode and return a content-free case record."""

    if arm not in RULE_SHIFT_LIFECYCLE_ARMS:
        raise ValueError(f"unknown rule-shift arm: {arm!r}")
    _validate_eval_args(
        repeats=1,
        shift_after=shift_after,
        distractor_steps=distractor_steps,
        max_steps=max_steps,
        max_cost_usd=max_cost_usd,
        max_total_cost_usd=None,
        shared_prefix_turns=0,
        bootstrap_samples=None,
        evidence_wake_live_reads=evidence_wake_live_reads,
        evidence_max_selected_events=evidence_max_selected_events,
        selective_wake_enabled=arm == ARM_SELECTIVE,
    )
    env = RuleShiftEnv(
        shift_after=shift_after,
        distractor_steps=distractor_steps,
    )
    task = SandboxTask(id="rule-shift", goal=RULE_SHIFT_GOAL)

    def verify(t, run_dir, *, python_exe=None):
        del t, run_dir, python_exe
        return {
            "tests_green": env.solved,
            "solve_status": "solved" if env.solved else "tests_fail",
            "pytest": None,
            "gold_diff": "",
        }

    run_dir = make_run_dir(prefix="brainregion-rule-shift-")
    try:
        with scoped_env(env):
            trajectory = await run_agent(
                backend,
                model,
                task,
                run_dir=run_dir,
                max_steps=max_steps,
                max_env_actions=max_steps,
                max_cost_usd=max_cost_usd,
                temperature=temperature,
                max_tokens=max_tokens,
                transcript_token_cap=transcript_token_cap,
                consecutive_error_limit=consecutive_error_limit,
                endpoint_id=endpoint_id,
                thinking=thinking,
                effort=effort,
                system_prompt=env.build_system_prompt(RULE_SHIFT_GOAL),
                verify_fn=verify,
                visual_ephemeral=True,
                tool_result_lifecycle=tool_result_lifecycle,
                tool_result_live_reads=tool_result_live_reads,
                epistemic_transcript_lifecycle=arm,
                epistemic_evidence_wake_live_reads=evidence_wake_live_reads,
                epistemic_evidence_max_selected_events=(
                    evidence_max_selected_events
                ),
                initial_observation=env.observation(),
            )
        return _case_from_trajectory(trajectory, env=env, arm=arm)
    finally:
        env.close()
        cleanup_run_dir(run_dir)


async def run_rule_shift_eval(
    backend: Any,
    model: str,
    *,
    endpoint_id: str | None = None,
    repeats: int = 2,
    shift_after: int = 3,
    distractor_steps: int = 0,
    max_steps: int = 10,
    max_cost_usd: float = 0.08,
    max_total_cost_usd: float | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    transcript_token_cap: int = 24000,
    consecutive_error_limit: int = 3,
    thinking: bool | None = False,
    effort: str | None = None,
    tool_result_lifecycle: str = "compact",
    tool_result_live_reads: int = 0,
    evidence_wake_live_reads: int = 2,
    evidence_max_selected_events: int = 4,
    shared_prefix_turns: int = 1,
    run_id: str = "",
    bootstrap_samples: int | None = None,
    arms: Sequence[str] = RULE_SHIFT_ARMS,
) -> dict[str, Any]:
    """Run a fresh lifecycle pair with alternating order and safe prefix replay."""

    arms = _normalize_arms(arms)
    _validate_eval_args(
        repeats=repeats,
        shift_after=shift_after,
        distractor_steps=distractor_steps,
        max_steps=max_steps,
        max_cost_usd=max_cost_usd,
        max_total_cost_usd=max_total_cost_usd,
        shared_prefix_turns=shared_prefix_turns,
        bootstrap_samples=bootstrap_samples,
        evidence_wake_live_reads=evidence_wake_live_reads,
        evidence_max_selected_events=evidence_max_selected_events,
        selective_wake_enabled=ARM_SELECTIVE in arms,
    )
    run_id = run_id or f"rule-shift-eval-{int(time.time() * 1000)}"
    cases: list[dict[str, Any]] = []
    order_counts = {f"{arm}_first": 0 for arm in arms}
    actual_provider_cost = 0.0
    actual_provider_calls = 0
    replayed_calls = 0
    execution_order = 0
    total_runs = repeats * len(arms)

    for repeat in range(repeats):
        ordered_arms = list(arms)
        if repeat % 2:
            ordered_arms.reverse()
        order_counts[f"{ordered_arms[0]}_first"] += 1
        tape = ModelPrefixTape(turn_limit=shared_prefix_turns)
        for arm_index, arm in enumerate(ordered_arms):
            role = "capture" if arm_index == 0 else "replay"
            run_backend = PrefixReplayBackend(backend, tape, role=role)
            execution_order += 1
            remaining_runs = total_runs - execution_order + 1
            run_budget = max_cost_usd
            if max_total_cost_usd is not None:
                remaining_cost = max(0.0, max_total_cost_usd - actual_provider_cost)
                run_budget = min(run_budget, remaining_cost / remaining_runs)
            try:
                if run_budget <= 0:
                    raise RuntimeError("experiment cost budget exhausted")
                case = await run_rule_shift_case(
                    run_backend,
                    model,
                    endpoint_id=endpoint_id,
                    arm=arm,
                    shift_after=shift_after,
                    distractor_steps=distractor_steps,
                    max_steps=max_steps,
                    max_cost_usd=run_budget,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    transcript_token_cap=transcript_token_cap,
                    consecutive_error_limit=consecutive_error_limit,
                    thinking=thinking,
                    effort=effort,
                    tool_result_lifecycle=tool_result_lifecycle,
                    tool_result_live_reads=tool_result_live_reads,
                    evidence_wake_live_reads=evidence_wake_live_reads,
                    evidence_max_selected_events=evidence_max_selected_events,
                )
            except Exception as exc:  # noqa: BLE001 - preserve the matched matrix
                case = _runner_error_case(arm=arm, error=exc)
            prefix_metrics = run_backend.public_metrics()
            actual_provider_cost += float(prefix_metrics["provider_cost_usd"])
            actual_provider_calls += int(prefix_metrics["provider_calls"])
            replayed_calls += int(prefix_metrics["replayed_calls"])
            case.update(
                {
                    "repeat": repeat,
                    "execution_order": execution_order,
                    "run_cost_budget_usd": run_budget,
                    "shared_prefix": prefix_metrics,
                }
            )
            cases.append(case)

    report = summarize_rule_shift_records(
        cases,
        run_id=run_id,
        bootstrap_samples=bootstrap_samples,
        arms=arms,
    )
    report["cases"] = cases
    report["execution"] = {
        "model": model,
        "endpoint_id": endpoint_id,
        "repeats": repeats,
        "arms": list(arms),
        "shift_after": shift_after,
        "distractor_steps": distractor_steps,
        "max_steps": max_steps,
        "max_cost_usd_per_run": max_cost_usd,
        "max_total_cost_usd": max_total_cost_usd,
        "actual_provider_cost_usd": round(actual_provider_cost, 6),
        "actual_provider_calls": actual_provider_calls,
        "replayed_model_calls": replayed_calls,
        "thinking": thinking,
        "effort": effort if thinking else None,
        "tool_result_lifecycle": tool_result_lifecycle,
        "tool_result_live_reads": tool_result_live_reads,
        "evidence_wake_live_reads": evidence_wake_live_reads,
        "evidence_max_selected_events": evidence_max_selected_events,
        "shared_prefix_turns": shared_prefix_turns,
        "shared_prefix_policy": "exact_first_response_before_suppression_v1",
        "arm_order_policy": "alternating_by_repeat",
        "arm_order_counts": order_counts,
        "counterbalanced_order": all(value > 0 for value in order_counts.values()),
        "contains_request_content": False,
        "contains_response_content": False,
        "contains_reasoning": False,
    }
    return report


def summarize_rule_shift_records(
    records: list[dict[str, Any]],
    *,
    run_id: str = "",
    bootstrap_samples: int | None = None,
    arms: Sequence[str] = RULE_SHIFT_ARMS,
) -> dict[str, Any]:
    if not records:
        raise ValueError("rule-shift records cannot be empty")
    arms = _normalize_arms(arms)
    grouped: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    for record in records:
        arm = str(record.get("arm") or "")
        if arm not in grouped:
            raise ValueError(f"unknown rule-shift arm: {arm!r}")
        grouped[arm].append(record)

    per_arm = {
        arm: _arm_summary(grouped[arm])
        for arm in arms
        if grouped[arm]
    }
    diagnostics = _pair_diagnostics(records, arms=arms)
    matched_rows = _matched_rows(records, arms=arms)
    exposed_repeats = {
        item["repeat"]
        for item in diagnostics
        if item["contrast_exposed"]
        and item["prefix_replay_valid"] is not False
        and item["first_action_match"] is True
    }
    exposed_rows = [row for row in matched_rows if row["repeat"] in exposed_repeats]
    run_id = run_id or f"rule-shift-summary-{int(time.time() * 1000)}"
    return {
        "run_id": run_id,
        "arms": list(arms),
        "n_runs": len(records),
        "bootstrap_unit": "repeat",
        "per_arm": per_arm,
        "matched_effect": _effect_summary(
            matched_rows,
            run_id=run_id,
            label="all",
            bootstrap_samples=bootstrap_samples,
            arms=arms,
        ),
        "exposure_aligned_effect": _effect_summary(
            exposed_rows,
            run_id=run_id,
            label="exposed",
            bootstrap_samples=bootstrap_samples,
            arms=arms,
        ),
        "pair_quality": _pair_quality(diagnostics),
        "pair_diagnostics": diagnostics,
        "interpretation": "descriptive_pilot_no_ability_claim",
        "contains_rule_content": False,
        "contains_frame_content": False,
        "contains_reasoning": False,
    }


def render_rule_shift_eval_summary(report: dict[str, Any]) -> str:
    execution = report.get("execution") or {}
    lines = [
        f"### rule-shift matched eval {report['run_id']} "
        f"(runs={report['n_runs']}, repeats={execution.get('repeats')})",
        f"model={execution.get('model')} counterbalanced={execution.get('counterbalanced_order')} "
        f"provider_calls={execution.get('actual_provider_calls')} "
        f"replayed={execution.get('replayed_model_calls')} "
        f"actual_cost=${float(execution.get('actual_provider_cost_usd') or 0):.6f}",
    ]
    for arm, summary in (report.get("per_arm") or {}).items():
        lines.append(
            f"  {arm}: valid={summary.get('n_valid_runs')}/{summary.get('n_runs')} "
            f"solve={summary.get('solve_rate')} steps={summary.get('mean_model_steps')} "
            f"tokens={summary.get('mean_total_tokens')} "
            f"cost=${float(summary.get('mean_cost_usd') or 0):.6f} "
            f"suppressed={summary.get('mean_suppressed_turns')} "
            f"evidence_receipts={summary.get('mean_evidence_receipts')} "
            f"workspace_events={summary.get('mean_evidence_workspace_events')} "
            f"workspace_injections={summary.get('mean_workspace_injections')} "
            f"wake_requests={summary.get('mean_wake_requests')} "
            f"attention_selected={summary.get('mean_attention_selected_events')} "
            f"attention_omitted={summary.get('mean_attention_omitted_events')}"
        )
    effect = report.get("matched_effect") or {}
    raw = effect.get("raw_deltas") or {}
    token_ci = (effect.get("bootstrap_deltas") or {}).get("total_tokens") or {}
    lines.append(
        f"  {effect.get('delta_direction')}: "
        f"solve={raw.get('solved')} steps={raw.get('model_steps')} "
        f"tokens={raw.get('total_tokens')} cost={raw.get('cost_usd')} "
        f"pairs={effect.get('n_pairs')}"
    )
    lines.append(
        f"  token_delta_ci=[{token_ci.get('low')}, {token_ci.get('high')}] "
        f"bootstrap_n={token_ci.get('n')}"
    )
    quality = report.get("pair_quality") or {}
    lines.append(
        f"  pair_quality={quality.get('status')} complete={quality.get('complete_pairs')} "
        f"exposed={quality.get('contrast_exposed_pairs')} "
        f"prefix_invalid={quality.get('prefix_replay_invalid_pairs')}"
    )
    lines.append("  interpretation=descriptive pilot; no ability claim")
    return "\n".join(lines)


def _case_from_trajectory(trajectory: Any, *, env: RuleShiftEnv, arm: str) -> dict[str, Any]:
    usage = normalize_usage(trajectory.total_main_usage)
    progress = trajectory.progress_trace
    operation_counts: dict[str, int] = {}
    error_kind_counts: dict[str, int] = {}
    for item in progress:
        operation = str(item.get("operation") or "model_turn")
        operation_counts[operation] = operation_counts.get(operation, 0) + 1
        error_kind = str(item.get("error_kind") or "")
        if error_kind:
            error_kind_counts[error_kind] = error_kind_counts.get(error_kind, 0) + 1
    tool_error_code_counts: dict[str, int] = {}
    for step in trajectory.steps:
        if step.error_kind != "tool_error":
            continue
        code = classify_epistemic_error(step.error)
        if code:
            tool_error_code_counts[code] = tool_error_code_counts.get(code, 0) + 1
    ledger = env.epistemic_ledger.public_metrics()
    lifecycle = trajectory.epistemic_transcript_lifecycle
    model_error_events = error_kind_counts.get("model_error", 0)
    return {
        "arm": arm,
        "solved": env.solved,
        "protocol_completed": trajectory.done,
        "state": env.snapshot().get("state"),
        "termination_reason": trajectory.termination_reason,
        "infrastructure_error": (
            trajectory.termination_reason == "model_error" or model_error_events > 0
        ),
        "infrastructure_degraded": model_error_events > 0,
        "model_steps": trajectory.n_steps,
        "environment_actions": len(env.action_trace),
        "distractor_actions": sum(
            item.get("action") == "action2" for item in env.action_trace
        ),
        "delayed_recall_exposed": _delayed_recall_exposed(env.action_trace),
        "environment_action_attempts": operation_counts.get("act", 0),
        "operation_counts": operation_counts,
        "error_steps": sum(bool(item.get("error")) for item in progress),
        "error_kind_counts": error_kind_counts,
        "tool_error_code_counts": tool_error_code_counts,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "cached_tokens": usage["cached_tokens"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "cost_usd": float(trajectory.total_main_cost_usd),
        "prediction_accuracy": ledger["prediction_accuracy"],
        "verified_insights": ledger["verified_insights"],
        "supersessions": ledger["supersessions"],
        "epistemic_ledger": ledger,
        "epistemic_transcript_lifecycle": lifecycle,
        "tool_result_lifecycle": trajectory.tool_result_lifecycle,
        "main_input_attribution": trajectory.main_input_attribution,
        "interaction_trace": list(env.action_trace),
        "contains_rule_content": False,
        "contains_frame_content": False,
        "contains_reasoning": False,
    }


def _runner_error_case(*, arm: str, error: Exception) -> dict[str, Any]:
    return {
        "arm": arm,
        "solved": False,
        "protocol_completed": False,
        "state": "UNKNOWN",
        "termination_reason": "runner_error",
        "infrastructure_error": True,
        "infrastructure_degraded": True,
        "model_steps": 0,
        "environment_actions": 0,
        "environment_action_attempts": 0,
        "distractor_actions": 0,
        "delayed_recall_exposed": False,
        "operation_counts": {},
        "error_steps": 0,
        "error_kind_counts": {"runner_error": 1},
        "tool_error_code_counts": {},
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
        "prediction_accuracy": None,
        "verified_insights": 0,
        "supersessions": 0,
        "epistemic_ledger": {},
        "epistemic_transcript_lifecycle": {},
        "tool_result_lifecycle": {},
        "main_input_attribution": {},
        "interaction_trace": [],
        "error_type": type(error).__name__,
        "contains_rule_content": False,
        "contains_frame_content": False,
        "contains_reasoning": False,
    }


def _normalize_arms(arms: Sequence[str]) -> tuple[str, str]:
    values = tuple(str(arm).strip().lower() for arm in arms)
    if len(values) != 2:
        raise ValueError("rule-shift eval requires exactly two arms")
    if values[0] == values[1]:
        raise ValueError("rule-shift eval arms must be distinct")
    unknown = [arm for arm in values if arm not in RULE_SHIFT_LIFECYCLE_ARMS]
    if unknown:
        raise ValueError(f"unknown rule-shift arm: {unknown[0]!r}")
    return values[0], values[1]


def _validate_eval_args(
    *,
    repeats: int,
    shift_after: int,
    distractor_steps: int,
    max_steps: int,
    max_cost_usd: float,
    max_total_cost_usd: float | None,
    shared_prefix_turns: int,
    bootstrap_samples: int | None,
    evidence_wake_live_reads: int,
    evidence_max_selected_events: int,
    selective_wake_enabled: bool,
) -> None:
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    if isinstance(shift_after, bool) or not isinstance(shift_after, int) or shift_after < 2:
        raise ValueError("shift_after must be an integer of at least 2")
    if (
        isinstance(distractor_steps, bool)
        or not isinstance(distractor_steps, int)
        or distractor_steps < 0
    ):
        raise ValueError("distractor_steps must be a non-negative integer")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if not math.isfinite(max_cost_usd) or max_cost_usd <= 0:
        raise ValueError("max_cost_usd must be positive")
    if max_total_cost_usd is not None and (
        not math.isfinite(max_total_cost_usd) or max_total_cost_usd <= 0
    ):
        raise ValueError("max_total_cost_usd must be positive when provided")
    if shared_prefix_turns not in {0, 1}:
        raise ValueError("shared_prefix_turns must be 0 or 1")
    if bootstrap_samples is not None and (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples <= 0
    ):
        raise ValueError("bootstrap_samples must be a positive integer when provided")
    if selective_wake_enabled and (
        isinstance(evidence_wake_live_reads, bool)
        or not isinstance(evidence_wake_live_reads, int)
        or evidence_wake_live_reads <= 0
    ):
        raise ValueError("evidence_wake_live_reads must be a positive integer")
    if selective_wake_enabled and (
        isinstance(evidence_max_selected_events, bool)
        or not isinstance(evidence_max_selected_events, int)
        or evidence_max_selected_events <= 0
    ):
        raise ValueError("evidence_max_selected_events must be a positive integer")


def _arm_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if not record.get("infrastructure_error")]
    lifecycles = [record.get("epistemic_transcript_lifecycle") or {} for record in valid]
    evidence_workspaces = [
        lifecycle.get("evidence_workspace") or {} for lifecycle in lifecycles
    ]
    selective_wakes = [lifecycle.get("selective_wake") or {} for lifecycle in lifecycles]
    event_attentions = [lifecycle.get("event_attention") or {} for lifecycle in lifecycles]
    return {
        "n_runs": len(records),
        "n_valid_runs": len(valid),
        "solve_rate": _mean(valid, "solved"),
        "protocol_completion_rate": _mean(valid, "protocol_completed"),
        "mean_model_steps": _mean(valid, "model_steps"),
        "mean_environment_actions": _mean(valid, "environment_actions"),
        "mean_distractor_actions": _mean(valid, "distractor_actions"),
        "delayed_recall_exposure_rate": _mean(valid, "delayed_recall_exposed"),
        "mean_error_steps": _mean(valid, "error_steps"),
        "mean_input_tokens": _mean(valid, "input_tokens"),
        "mean_total_tokens": _mean(valid, "total_tokens"),
        "mean_cost_usd": _mean(valid, "cost_usd"),
        "mean_prediction_accuracy": _mean(valid, "prediction_accuracy"),
        "mean_verified_insights": _mean(valid, "verified_insights"),
        "mean_supersessions": _mean(valid, "supersessions"),
        "mean_suppressed_turns": _mean(lifecycles, "suppressed_turns"),
        "mean_evidence_receipts": _mean(lifecycles, "evidence_receipts"),
        "mean_evidence_workspace_events": _mean(evidence_workspaces, "events"),
        "mean_evidence_workspace_observations": _mean(
            evidence_workspaces, "observations"
        ),
        "mean_evidence_workspace_deduplicated": _mean(
            evidence_workspaces, "deduplicated_observations"
        ),
        "mean_workspace_injections": _mean(lifecycles, "workspace_refreshes"),
        "mean_workspace_skips": _mean(lifecycles, "workspace_skips"),
        "mean_workspace_estimated_tokens_injected": _mean(
            lifecycles, "workspace_estimated_tokens_injected"
        ),
        "mean_wake_requests": _mean(selective_wakes, "requests"),
        "mean_wake_activations": _mean(selective_wakes, "activations"),
        "mean_attention_selection_passes": _mean(
            event_attentions, "selection_passes"
        ),
        "mean_attention_selected_events": _mean(
            event_attentions, "selected_events"
        ),
        "mean_attention_omitted_events": _mean(
            event_attentions, "omitted_events"
        ),
        "mean_attention_empty_wakes": _mean(event_attentions, "empty_wakes"),
        "mean_estimated_input_tokens_avoided": _mean(
            lifecycles, "estimated_input_tokens_avoided"
        ),
        "infrastructure_failures": len(records) - len(valid),
    }


def _matched_rows(
    records: list[dict[str, Any]],
    *,
    arms: tuple[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["repeat"]), {})[record["arm"]] = record
    rows = []
    for repeat, pair in sorted(grouped.items()):
        if not all(arm in pair for arm in arms):
            continue
        if any(pair[arm].get("infrastructure_error") for arm in arms):
            continue
        rows.append({"repeat": repeat, **pair})
    return rows


def _pair_diagnostics(
    records: list[dict[str, Any]],
    *,
    arms: tuple[str, str],
) -> list[dict[str, Any]]:
    control_arm, treatment_arm = arms
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["repeat"]), {})[record["arm"]] = record
    diagnostics = []
    for repeat, pair in sorted(grouped.items()):
        complete = all(arm in pair for arm in arms)
        if not complete:
            continue
        control = pair[control_arm]
        treatment = pair[treatment_arm]
        prefixes = [
            control.get("shared_prefix") or {},
            treatment.get("shared_prefix") or {},
        ]
        replay = next((item for item in prefixes if item.get("role") == "replay"), {})
        prefix_valid = (
            int(replay.get("replay_mismatches") or 0) == 0
            and int(replay.get("replay_shortfalls") or 0) == 0
        )
        control_first = _first_action_signature(control)
        treatment_first = _first_action_signature(treatment)
        control_exposed = int(
            (control.get("epistemic_transcript_lifecycle") or {}).get(
                "suppressed_turns", 0
            )
        ) > 0
        treatment_exposed = int(
            (treatment.get("epistemic_transcript_lifecycle") or {}).get(
                "suppressed_turns", 0
            )
        ) > 0
        diagnostics.append(
            {
                "repeat": repeat,
                "complete": True,
                "infrastructure_error": bool(
                    control.get("infrastructure_error")
                    or treatment.get("infrastructure_error")
                ),
                "control": control_arm,
                "treatment": treatment_arm,
                "control_execution_order": control.get("execution_order"),
                "treatment_execution_order": treatment.get("execution_order"),
                "prefix_replay_valid": prefix_valid,
                "first_action_match": (
                    control_first == treatment_first
                    if control_first is not None and treatment_first is not None
                    else None
                ),
                "control_exposed": control_exposed,
                "treatment_exposed": treatment_exposed,
                "contrast_exposed": control_exposed or treatment_exposed,
                "contains_rule_content": False,
                "contains_reasoning": False,
            }
        )
    return diagnostics


def _first_action_signature(record: dict[str, Any]) -> tuple[Any, ...] | None:
    trace = record.get("interaction_trace") or []
    if not trace:
        return None
    first = trace[0]
    return (
        first.get("action"),
        first.get("change_scale"),
        first.get("epistemic_hypothesis_fingerprint"),
        first.get("epistemic_status"),
        tuple(first.get("epistemic_mismatch_fields") or []),
    )


def _delayed_recall_exposed(trace: list[dict[str, Any]]) -> bool:
    for index, event in enumerate(trace):
        if (
            event.get("action") != "action1"
            or event.get("change_scale") != "global"
            or event.get("epistemic_status") != "refuted"
        ):
            continue
        tail = trace[index + 1 :]
        action2_indexes = [
            tail_index
            for tail_index, tail_event in enumerate(tail)
            if tail_event.get("action") == "action2"
        ]
        if not action2_indexes:
            continue
        return any(
            tail_event.get("action") == "action1"
            for tail_event in tail[max(action2_indexes) + 1 :]
        )
    return False


def _pair_quality(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in diagnostics if not item["infrastructure_error"]]
    prefix_invalid = [item for item in valid if item["prefix_replay_valid"] is False]
    first_diverged = [item for item in valid if item["first_action_match"] is False]
    if not diagnostics:
        status = "no_complete_pairs"
    elif not valid:
        status = "infrastructure_failure"
    elif prefix_invalid:
        status = "prefix_replay_invalid"
    elif first_diverged:
        status = "first_action_diverged"
    elif len(valid) < len(diagnostics):
        status = "matched_with_infrastructure_failures"
    else:
        status = "matched"
    return {
        "status": status,
        "complete_pairs": len(diagnostics),
        "valid_pairs": len(valid),
        "treatment_exposed_pairs": sum(item["treatment_exposed"] for item in valid),
        "contrast_exposed_pairs": sum(item["contrast_exposed"] for item in valid),
        "prefix_replay_invalid_pairs": len(prefix_invalid),
        "first_action_diverged_pairs": len(first_diverged),
        "infrastructure_failed_pairs": len(diagnostics) - len(valid),
    }


def _effect_summary(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    label: str,
    bootstrap_samples: int | None,
    arms: tuple[str, str],
) -> dict[str, Any]:
    control_arm, treatment_arm = arms
    direction = f"{treatment_arm}_minus_{control_arm}"
    return {
        "control": control_arm,
        "treatment": treatment_arm,
        "delta_direction": direction,
        "n_pairs": len(rows),
        "raw_deltas": {
            metric: _paired_delta(rows, metric, arms=arms) for metric in _METRICS
        },
        "bootstrap_deltas": {
            metric: bootstrap_statistic(
                rows,
                lambda sample, metric=metric: _paired_delta(sample, metric, arms=arms),
                B=bootstrap_samples,
                seed=seed_for(run_id, f"rule-shift:{direction}:{label}:{metric}"),
            )
            for metric in _METRICS
        },
    }


def _paired_delta(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    arms: tuple[str, str],
) -> float | None:
    control_arm, treatment_arm = arms
    values = [
        float(row[treatment_arm][metric]) - float(row[control_arm][metric])
        for row in rows
        if row[treatment_arm].get(metric) is not None
        and row[control_arm].get(metric) is not None
    ]
    return sum(values) / len(values) if values else None


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(record[key])
        for record in records
        if record.get(key) is not None and math.isfinite(float(record[key]))
    ]
    return sum(values) / len(values) if values else None


__all__ = [
    "ARM_EVIDENCE",
    "ARM_FULL",
    "ARM_SELECTIVE",
    "ARM_SUPPRESS",
    "RULE_SHIFT_ARMS",
    "RULE_SHIFT_LIFECYCLE_ARMS",
    "RULE_SHIFT_GOAL",
    "render_rule_shift_eval_summary",
    "run_rule_shift_case",
    "run_rule_shift_eval",
    "summarize_rule_shift_records",
]
