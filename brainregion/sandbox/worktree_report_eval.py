"""Matched real-worktree evaluation for RegionReport delivery formats."""

from __future__ import annotations

import sys
import time
from typing import Any

from brainregion.runtime import normalize_usage

from .delegation_eval import render_expert_decision_cards, render_expert_reports
from .loop import run_agent
from .task import WorktreeTask
from .worktree import (
    bootstrap_worktree,
    capture_worktree_diff,
    detect_venv_python,
    worktree,
)
from .worktree_memory_eval import (
    WorktreeMemoryExpertSpec,
    _positive_int,
    _protected_path_integrity,
    _protected_path_snapshot,
    _run_expert,
    _trajectory_diagnostics,
)

ARM_NO_REPORT = "no_report"
ARM_FULL_REPORT = "full_report"
ARM_DECISION_CARD = "decision_card"
WORKTREE_REPORT_ARMS = (ARM_NO_REPORT, ARM_FULL_REPORT, ARM_DECISION_CARD)


def normalize_worktree_report_arms(
    arms: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    selected = tuple(arms or WORKTREE_REPORT_ARMS)
    if not selected:
        raise ValueError("worktree report arms cannot be empty")
    if len(selected) != len(set(selected)):
        raise ValueError("worktree report arms must be unique")
    unknown = [arm for arm in selected if arm not in WORKTREE_REPORT_ARMS]
    if unknown:
        raise ValueError(f"unknown worktree report arm(s): {unknown}")
    return selected


def _mean(items: list[dict[str, Any]], field: str) -> float:
    return round(sum(float(item.get(field, 0)) for item in items) / len(items), 4)


def _comparison(
    per_arm: dict[str, dict[str, Any]], treatment: str, control: str
) -> dict[str, float]:
    left = per_arm[treatment]
    right = per_arm[control]
    return {
        "solve_rate_delta": round(left["solve_rate"] - right["solve_rate"], 4),
        "main_steps_delta": round(left["mean_main_steps"] - right["mean_main_steps"], 4),
        "workspace_effects_delta": round(
            left["mean_workspace_effects"] - right["mean_workspace_effects"], 4
        ),
        "parse_errors_delta": round(
            left["mean_main_parse_errors"] - right["mean_main_parse_errors"], 4
        ),
        "saturated_output_calls_delta": round(
            left["mean_saturated_output_calls"]
            - right["mean_saturated_output_calls"],
            4,
        ),
        "main_input_tokens_delta": round(
            left["mean_main_input_tokens"] - right["mean_main_input_tokens"], 4
        ),
        "main_cached_tokens_delta": round(
            left["mean_main_cached_tokens"] - right["mean_main_cached_tokens"], 4
        ),
        "main_reasoning_tokens_delta": round(
            left["mean_main_reasoning_tokens"]
            - right["mean_main_reasoning_tokens"],
            4,
        ),
        "main_cost_usd_delta": round(
            left["mean_main_cost_usd"] - right["mean_main_cost_usd"], 6
        ),
    }


def summarize_worktree_report_records(
    records: list[dict[str, Any]],
    *,
    run_id: str = "",
    arms: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("worktree report records cannot be empty")
    selected_arms = normalize_worktree_report_arms(arms)
    grouped = {arm: [] for arm in selected_arms}
    pairs: dict[int, set[str]] = {}
    for record in records:
        arm = str(record.get("arm") or "")
        if arm not in WORKTREE_REPORT_ARMS:
            raise ValueError(f"unknown worktree report arm: {arm!r}")
        if arm not in grouped:
            raise ValueError(f"worktree report arm not selected: {arm!r}")
        repeat = int(record.get("repeat", 0))
        if arm in pairs.setdefault(repeat, set()):
            raise ValueError(f"duplicate worktree report record: {repeat}/{arm}")
        pairs[repeat].add(arm)
        grouped[arm].append(record)
    incomplete = [
        repeat for repeat, repeat_arms in pairs.items() if repeat_arms != set(selected_arms)
    ]
    if incomplete:
        raise ValueError(f"incomplete worktree report repeats: {incomplete}")

    per_arm: dict[str, dict[str, Any]] = {}
    for arm, items in grouped.items():
        runs = len(items)
        per_arm[arm] = {
            "runs": runs,
            "solved": sum(int(bool(item.get("solved"))) for item in items),
            "solve_rate": round(
                sum(int(bool(item.get("solved"))) for item in items) / runs, 4
            ),
            "mean_main_steps": _mean(items, "main_steps"),
            "mean_workspace_effects": _mean(items, "workspace_effects"),
            "mean_verification_runs": _mean(items, "verification_runs"),
            "mean_main_input_tokens": _mean(items, "main_input_tokens"),
            "mean_main_output_tokens": _mean(items, "main_output_tokens"),
            "mean_main_cached_tokens": _mean(items, "main_cached_tokens"),
            "mean_main_reasoning_tokens": _mean(items, "main_reasoning_tokens"),
            "mean_main_total_tokens": _mean(items, "main_total_tokens"),
            "mean_main_cost_usd": _mean(items, "main_cost_usd"),
            "mean_advisory_chars": _mean(items, "advisory_chars"),
            "mean_main_parse_errors": round(
                sum(
                    int(
                        ((item.get("main_diagnostics") or {}).get("error_kind_counts") or {}).get(
                            "parse_error", 0
                        )
                    )
                    for item in items
                )
                / runs,
                4,
            ),
            "mean_saturated_output_calls": round(
                sum(
                    int((item.get("main_diagnostics") or {}).get("saturated_output_calls", 0))
                    for item in items
                )
                / runs,
                4,
            ),
            "expert_report_adoption_rate": round(
                sum(int(bool(item.get("adopted_expert_report"))) for item in items)
                / runs,
                4,
            ),
        }
    comparisons: dict[str, dict[str, float]] = {}
    if {ARM_NO_REPORT, ARM_FULL_REPORT} <= set(selected_arms):
        comparisons["full_report_minus_no_report"] = _comparison(
            per_arm, ARM_FULL_REPORT, ARM_NO_REPORT
        )
    if {ARM_FULL_REPORT, ARM_DECISION_CARD} <= set(selected_arms):
        comparisons["decision_card_minus_full_report"] = _comparison(
            per_arm, ARM_DECISION_CARD, ARM_FULL_REPORT
        )
    if {ARM_NO_REPORT, ARM_DECISION_CARD} <= set(selected_arms):
        comparisons["decision_card_minus_no_report"] = _comparison(
            per_arm, ARM_DECISION_CARD, ARM_NO_REPORT
        )
    return {
        "run_id": run_id,
        "mode": "real_worktree_region_report_utilization",
        "pair_count": len(pairs),
        "arms": list(selected_arms),
        "per_arm": per_arm,
        "comparisons": comparisons,
        "infrastructure_usable": not any(
            bool(item.get("infrastructure_error")) for item in records
        ),
        "contains_source_content": False,
        "contains_memory_content": False,
        "contains_report_content": False,
        "contains_diff_content": False,
        "contains_reasoning": False,
    }


async def run_worktree_report_utilization_eval(
    backend: Any,
    main_model: str,
    task: WorktreeTask,
    expert: WorktreeMemoryExpertSpec,
    *,
    main_endpoint_id: str | None = None,
    repeats: int = 1,
    max_steps: int = 10,
    max_cost_usd: float = 0.5,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    transcript_token_cap: int = 24000,
    consecutive_error_limit: int = 3,
    thinking: bool | None = None,
    effort: str | None = None,
    expert_max_context_tokens: int = 6000,
    expert_max_tokens: int = 1200,
    expert_temperature: float = 0.0,
    python_exe: str | None = None,
    arms: list[str] | tuple[str, ...] | None = None,
    run_id: str = "",
) -> dict[str, Any]:
    """Compare no report, full report, and a compact card from one shared report."""

    repeats = _positive_int(repeats, "repeats")
    max_steps = _positive_int(max_steps, "max_steps")
    expert_max_context_tokens = _positive_int(
        expert_max_context_tokens, "expert_max_context_tokens"
    )
    if not str(main_model or "").strip():
        raise ValueError("main_model cannot be empty")
    selected_arms = normalize_worktree_report_arms(arms)
    needs_expert_report = any(arm != ARM_NO_REPORT for arm in selected_arms)
    if not task.protected_paths:
        raise ValueError("worktree report eval requires protected_paths")
    if needs_expert_report and not task.expert_context_paths:
        raise ValueError("report delivery arms require expert_context_paths")
    if needs_expert_report and not task.seed_memory:
        raise ValueError("worktree report eval requires seed_memory")

    run_id = run_id or f"worktree-report-eval-{int(time.time() * 1000)}"
    records: list[dict[str, Any]] = []
    expert_runs: list[dict[str, Any]] = []
    order_counts = {f"{arm}_first": 0 for arm in selected_arms}
    execution_order = 0

    for repeat in range(repeats):
        report: dict[str, Any] | None = None
        if needs_expert_report:
            with worktree(task.repo_path, task.base_ref) as expert_handle:
                bootstrap = bootstrap_worktree(expert_handle, task.bootstrap_commands)
                expert_bootstrap_failures = sum(
                    int(item["returncode"] != 0) for item in bootstrap
                )
                report, expert_metrics = await _run_expert(
                    backend,
                    task,
                    expert_handle.path,
                    expert,
                    include_memory=True,
                    max_context_tokens=expert_max_context_tokens,
                    max_tokens=expert_max_tokens,
                    temperature=expert_temperature,
                    effort=effort,
                )
            expert_runs.append(
                {
                    **expert_metrics,
                    "repeat": repeat,
                    "bootstrap_failure_count": expert_bootstrap_failures,
                }
            )
            if report is None:
                raise ValueError(
                    f"shared expert report was not produced for repeat {repeat}; "
                    "report-utilization arms cannot be matched"
                )

        advisories = {ARM_NO_REPORT: ""}
        if report is not None:
            advisories[ARM_FULL_REPORT] = render_expert_reports((report,))
            advisories[ARM_DECISION_CARD] = render_expert_decision_cards((report,))
        offset = repeat % len(selected_arms)
        ordered_arms = (*selected_arms[offset:], *selected_arms[:offset])
        order_counts[f"{ordered_arms[0]}_first"] += 1

        for arm in ordered_arms:
            execution_order += 1
            advisory = advisories[arm]
            with worktree(task.repo_path, task.base_ref) as handle:
                bootstrap = bootstrap_worktree(handle, task.bootstrap_commands)
                bootstrap_failures = sum(
                    int(item["returncode"] != 0) for item in bootstrap
                )
                protected_baseline = _protected_path_snapshot(
                    handle.path, task.protected_paths
                )
                selected_python = (
                    python_exe or detect_venv_python(handle.path) or sys.executable
                )
                trajectory = await run_agent(
                    backend,
                    main_model,
                    task,
                    run_dir=handle.path,
                    arm="none",
                    max_steps=max_steps,
                    max_cost_usd=max_cost_usd,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    transcript_token_cap=transcript_token_cap,
                    consecutive_error_limit=consecutive_error_limit,
                    endpoint_id=main_endpoint_id,
                    thinking=thinking,
                    effort=effort,
                    advisory_context=advisory,
                    python_exe=selected_python,
                )
                protected_unchanged, protected_modified_count = (
                    _protected_path_integrity(handle.path, protected_baseline)
                )
                diff = capture_worktree_diff(handle)

            usage = normalize_usage(trajectory.total_main_usage)
            records.append(
                {
                    "arm": arm,
                    "repeat": repeat,
                    "execution_order": execution_order,
                    "solved": bool(trajectory.tests_green and protected_unchanged),
                    "verification_tests_green": trajectory.tests_green,
                    "evaluation_failure_reason": (
                        None if protected_unchanged else "protected_paths_modified"
                    ),
                    "protocol_completed": trajectory.done,
                    "termination_reason": trajectory.termination_reason,
                    "infrastructure_error": trajectory.termination_reason == "model_error",
                    "main_steps": trajectory.n_steps,
                    "workspace_effects": trajectory.workspace_effects,
                    "verification_runs": trajectory.verification_runs,
                    "last_verification_passed": trajectory.last_verification_passed,
                    "adopted_expert_report": bool(trajectory.adopted_assignment_ids),
                    "advisory_chars": len(advisory),
                    "main_input_tokens": usage["input_tokens"],
                    "main_output_tokens": usage["output_tokens"],
                    "main_cached_tokens": usage["cached_tokens"],
                    "main_reasoning_tokens": usage["reasoning_tokens"],
                    "main_total_tokens": usage["total_tokens"],
                    "main_cost_usd": round(float(trajectory.total_main_cost_usd), 6),
                    "main_diagnostics": _trajectory_diagnostics(
                        trajectory, max_tokens=max_tokens
                    ),
                    "bootstrap_failure_count": bootstrap_failures,
                    "protected_paths_count": len(protected_baseline),
                    "protected_paths_unchanged": protected_unchanged,
                    "protected_paths_modified_count": protected_modified_count,
                    "diff_changed": bool(str(diff.get("diff_stat") or "").strip()),
                    "shared_report_semantics": arm != ARM_NO_REPORT,
                    "contains_advisory_content": False,
                    "contains_diff_content": False,
                    "contains_tool_results": False,
                    "contains_reasoning": False,
                }
            )

    summary = summarize_worktree_report_records(
        records, run_id=run_id, arms=selected_arms
    )
    expert_cost = round(sum(float(item.get("cost_usd", 0.0)) for item in expert_runs), 6)
    main_cost = round(sum(float(item.get("main_cost_usd", 0.0)) for item in records), 6)
    summary.update(
        {
            "records": records,
            "expert_generation": {
                "runs": len(expert_runs),
                "skipped": not needs_expert_report,
                "report_rate": (
                    round(
                        sum(int(bool(item.get("report_produced"))) for item in expert_runs)
                        / len(expert_runs),
                        4,
                    )
                    if expert_runs
                    else 0.0
                ),
                "mean_memory_blocks": (
                    round(
                        sum(int(item.get("memory_blocks", 0)) for item in expert_runs)
                        / len(expert_runs),
                        4,
                    )
                    if expert_runs
                    else 0.0
                ),
                "memory_records_excluded_by_scope": sum(
                    int(item.get("memory_records_excluded_by_scope", 0))
                    for item in expert_runs
                ),
                "total_tokens": sum(int(item.get("total_tokens", 0)) for item in expert_runs),
                "cost_usd": expert_cost,
                "contains_report_content": False,
                "contains_memory_content": False,
                "contains_reasoning": False,
            },
            "total_cost_usd": round(expert_cost + main_cost, 6),
            "execution": {
                "runner": "isolated_git_worktree",
                "task_id": task.id,
                "base_ref": task.base_ref,
                "main_model": main_model,
                "main_endpoint_id": main_endpoint_id,
                "expert_model": expert.model,
                "expert_endpoint_id": expert.endpoint_id,
                "expert_region": expert.region,
                "repeats": repeats,
                "arms": list(selected_arms),
                "expert_report_calls_per_repeat": 1 if needs_expert_report else 0,
                "report_semantics_reused_across_delivery_arms": needs_expert_report,
                "arm_order_policy": "rotating_by_repeat",
                "arm_order_counts": order_counts,
                "protected_path_integrity_enforced": True,
                "contains_repo_path": False,
                "contains_source_content": False,
                "contains_memory_content": False,
                "contains_report_content": False,
                "contains_diff_content": False,
                "contains_reasoning": False,
            },
        }
    )
    return summary


def render_worktree_report_summary(report: dict[str, Any]) -> str:
    lines = [f"Worktree RegionReport utilization: {report.get('run_id', '')}"]
    for arm in report.get("arms") or WORKTREE_REPORT_ARMS:
        item = (report.get("per_arm") or {}).get(arm) or {}
        lines.append(
            f"  {arm}: solved={item.get('solved')}/{item.get('runs')} "
            f"steps={item.get('mean_main_steps')} effects={item.get('mean_workspace_effects')} "
            f"parse_errors={item.get('mean_main_parse_errors')} "
            f"saturated={item.get('mean_saturated_output_calls')} "
            f"cached={item.get('mean_main_cached_tokens')} "
            f"reasoning={item.get('mean_main_reasoning_tokens')} "
            f"cost=${float(item.get('mean_main_cost_usd') or 0):.6f}"
        )
    for name, comparison in (report.get("comparisons") or {}).items():
        label = name.replace("_minus_", " minus ").replace("_", "-")
        lines.append(
            f"  {label}: solve_delta={comparison.get('solve_rate_delta')} "
            f"effects_delta={comparison.get('workspace_effects_delta')} "
            f"parse_errors_delta={comparison.get('parse_errors_delta')} "
            f"cached_delta={comparison.get('main_cached_tokens_delta')} "
            f"reasoning_delta={comparison.get('main_reasoning_tokens_delta')} "
            f"cost_delta=${float(comparison.get('main_cost_usd_delta') or 0):.6f}"
        )
    return "\n".join(lines)


__all__ = [
    "ARM_DECISION_CARD",
    "ARM_FULL_REPORT",
    "ARM_NO_REPORT",
    "WORKTREE_REPORT_ARMS",
    "normalize_worktree_report_arms",
    "render_worktree_report_summary",
    "run_worktree_report_utilization_eval",
    "summarize_worktree_report_records",
]
