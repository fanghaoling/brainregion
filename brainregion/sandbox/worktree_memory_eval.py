"""Matched real-worktree evaluation for expert-scoped memory delivery."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brainregion.core.activation import ActivationPlan
from brainregion.core.cognitive_workspace import CognitiveWorkspace
from brainregion.core.context import ContextBlock
from brainregion.core.context_loader import ActivatedContext
from brainregion.core.region_expert import RegionExpertEngine
from brainregion.core.region_reporting import RegionCoordinationBoard
from brainregion.runtime import normalize_usage

from .delegation_eval import render_expert_reports
from .loop import run_agent
from .task import WorktreeTask
from .worktree import (
    bootstrap_worktree,
    capture_worktree_diff,
    detect_venv_python,
    worktree,
)

ARM_MAIN_ONLY = "main_only"
ARM_EXPERT_NO_MEMORY = "expert_without_memory"
ARM_EXPERT_SCOPED_MEMORY = "expert_with_scoped_memory"
WORKTREE_MEMORY_ARMS = (
    ARM_MAIN_ONLY,
    ARM_EXPERT_NO_MEMORY,
    ARM_EXPERT_SCOPED_MEMORY,
)

_MAX_CONTEXT_PATHS = 16
_MAX_CONTEXT_FILE_CHARS = 64_000
_MAX_CONTEXT_TOTAL_CHARS = 256_000
_MAX_MEMORY_RECORDS = 24


@dataclass(frozen=True)
class WorktreeMemoryExpertSpec:
    assignment_id: str
    region: str
    question: str
    model: str
    endpoint_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("assignment_id", "region", "question", "model"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            if len(value) > (4000 if name == "question" else 300):
                raise ValueError(f"{name} is too long")
            object.__setattr__(self, name, value.casefold() if name == "region" else value)
        endpoint_id = str(self.endpoint_id or "").strip() or None
        object.__setattr__(self, "endpoint_id", endpoint_id)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _safe_source_blocks(root: str, paths: list[str]) -> tuple[ContextBlock, ...]:
    if not paths:
        raise ValueError("expert_context_paths cannot be empty")
    if len(paths) > _MAX_CONTEXT_PATHS:
        raise ValueError(f"expert_context_paths cannot exceed {_MAX_CONTEXT_PATHS}")
    root_path = Path(root).resolve()
    blocks: list[ContextBlock] = []
    total_chars = 0
    seen: set[str] = set()
    for raw_path in paths:
        relative = str(raw_path or "").strip().replace("\\", "/")
        if not relative or relative in seen:
            raise ValueError("expert_context_paths must be non-empty and unique")
        candidate = (root_path / relative).resolve()
        try:
            candidate.relative_to(root_path)
        except ValueError as exc:
            raise ValueError(f"expert context path escapes worktree: {relative!r}") from exc
        if not candidate.is_file():
            raise ValueError(f"expert context path is not a file: {relative!r}")
        content = candidate.read_text(encoding="utf-8", errors="replace")
        if len(content) > _MAX_CONTEXT_FILE_CHARS:
            raise ValueError(f"expert context file exceeds character cap: {relative!r}")
        total_chars += len(content)
        if total_chars > _MAX_CONTEXT_TOTAL_CHARS:
            raise ValueError("expert context files exceed total character cap")
        seen.add(relative)
        blocks.append(
            ContextBlock(
                source="worktree",
                title=f"Repository file: {relative}",
                content=content,
                metadata={"path": relative, "kind": "source_snapshot"},
            )
        )
    return tuple(blocks)


def _scoped_memory_blocks(
    records: list[dict[str, Any]], region: str
) -> tuple[tuple[ContextBlock, ...], int]:
    if len(records) > _MAX_MEMORY_RECORDS:
        raise ValueError(f"seed_memory cannot exceed {_MAX_MEMORY_RECORDS} records")
    blocks: list[ContextBlock] = []
    excluded = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("seed_memory records must be objects")
        record_region = str(record.get("region") or "shared").strip().casefold()
        if record_region not in {"", "shared", region}:
            excluded += 1
            continue
        summary = str(record.get("summary") or "").strip()
        if not summary or len(summary) > 8000:
            raise ValueError("selected seed_memory summary must contain 1..8000 characters")
        memory_id = str(record.get("id") or f"memory-{index}").strip()
        if not memory_id or len(memory_id) > 200:
            raise ValueError("seed_memory id must contain 1..200 characters")
        blocks.append(
            ContextBlock(
                source="memory",
                title=str(record.get("title") or f"Scoped project memory {memory_id}")[:300],
                content=summary,
                metadata={
                    "id": memory_id,
                    "region": record_region or "shared",
                    "status": str(record.get("status") or "unknown")[:100],
                },
            )
        )
    return tuple(blocks), excluded


def _activated_context(
    blocks: tuple[ContextBlock, ...], *, region: str, mode: str
) -> ActivatedContext:
    return ActivatedContext(
        activation=ActivationPlan(
            decisions=(),
            woken_regions=(region,),
            context_requests=(),
            trace={"strategy": mode, "models_called": False},
        ),
        blocks=blocks,
        loads=(),
        trace={"provider": mode, "models_called": False},
    )


async def _run_expert(
    backend: Any,
    task: WorktreeTask,
    root: str,
    spec: WorktreeMemoryExpertSpec,
    *,
    include_memory: bool,
    max_context_tokens: int,
    max_tokens: int,
    temperature: float,
    effort: str | None,
) -> tuple[str, dict[str, Any]]:
    source_blocks = _safe_source_blocks(root, task.expert_context_paths)
    memory_blocks: tuple[ContextBlock, ...] = ()
    excluded_memory = 0
    if include_memory:
        memory_blocks, excluded_memory = _scoped_memory_blocks(task.seed_memory, spec.region)
        if not memory_blocks:
            raise ValueError("expert_with_scoped_memory requires matching seed_memory")
    blocks = (*source_blocks, *memory_blocks)
    workspace = CognitiveWorkspace(max_entries=2)
    workspace.stage(
        _activated_context(
            blocks,
            region=spec.region,
            mode="worktree_source_and_scoped_memory" if include_memory else "worktree_source_only",
        ),
        task_id=task.id,
        audience="region",
        target_region=spec.region,
        assignment_id=spec.assignment_id,
        ttl_steps=1,
    )
    result = await RegionExpertEngine(backend=backend).run(
        workspace=workspace,
        coordination=RegionCoordinationBoard(),
        task_id=task.id,
        region=spec.region,
        assignment_id=spec.assignment_id,
        task=(
            f"Goal: {task.goal}\n"
            f"Expert question: {spec.question}\n"
            "Diagnose the bounded repository task. Return a grounded report; do not edit files."
        ),
        model=spec.model,
        endpoint_id=spec.endpoint_id,
        max_context_tokens=max_context_tokens,
        max_blocks=len(blocks),
        max_tokens=max_tokens,
        temperature=temperature,
        effort=effort,
    )
    published = result.published_report or {}
    report = published.get("report") if isinstance(published, dict) else None
    advisory = render_expert_reports((report,)) if isinstance(report, dict) else ""
    usage = normalize_usage(result.usage or {})
    return advisory, {
        "model_called": result.model_called,
        "report_produced": bool(advisory),
        "error_observed": bool(result.error),
        "parse_ok": result.parse_ok,
        "source_blocks": len(source_blocks),
        "memory_blocks": len(memory_blocks),
        "memory_records_excluded_by_scope": excluded_memory,
        "context_tokens_estimated": result.context_tokens_estimated,
        "context_truncated": result.context_truncated,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "cost_usd": round(float(result.cost_usd or 0.0), 6),
        "cost_source": result.cost_source,
        "private_context_returned": False,
        "contains_report_content": False,
        "contains_reasoning": False,
    }


async def run_worktree_memory_eval(
    backend: Any,
    main_model: str,
    task: WorktreeTask,
    expert: WorktreeMemoryExpertSpec,
    *,
    main_endpoint_id: str | None = None,
    repeats: int = 2,
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
    run_id: str = "",
) -> dict[str, Any]:
    """Run three isolated real-repository arms with memory private to the expert."""

    repeats = _positive_int(repeats, "repeats")
    max_steps = _positive_int(max_steps, "max_steps")
    expert_max_context_tokens = _positive_int(
        expert_max_context_tokens, "expert_max_context_tokens"
    )
    if not str(main_model or "").strip():
        raise ValueError("main_model cannot be empty")
    if not task.expert_context_paths:
        raise ValueError("worktree memory eval requires expert_context_paths")
    if not task.seed_memory:
        raise ValueError("worktree memory eval requires seed_memory")
    run_id = run_id or f"worktree-memory-eval-{int(time.time() * 1000)}"
    records: list[dict[str, Any]] = []
    order_counts = {f"{arm}_first": 0 for arm in WORKTREE_MEMORY_ARMS}
    execution_order = 0
    for repeat in range(repeats):
        offset = repeat % len(WORKTREE_MEMORY_ARMS)
        ordered_arms = (*WORKTREE_MEMORY_ARMS[offset:], *WORKTREE_MEMORY_ARMS[:offset])
        order_counts[f"{ordered_arms[0]}_first"] += 1
        for arm in ordered_arms:
            execution_order += 1
            expert_metrics = {
                "model_called": False,
                "report_produced": False,
                "memory_blocks": 0,
                "private_context_returned": False,
                "contains_report_content": False,
                "contains_reasoning": False,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }
            bootstrap_failures = 0
            with worktree(task.repo_path, task.base_ref) as handle:
                bootstrap = bootstrap_worktree(handle, task.bootstrap_commands)
                bootstrap_failures = sum(int(item["returncode"] != 0) for item in bootstrap)
                advisory = ""
                if arm != ARM_MAIN_ONLY:
                    advisory, expert_metrics = await _run_expert(
                        backend,
                        task,
                        handle.path,
                        expert,
                        include_memory=arm == ARM_EXPERT_SCOPED_MEMORY,
                        max_context_tokens=expert_max_context_tokens,
                        max_tokens=expert_max_tokens,
                        temperature=expert_temperature,
                        effort=effort,
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
                diff = capture_worktree_diff(handle)
            main_usage = normalize_usage(trajectory.total_main_usage)
            records.append(
                {
                    "arm": arm,
                    "repeat": repeat,
                    "execution_order": execution_order,
                    "solved": trajectory.tests_green,
                    "protocol_completed": trajectory.done,
                    "termination_reason": trajectory.termination_reason,
                    "infrastructure_error": trajectory.termination_reason == "model_error",
                    "main_steps": trajectory.n_steps,
                    "workspace_effects": trajectory.workspace_effects,
                    "repeated_attempts": max(0, trajectory.workspace_effects - 1),
                    "verification_runs": trajectory.verification_runs,
                    "last_verification_passed": trajectory.last_verification_passed,
                    "adopted_expert_report": bool(trajectory.adopted_assignment_ids),
                    "main_input_tokens": main_usage["input_tokens"],
                    "main_output_tokens": main_usage["output_tokens"],
                    "main_total_tokens": main_usage["total_tokens"],
                    "main_cost_usd": round(float(trajectory.total_main_cost_usd), 6),
                    "expert": expert_metrics,
                    "total_tokens": main_usage["total_tokens"]
                    + int(expert_metrics.get("total_tokens", 0)),
                    "total_cost_usd": round(
                        float(trajectory.total_main_cost_usd)
                        + float(expert_metrics.get("cost_usd", 0.0)),
                        6,
                    ),
                    "bootstrap_failure_count": bootstrap_failures,
                    "diff_changed": bool(str(diff.get("diff_stat") or "").strip()),
                    "contains_diff_content": False,
                    "contains_private_context": False,
                    "contains_tool_results": False,
                    "contains_reasoning": False,
                }
            )
    report = summarize_worktree_memory_records(records, run_id=run_id)
    report["records"] = records
    report["execution"] = {
        "runner": "isolated_git_worktree",
        "task_id": task.id,
        "base_ref": task.base_ref,
        "main_model": main_model,
        "main_endpoint_id": main_endpoint_id,
        "expert_model": expert.model,
        "expert_endpoint_id": expert.endpoint_id,
        "expert_region": expert.region,
        "repeats": repeats,
        "arm_order_policy": "rotating_by_repeat",
        "arm_order_counts": order_counts,
        "explicit_model_calls": True,
        "changes_runtime_policy": False,
        "contains_repo_path": False,
        "contains_source_content": False,
        "contains_memory_content": False,
        "contains_report_content": False,
        "contains_diff_content": False,
        "contains_reasoning": False,
    }
    return report


def _mean(items: list[dict[str, Any]], field: str) -> float:
    return round(sum(float(item.get(field, 0)) for item in items) / len(items), 4)


def summarize_worktree_memory_records(
    records: list[dict[str, Any]], *, run_id: str = ""
) -> dict[str, Any]:
    if not records:
        raise ValueError("worktree memory records cannot be empty")
    grouped = {arm: [] for arm in WORKTREE_MEMORY_ARMS}
    pairs: dict[int, set[str]] = {}
    for record in records:
        arm = str(record.get("arm") or "")
        if arm not in grouped:
            raise ValueError(f"unknown worktree memory arm: {arm!r}")
        repeat = int(record.get("repeat", 0))
        if arm in pairs.setdefault(repeat, set()):
            raise ValueError(f"duplicate worktree memory record: {repeat}/{arm}")
        pairs[repeat].add(arm)
        grouped[arm].append(record)
    incomplete = [repeat for repeat, arms in pairs.items() if arms != set(WORKTREE_MEMORY_ARMS)]
    if incomplete:
        raise ValueError(f"incomplete worktree memory repeats: {incomplete}")
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
            "mean_repeated_attempts": _mean(items, "repeated_attempts"),
            "mean_verification_runs": _mean(items, "verification_runs"),
            "mean_total_tokens": _mean(items, "total_tokens"),
            "mean_total_cost_usd": _mean(items, "total_cost_usd"),
            "expert_activation_rate": round(
                sum(int(bool((item.get("expert") or {}).get("model_called"))) for item in items)
                / runs,
                4,
            ),
            "expert_report_rate": round(
                sum(int(bool((item.get("expert") or {}).get("report_produced"))) for item in items)
                / runs,
                4,
            ),
            "mean_expert_memory_blocks": round(
                sum(int((item.get("expert") or {}).get("memory_blocks", 0)) for item in items)
                / runs,
                4,
            ),
        }
    main = per_arm[ARM_MAIN_ONLY]
    no_memory = per_arm[ARM_EXPERT_NO_MEMORY]
    scoped = per_arm[ARM_EXPERT_SCOPED_MEMORY]
    return {
        "run_id": run_id,
        "mode": "real_worktree_scoped_memory_ab",
        "pair_count": len(pairs),
        "per_arm": per_arm,
        "comparisons": {
            "expert_presence": {
                "solve_delta_expert_without_memory_minus_main": round(
                    no_memory["solve_rate"] - main["solve_rate"], 4
                )
            },
            "scoped_memory_value": {
                "solve_delta_scoped_memory_minus_expert_without_memory": round(
                    scoped["solve_rate"] - no_memory["solve_rate"], 4
                ),
                "main_steps_delta_scoped_memory_minus_expert_without_memory": round(
                    scoped["mean_main_steps"] - no_memory["mean_main_steps"], 4
                ),
                "total_tokens_delta_scoped_memory_minus_expert_without_memory": round(
                    scoped["mean_total_tokens"] - no_memory["mean_total_tokens"], 4
                ),
                "interpretation": "paired_real_task_association_not_general_memory_value",
            },
        },
        "infrastructure_usable": not any(
            bool(item.get("infrastructure_error")) for item in records
        ),
        "total_cost_usd": round(
            sum(float(item.get("total_cost_usd", 0.0)) for item in records), 6
        ),
        "contains_source_content": False,
        "contains_memory_content": False,
        "contains_report_content": False,
        "contains_diff_content": False,
        "contains_reasoning": False,
    }


def render_worktree_memory_summary(report: dict[str, Any]) -> str:
    lines = [f"Worktree scoped-memory A/B: {report.get('run_id') or '-'}"]
    for arm in WORKTREE_MEMORY_ARMS:
        metrics = (report.get("per_arm") or {}).get(arm) or {}
        lines.append(
            f"  {arm}: solved={metrics.get('solved')}/{metrics.get('runs')} "
            f"rate={metrics.get('solve_rate')} steps={metrics.get('mean_main_steps')} "
            f"tokens={metrics.get('mean_total_tokens')} "
            f"cost=${float(metrics.get('mean_total_cost_usd') or 0.0):.6f}"
        )
    memory = (report.get("comparisons") or {}).get("scoped_memory_value") or {}
    lines.append(
        "  scoped-no-memory: solve_delta="
        f"{memory.get('solve_delta_scoped_memory_minus_expert_without_memory')} "
        f"steps_delta={memory.get('main_steps_delta_scoped_memory_minus_expert_without_memory')}"
    )
    return "\n".join(lines)
