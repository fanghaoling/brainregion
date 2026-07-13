"""Matched A/B/C evaluation for main-brain and independent expert delegation.

The harness owns experiment ordering and metrics, not model selection. Callers
provide async expert and main runners, so the same design can exercise mocked,
MCP, CLI, or sandbox-backed execution without coupling the eval layer to one
host. Expert runners never receive peer reports.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable

from brainregion.core.region_reporting import RegionReport
from brainregion.core.task_coordination import ExpertAssignment, TaskSpec
from brainregion.runtime import merge_usage, normalize_usage

from .stats import bootstrap_statistic, seed_for

ARM_MAIN_ONLY = "main_only"
ARM_SINGLE_EXPERT = "single_expert"
ARM_MULTI_EXPERT = "multi_expert"
ARM_TRIGGERED_SINGLE_EXPERT = "triggered_single_expert"
DEFAULT_DELEGATION_ARMS = (ARM_MAIN_ONLY, ARM_SINGLE_EXPERT, ARM_MULTI_EXPERT)
DELEGATION_ARMS = (*DEFAULT_DELEGATION_ARMS, ARM_TRIGGERED_SINGLE_EXPERT)
FORMAL_MIN_TASKS = 30
_RECORD_FIELDS = frozenset(
    {
        "task_id",
        "repeat",
        "arm",
        "solved",
        "score",
        "steps",
        "repeated_attempts",
        "reports_produced",
        "reports_adopted",
        "expert_failures",
        "main_input_tokens",
        "main_total_tokens",
        "expert_total_tokens",
        "total_tokens",
        "main_cost_usd",
        "expert_cost_usd",
        "total_cost_usd",
        "main_error",
    }
)
_OPTIONAL_RECORD_FIELDS = frozenset(
    {"protocol_completed", "infrastructure_error", "expert_activations"}
)


def _required_text(value: Any, name: str, *, max_length: int = 4000) -> str:
    text = _bounded_text(value, name, max_length=max_length)
    if not text:
        raise ValueError(f"{name} cannot be empty")
    return text


def _bounded_text(value: Any, name: str, *, max_length: int = 4000) -> str:
    text = str(value or "").strip()
    if len(text) > max_length:
        raise ValueError(f"{name} cannot exceed {max_length} characters")
    return text


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative number")
    number = float(value or 0.0)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return number


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _arms(arms: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    selected = tuple(arms or DEFAULT_DELEGATION_ARMS)
    if not selected:
        raise ValueError("delegation arms cannot be empty")
    unknown = [arm for arm in selected if arm not in DELEGATION_ARMS]
    if unknown:
        raise ValueError(f"unknown delegation arm(s): {unknown}")
    if len(set(selected)) != len(selected):
        raise ValueError("delegation arms cannot contain duplicates")
    return selected


@dataclass(frozen=True)
class DelegationEvalTask:
    task: TaskSpec
    assignments: tuple[ExpertAssignment, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DelegationEvalTask":
        if not isinstance(data, dict):
            raise ValueError("delegation eval task must be an object")
        unknown = set(data) - {
            "task_id",
            "goal",
            "parent_task_id",
            "success_criteria",
            "constraints",
            "status",
            "assignments",
        }
        if unknown:
            raise ValueError(f"delegation eval task unknown field(s): {sorted(unknown)}")
        task = TaskSpec.from_dict({key: value for key, value in data.items() if key != "assignments"})
        raw_assignments = data.get("assignments") or []
        if not isinstance(raw_assignments, list):
            raise ValueError("assignments must be an array")
        assignments = tuple(ExpertAssignment.from_dict(task.task_id, assignment) for assignment in raw_assignments)
        ids = [assignment.assignment_id for assignment in assignments]
        if len(ids) != len(set(ids)):
            raise ValueError("assignment_id values must be unique within a task")
        return cls(task=task, assignments=assignments)

    @classmethod
    def from_task_status(cls, status: dict[str, Any]) -> "DelegationEvalTask":
        if not isinstance(status, dict) or not isinstance(status.get("task"), dict):
            raise ValueError("task status must contain a task object")
        assignments = []
        for raw in status.get("assignments") or []:
            assignment = dict(raw)
            assignment.pop("task_id", None)
            assignment.pop("report_count", None)
            assignment.pop("latest_report", None)
            assignments.append(assignment)
        data = {**status["task"], "assignments": assignments}
        return cls.from_dict(data)

    @property
    def task_id(self) -> str:
        return self.task.task_id

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.task.to_dict(),
            "assignments": [assignment.to_dict() for assignment in self.assignments],
        }


@dataclass(frozen=True)
class DelegationRun:
    task_id: str
    repeat: int
    arm: str
    assignment_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "repeat": self.repeat,
            "arm": self.arm,
            "assignment_ids": list(self.assignment_ids),
        }


@dataclass(frozen=True)
class ExpertEvalResult:
    assignment_id: str
    report: dict[str, Any] | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    cost_source: str | None = None
    model: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignment_id": self.assignment_id,
            "report": dict(self.report) if self.report else None,
            "usage": normalize_usage(self.usage),
            "cost_usd": _nonnegative_float(self.cost_usd, "expert cost_usd"),
            "cost_source": _bounded_text(self.cost_source, "expert cost_source", max_length=100) or None,
            "model": self.model,
            "error": _bounded_text(self.error, "expert error", max_length=500),
            "contains_private_context": False,
        }


@dataclass(frozen=True)
class ExpertActivation:
    """Public, aggregate result returned by an on-demand expert activation."""

    reports: tuple[dict[str, Any], ...]
    assignment_ids: tuple[str, ...]
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    cost_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonnegative_float(self.cost_usd, "expert activation cost_usd")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reports": [dict(report) for report in self.reports],
            "assignment_ids": list(self.assignment_ids),
            "usage": normalize_usage(self.usage),
            "cost_usd": float(self.cost_usd),
            "cost_sources": list(self.cost_sources),
            "contains_private_context": False,
        }


@dataclass(frozen=True)
class MainEvalResult:
    solved: bool
    score: float | None = None
    steps: int = 0
    repeated_attempts: int = 0
    protocol_completed: bool | None = None
    termination_reason: str = ""
    infrastructure_error: bool = False
    adopted_assignment_ids: tuple[str, ...] = ()
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    answer_summary: str = ""
    error: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.solved, bool):
            raise ValueError("main solved must be a boolean")
        if self.score is not None:
            score = float(self.score)
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("main score must be between 0 and 1")
        _nonnegative_int(self.steps, "main steps")
        _nonnegative_int(self.repeated_attempts, "main repeated_attempts")
        if self.protocol_completed is not None and not isinstance(self.protocol_completed, bool):
            raise ValueError("main protocol_completed must be a boolean or null")
        if not isinstance(self.infrastructure_error, bool):
            raise ValueError("main infrastructure_error must be a boolean")
        _bounded_text(self.termination_reason, "termination_reason", max_length=100)
        _nonnegative_float(self.cost_usd, "main cost_usd")
        _bounded_text(self.answer_summary, "answer_summary")
        _bounded_text(self.error, "main error", max_length=500)

    def to_dict(self) -> dict[str, Any]:
        return {
            "solved": self.solved,
            "score": self.score,
            "steps": self.steps,
            "repeated_attempts": self.repeated_attempts,
            "protocol_completed": self.protocol_completed,
            "termination_reason": self.termination_reason,
            "infrastructure_error": self.infrastructure_error,
            "adopted_assignment_ids": list(dict.fromkeys(self.adopted_assignment_ids)),
            "usage": normalize_usage(self.usage),
            "cost_usd": float(self.cost_usd),
            "answer_summary": self.answer_summary,
            "error": self.error,
            "contains_reasoning": False,
        }


@dataclass(frozen=True)
class DelegationCase:
    task_id: str
    repeat: int
    arm: str
    expert_results: tuple[ExpertEvalResult, ...]
    main_result: MainEvalResult

    def metric_record(self) -> dict[str, Any]:
        expert_usage = merge_usage(*(result.usage for result in self.expert_results))
        main_usage = normalize_usage(self.main_result.usage)
        total_usage = merge_usage(main_usage, expert_usage)
        successful_assignments = {
            result.assignment_id for result in self.expert_results if result.report is not None and not result.error
        }
        adopted = successful_assignments.intersection(self.main_result.adopted_assignment_ids)
        return {
            "task_id": self.task_id,
            "repeat": self.repeat,
            "arm": self.arm,
            "solved": self.main_result.solved,
            "score": self.main_result.score,
            "steps": self.main_result.steps,
            "repeated_attempts": self.main_result.repeated_attempts,
            "protocol_completed": self.main_result.protocol_completed,
            "infrastructure_error": self.main_result.infrastructure_error,
            "reports_produced": len(successful_assignments),
            "reports_adopted": len(adopted),
            "expert_activations": len(self.expert_results),
            "expert_failures": sum(1 for result in self.expert_results if result.error),
            "main_input_tokens": main_usage["input_tokens"],
            "main_total_tokens": main_usage["total_tokens"],
            "expert_total_tokens": expert_usage["total_tokens"],
            "total_tokens": total_usage["total_tokens"],
            "main_cost_usd": float(self.main_result.cost_usd),
            "expert_cost_usd": sum(float(result.cost_usd) for result in self.expert_results),
            "total_cost_usd": float(self.main_result.cost_usd)
            + sum(float(result.cost_usd) for result in self.expert_results),
            "main_error": bool(self.main_result.error),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "repeat": self.repeat,
            "arm": self.arm,
            "expert_results": [result.to_dict() for result in self.expert_results],
            "main_result": self.main_result.to_dict(),
            "metrics": self.metric_record(),
        }


ExpertRunner = Callable[[DelegationEvalTask, DelegationRun, ExpertAssignment], Awaitable[ExpertEvalResult]]
ExpertActivator = Callable[[], Awaitable[ExpertActivation]]
MainRunner = Callable[
    [DelegationEvalTask, DelegationRun, tuple[dict[str, Any], ...]],
    Awaitable[MainEvalResult],
]
TriggeredMainRunner = Callable[
    [DelegationEvalTask, DelegationRun, ExpertActivator],
    Awaitable[MainEvalResult],
]


def _selected_assignments(task: DelegationEvalTask, arm: str) -> tuple[ExpertAssignment, ...]:
    if arm == ARM_MAIN_ONLY:
        return ()
    if not task.assignments:
        raise ValueError(f"task {task.task_id!r} has no expert assignments for arm {arm}")
    if arm in {ARM_SINGLE_EXPERT, ARM_TRIGGERED_SINGLE_EXPERT}:
        return task.assignments[:1]
    return task.assignments


def build_delegation_plan(
    task: DelegationEvalTask,
    *,
    repeats: int = 1,
    arms: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    selected_arms = _arms(arms)
    runs = []
    for repeat_index in range(repeats):
        for arm in selected_arms:
            assignments = _selected_assignments(task, arm)
            runs.append(
                DelegationRun(
                    task_id=task.task_id,
                    repeat=repeat_index,
                    arm=arm,
                    assignment_ids=tuple(item.assignment_id for item in assignments),
                ).to_dict()
            )
    return {
        "task_id": task.task_id,
        "arms": list(selected_arms),
        "repeats": repeats,
        "matched_order": "task_then_repeat_then_arm",
        "runs": runs,
        "models_called": False,
    }


def _validated_expert_result(
    task: DelegationEvalTask,
    assignment: ExpertAssignment,
    result: ExpertEvalResult,
) -> ExpertEvalResult:
    if not isinstance(result, ExpertEvalResult):
        raise TypeError("expert runner must return ExpertEvalResult")
    result.to_dict()  # validate cost and bounded public diagnostics before aggregation
    if result.error or result.report is None:
        return replace(result, assignment_id=assignment.assignment_id, report=None)
    data = dict(result.report)
    data.pop("report_id", None)
    data.pop("task_id", None)
    data["assignment_id"] = assignment.assignment_id
    data["region"] = assignment.region
    try:
        report = RegionReport.from_dict(task.task_id, data).to_dict()
    except ValueError as exc:
        return replace(
            result,
            assignment_id=assignment.assignment_id,
            report=None,
            error=f"invalid_region_report: {exc}"[:500],
        )
    return replace(result, assignment_id=assignment.assignment_id, report=report)


def _activation_from_results(results: list[ExpertEvalResult]) -> ExpertActivation:
    reports = tuple(
        dict(result.report) for result in results if result.report is not None and not result.error
    )
    return ExpertActivation(
        reports=reports,
        assignment_ids=tuple(result.assignment_id for result in results),
        usage=merge_usage(*(result.usage for result in results)),
        cost_usd=sum(float(result.cost_usd) for result in results),
        cost_sources=tuple(dict.fromkeys(result.cost_source for result in results if result.cost_source)),
    )


async def run_delegation_eval(
    tasks: list[DelegationEvalTask],
    *,
    main_runner: MainRunner,
    expert_runner: ExpertRunner,
    triggered_main_runner: TriggeredMainRunner | None = None,
    repeats: int = 1,
    arms: list[str] | tuple[str, ...] | None = None,
    run_id: str = "",
    bootstrap_samples: int | None = None,
) -> dict[str, Any]:
    if not tasks:
        raise ValueError("delegation eval tasks cannot be empty")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("delegation eval task ids must be unique")
    selected_arms = _arms(arms)
    if ARM_TRIGGERED_SINGLE_EXPERT in selected_arms and triggered_main_runner is None:
        raise ValueError("triggered_single_expert requires triggered_main_runner")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    run_id = run_id or f"delegation-{int(time.time() * 1000)}"
    cases: list[DelegationCase] = []

    for task in tasks:
        for repeat_index in range(repeats):
            for arm in selected_arms:
                assignments = _selected_assignments(task, arm)
                run = DelegationRun(
                    task_id=task.task_id,
                    repeat=repeat_index,
                    arm=arm,
                    assignment_ids=tuple(item.assignment_id for item in assignments),
                )
                expert_results: list[ExpertEvalResult] = []

                async def activate_experts() -> ExpertActivation:
                    if expert_results:
                        return _activation_from_results(expert_results)
                    for assignment in assignments:
                        try:
                            raw_result = await expert_runner(task, run, assignment)
                            result = _validated_expert_result(task, assignment, raw_result)
                        except Exception as exc:  # noqa: BLE001 - isolate one expert
                            result = ExpertEvalResult(
                                assignment_id=assignment.assignment_id,
                                error=f"expert_runner_error: {exc}"[:500],
                            )
                        expert_results.append(result)
                    return _activation_from_results(expert_results)

                try:
                    if arm == ARM_TRIGGERED_SINGLE_EXPERT:
                        if triggered_main_runner is None:
                            raise RuntimeError("triggered main runner is unavailable")
                        main_result = await triggered_main_runner(task, run, activate_experts)
                    else:
                        activation = await activate_experts()
                        main_result = await main_runner(task, run, activation.reports)
                    if not isinstance(main_result, MainEvalResult):
                        raise TypeError("main runner must return MainEvalResult")
                except Exception as exc:  # noqa: BLE001 - keep matched matrix complete
                    main_result = MainEvalResult(
                        solved=False,
                        infrastructure_error=True,
                        error=f"main_runner_error: {exc}"[:500],
                    )
                cases.append(
                    DelegationCase(
                        task_id=task.task_id,
                        repeat=repeat_index,
                        arm=arm,
                        expert_results=tuple(expert_results),
                        main_result=main_result,
                    )
                )

    summary = summarize_delegation_records(
        [case.metric_record() for case in cases],
        run_id=run_id,
        bootstrap_samples=bootstrap_samples,
    )
    return {
        **summary,
        "cases": [case.to_dict() for case in cases],
        "contains_reasoning": False,
        "contains_private_context": False,
    }


_METRICS = (
    "solved",
    "score",
    "steps",
    "repeated_attempts",
    "protocol_completed",
    "expert_activations",
    "reports_produced",
    "reports_adopted",
    "expert_failures",
    "main_input_tokens",
    "main_total_tokens",
    "expert_total_tokens",
    "total_tokens",
    "main_cost_usd",
    "expert_cost_usd",
    "total_cost_usd",
    "main_error",
)


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return sum(values) / len(values) if values else None


def _per_arm(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["arm"], []).append(record)
    output: dict[str, dict[str, Any]] = {}
    for arm, arm_records in grouped.items():
        valid_records = [record for record in arm_records if not record.get("infrastructure_error")]
        reports = sum(int(record["reports_produced"]) for record in valid_records)
        observable_reports = sum(
            int(record["reports_produced"]) for record in valid_records if record.get("protocol_completed") is not False
        )
        adopted = sum(
            int(record["reports_adopted"]) for record in valid_records if record.get("protocol_completed") is not False
        )
        output[arm] = {
            "n_runs": len(arm_records),
            "n_valid_runs": len(valid_records),
            "n_tasks": len({record["task_id"] for record in arm_records}),
            "valid_run_rate": len(valid_records) / len(arm_records),
            "raw_solve_rate": _mean(arm_records, "solved"),
            "solve_rate": _mean(valid_records, "solved"),
            "mean_score": _mean(valid_records, "score"),
            "mean_steps": _mean(valid_records, "steps"),
            "mean_repeated_attempts": _mean(valid_records, "repeated_attempts"),
            "protocol_completion_rate": _mean(valid_records, "protocol_completed"),
            "expert_activation_rate": (
                sum(int(record["expert_activations"] > 0) for record in valid_records) / len(valid_records)
                if valid_records
                else None
            ),
            "mean_expert_activations": _mean(valid_records, "expert_activations"),
            "mean_main_input_tokens": _mean(valid_records, "main_input_tokens"),
            "mean_main_total_tokens": _mean(valid_records, "main_total_tokens"),
            "mean_expert_total_tokens": _mean(valid_records, "expert_total_tokens"),
            "mean_total_tokens": _mean(valid_records, "total_tokens"),
            "mean_main_cost_usd": _mean(valid_records, "main_cost_usd"),
            "mean_expert_cost_usd": _mean(valid_records, "expert_cost_usd"),
            "mean_total_cost_usd": _mean(valid_records, "total_cost_usd"),
            "report_adoption_rate": adopted / observable_reports if observable_reports else None,
            "adoption_observation_rate": observable_reports / reports if reports else None,
            "expert_failures": sum(int(record["expert_failures"]) for record in arm_records),
            "main_failures": sum(bool(record["main_error"]) for record in arm_records),
            "infrastructure_failures": sum(bool(record.get("infrastructure_error")) for record in arm_records),
        }
    return output


def _paired_task_rows(records: list[dict[str, Any]], control: str, treatment: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for record in records:
        grouped.setdefault(record["task_id"], {}).setdefault(record["repeat"], {})[record["arm"]] = record
    rows: list[dict[str, Any]] = []
    for task_id, by_repeat in grouped.items():
        matched = [
            arms
            for arms in by_repeat.values()
            if control in arms
            and treatment in arms
            and not arms[control].get("infrastructure_error")
            and not arms[treatment].get("infrastructure_error")
        ]
        if not matched:
            continue
        row: dict[str, Any] = {"task_id": task_id, "matched_repeats": len(matched)}
        for arm in (control, treatment):
            arm_records = [arms[arm] for arms in matched]
            row[arm] = {metric: _mean(arm_records, metric) for metric in _METRICS}
        rows.append(row)
    return rows


def _delta(rows: list[dict[str, Any]], control: str, treatment: str, metric: str) -> float | None:
    paired = [
        row[treatment][metric] - row[control][metric]
        for row in rows
        if row[treatment][metric] is not None and row[control][metric] is not None
    ]
    return sum(paired) / len(paired) if paired else None


def _gate(solve_delta: dict[str, Any], n_tasks: int) -> dict[str, Any]:
    prefix = "pilot_" if n_tasks < FORMAL_MIN_TASKS else ""
    if solve_delta["point"] is None:
        return {
            "decision": f"{prefix}INCONCLUSIVE",
            "primary": "solve_rate_delta",
            "reason": "not estimable (requires at least two complete task pairs)",
        }
    low, high = solve_delta["low"], solve_delta["high"]
    if low is not None and low > 0:
        decision, reason = f"{prefix}GO", "solve-rate CI is entirely above zero"
    elif high is not None and high < 0:
        decision, reason = f"{prefix}NO_GO", "solve-rate CI is entirely below zero"
    else:
        decision, reason = f"{prefix}INCONCLUSIVE", "solve-rate CI crosses zero"
    return {"decision": decision, "primary": "solve_rate_delta", "reason": reason}


def summarize_delegation_records(
    records: list[dict[str, Any]],
    *,
    run_id: str = "",
    bootstrap_samples: int | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("delegation records cannot be empty")
    normalized: list[dict[str, Any]] = []
    seen_runs: set[tuple[str, int, str]] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("delegation record must be an object")
        unknown = set(record) - _RECORD_FIELDS - _OPTIONAL_RECORD_FIELDS
        if unknown:
            raise ValueError(f"delegation record unknown field(s): {sorted(unknown)}")
        arm = str(record.get("arm") or "")
        missing = _RECORD_FIELDS - set(record)
        if missing:
            raise ValueError(f"delegation record missing field(s): {sorted(missing)}")
        if arm not in DELEGATION_ARMS:
            raise ValueError(f"unknown delegation arm: {arm!r}")
        if not isinstance(record.get("solved"), bool):
            raise ValueError("delegation record solved must be a boolean")
        score = record.get("score")
        if score is not None:
            if isinstance(score, bool):
                raise ValueError("delegation record score must be between 0 and 1")
            score = float(score)
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("delegation record score must be between 0 and 1")
        reports_produced = _nonnegative_int(record.get("reports_produced"), "reports_produced")
        reports_adopted = _nonnegative_int(record.get("reports_adopted"), "reports_adopted")
        expert_failures = _nonnegative_int(record.get("expert_failures"), "expert_failures")
        expert_activations = _nonnegative_int(
            record.get("expert_activations", reports_produced + expert_failures),
            "expert_activations",
        )
        if reports_adopted > reports_produced:
            raise ValueError("reports_adopted cannot exceed reports_produced")
        if reports_produced + expert_failures > expert_activations:
            raise ValueError("expert_activations cannot be lower than produced reports plus failures")
        if not isinstance(record.get("main_error", False), bool):
            raise ValueError("delegation record main_error must be a boolean")
        protocol_completed = record.get("protocol_completed")
        if protocol_completed is not None and not isinstance(protocol_completed, bool):
            raise ValueError("delegation record protocol_completed must be a boolean or null")
        infrastructure_error = record.get("infrastructure_error", False)
        if not isinstance(infrastructure_error, bool):
            raise ValueError("delegation record infrastructure_error must be a boolean")
        if protocol_completed is False and reports_adopted:
            raise ValueError("reports_adopted requires an observable protocol completion")
        item = {
            "task_id": _required_text(record.get("task_id"), "task_id", max_length=200),
            "repeat": _nonnegative_int(record.get("repeat"), "repeat"),
            "arm": arm,
            "solved": record["solved"],
            "score": score,
            "steps": _nonnegative_int(record.get("steps"), "steps"),
            "repeated_attempts": _nonnegative_int(record.get("repeated_attempts"), "repeated_attempts"),
            "protocol_completed": protocol_completed,
            "infrastructure_error": infrastructure_error,
            "reports_produced": reports_produced,
            "reports_adopted": reports_adopted,
            "expert_activations": expert_activations,
            "expert_failures": expert_failures,
            "main_input_tokens": _nonnegative_int(record.get("main_input_tokens"), "main_input_tokens"),
            "main_total_tokens": _nonnegative_int(record.get("main_total_tokens"), "main_total_tokens"),
            "expert_total_tokens": _nonnegative_int(record.get("expert_total_tokens"), "expert_total_tokens"),
            "total_tokens": _nonnegative_int(record.get("total_tokens"), "total_tokens"),
            "main_cost_usd": _nonnegative_float(record.get("main_cost_usd"), "main_cost_usd"),
            "expert_cost_usd": _nonnegative_float(record.get("expert_cost_usd"), "expert_cost_usd"),
            "total_cost_usd": _nonnegative_float(record.get("total_cost_usd"), "total_cost_usd"),
            "main_error": record["main_error"],
        }
        run_key = (item["task_id"], item["repeat"], arm)
        if run_key in seen_runs:
            raise ValueError(f"duplicate delegation run: {run_key}")
        seen_runs.add(run_key)
        if item["main_input_tokens"] > item["main_total_tokens"]:
            raise ValueError("main_input_tokens cannot exceed main_total_tokens")
        if item["main_total_tokens"] + item["expert_total_tokens"] != item["total_tokens"]:
            raise ValueError("total_tokens must equal main_total_tokens plus expert_total_tokens")
        expected_cost = item["main_cost_usd"] + item["expert_cost_usd"]
        if not math.isclose(item["total_cost_usd"], expected_cost, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("total_cost_usd must equal main_cost_usd plus expert_cost_usd")
        if arm == ARM_MAIN_ONLY and any(
            item[key]
            for key in (
                "reports_produced",
                "reports_adopted",
                "expert_activations",
                "expert_failures",
                "expert_total_tokens",
                "expert_cost_usd",
            )
        ):
            raise ValueError("main_only records cannot contain expert activity")
        normalized.append(item)
    selected_arms = tuple(arm for arm in DELEGATION_ARMS if any(r["arm"] == arm for r in normalized))
    run_id = run_id or f"delegation-summary-{int(time.time() * 1000)}"
    pairwise: dict[str, Any] = {}
    if ARM_MAIN_ONLY in selected_arms:
        for treatment in selected_arms:
            if treatment == ARM_MAIN_ONLY:
                continue
            pair_rows = _paired_task_rows(normalized, ARM_MAIN_ONLY, treatment)
            deltas: dict[str, Any] = {}
            for metric in (
                "solved",
                "score",
                "protocol_completed",
                "repeated_attempts",
                "main_input_tokens",
                "total_tokens",
                "total_cost_usd",
            ):
                label = f"{ARM_MAIN_ONLY}_vs_{treatment}:{metric}"
                deltas[f"{metric}_delta"] = bootstrap_statistic(
                    pair_rows,
                    lambda sample, c=ARM_MAIN_ONLY, t=treatment, m=metric: _delta(sample, c, t, m),
                    B=bootstrap_samples,
                    seed=seed_for(run_id, label),
                )
            pairwise[f"{ARM_MAIN_ONLY}_vs_{treatment}"] = {
                "control": ARM_MAIN_ONLY,
                "treatment": treatment,
                "n_tasks": len(pair_rows),
                "n_matched_repeats": sum(row["matched_repeats"] for row in pair_rows),
                "deltas": deltas,
                "gate": _gate(deltas["solved_delta"], len(pair_rows)),
            }
    return {
        "run_id": run_id,
        "arms": list(selected_arms),
        "n_tasks": len({record["task_id"] for record in normalized}),
        "n_runs": len(normalized),
        "bootstrap_unit": "task",
        "per_arm": _per_arm(normalized),
        "pairwise": pairwise,
        "records": normalized,
    }


__all__ = [
    "ARM_MAIN_ONLY",
    "ARM_MULTI_EXPERT",
    "ARM_SINGLE_EXPERT",
    "ARM_TRIGGERED_SINGLE_EXPERT",
    "DEFAULT_DELEGATION_ARMS",
    "DELEGATION_ARMS",
    "DelegationCase",
    "DelegationEvalTask",
    "DelegationRun",
    "ExpertEvalResult",
    "ExpertActivation",
    "MainEvalResult",
    "build_delegation_plan",
    "run_delegation_eval",
    "summarize_delegation_records",
]
