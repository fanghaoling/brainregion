"""Task decomposition contracts and a process-local coordination board.

The board stores goals, constraints, assignments, dependencies, and context
request metadata. It never stores ContextBlock contents or model reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock
from typing import Any, Literal

TaskStatus = Literal["queued", "working", "done", "blocked", "cancelled"]
AssignmentStatus = Literal["queued", "working", "done", "blocked", "cancelled"]

_STATUSES = frozenset({"queued", "working", "done", "blocked", "cancelled"})


def _required_text(value: Any, name: str, *, max_length: int = 2000) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} cannot be empty")
    if len(text) > max_length:
        raise ValueError(f"{name} cannot exceed {max_length} characters")
    return text


def _optional_text(value: Any, name: str, *, max_length: int = 4000) -> str:
    text = str(value or "").strip()
    if len(text) > max_length:
        raise ValueError(f"{name} cannot exceed {max_length} characters")
    return text


def _string_tuple(value: Any, name: str, *, max_items: int = 64) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(f"{name} must be an array")
    output: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in output:
            output.append(text)
    if len(output) > max_items:
        raise ValueError(f"{name} cannot contain more than {max_items} items")
    return tuple(output)


def _status(value: Any, name: str, default: str) -> str:
    normalized = str(value or default).strip().casefold()
    if normalized not in _STATUSES:
        raise ValueError(f"{name} must be one of {sorted(_STATUSES)}")
    return normalized


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class MemoryRequest:
    query: str = ""
    purpose: str = ""
    regions: tuple[str, ...] = ()
    selectors: tuple[str, ...] = ()
    top_k: int = 5
    max_context_tokens: int = 1600
    audience: Literal["region"] = "region"
    target_region: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, default_region: str) -> "MemoryRequest":
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("memory_request must be an object")
        unknown = set(data) - {
            "query",
            "purpose",
            "regions",
            "selectors",
            "top_k",
            "max_context_tokens",
            "audience",
            "target_region",
        }
        if unknown:
            raise ValueError(f"memory_request unknown field(s): {sorted(unknown)}")
        audience = str(data.get("audience") or "region").strip().casefold()
        if audience != "region":
            raise ValueError("memory_request.audience must be region")
        return cls(
            query=_optional_text(data.get("query"), "memory_request.query"),
            purpose=_optional_text(data.get("purpose"), "memory_request.purpose"),
            regions=_string_tuple(data.get("regions"), "memory_request.regions"),
            selectors=_string_tuple(data.get("selectors"), "memory_request.selectors"),
            top_k=_positive_int(data.get("top_k", 5), "memory_request.top_k"),
            max_context_tokens=_positive_int(
                data.get("max_context_tokens", 1600),
                "memory_request.max_context_tokens",
            ),
            audience="region",
            target_region=_optional_text(
                data.get("target_region") or default_region,
                "memory_request.target_region",
                max_length=200,
            ).casefold(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "purpose": self.purpose,
            "regions": list(self.regions),
            "selectors": list(self.selectors),
            "top_k": self.top_k,
            "max_context_tokens": self.max_context_tokens,
            "audience": self.audience,
            "target_region": self.target_region,
        }


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    goal: str
    parent_task_id: str = ""
    success_criteria: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    status: TaskStatus = "queued"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskSpec":
        if not isinstance(data, dict):
            raise ValueError("task must be an object")
        unknown = set(data) - {
            "task_id",
            "goal",
            "parent_task_id",
            "success_criteria",
            "constraints",
            "status",
        }
        if unknown:
            raise ValueError(f"task unknown field(s): {sorted(unknown)}")
        return cls(
            task_id=_required_text(data.get("task_id"), "task_id", max_length=200),
            goal=_required_text(data.get("goal"), "goal"),
            parent_task_id=_optional_text(data.get("parent_task_id"), "parent_task_id", max_length=200),
            success_criteria=_string_tuple(data.get("success_criteria"), "success_criteria"),
            constraints=_string_tuple(data.get("constraints"), "constraints"),
            status=_status(data.get("status"), "task status", "queued"),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "goal": self.goal,
            "success_criteria": list(self.success_criteria),
            "constraints": list(self.constraints),
            "status": self.status,
        }


@dataclass(frozen=True)
class ExpertAssignment:
    assignment_id: str
    task_id: str
    region: str
    question: str
    scope: str = ""
    depends_on: tuple[str, ...] = ()
    memory_request: MemoryRequest = MemoryRequest()
    expected_output: str = "region_report"
    status: AssignmentStatus = "queued"

    @classmethod
    def from_dict(cls, task_id: str, data: dict[str, Any]) -> "ExpertAssignment":
        if not isinstance(data, dict):
            raise ValueError("assignment must be an object")
        unknown = set(data) - {
            "assignment_id",
            "region",
            "question",
            "scope",
            "depends_on",
            "memory_request",
            "expected_output",
            "status",
        }
        if unknown:
            raise ValueError(f"assignment unknown field(s): {sorted(unknown)}")
        region = _required_text(data.get("region"), "region", max_length=200).casefold()
        expected_output = str(data.get("expected_output") or "region_report").strip().casefold()
        if expected_output != "region_report":
            raise ValueError("expected_output must be region_report")
        return cls(
            assignment_id=_required_text(data.get("assignment_id"), "assignment_id", max_length=200),
            task_id=_required_text(task_id, "task_id", max_length=200),
            region=region,
            question=_required_text(data.get("question"), "question"),
            scope=_optional_text(data.get("scope"), "scope"),
            depends_on=_string_tuple(data.get("depends_on"), "depends_on"),
            memory_request=MemoryRequest.from_dict(data.get("memory_request"), default_region=region),
            expected_output=expected_output,
            status=_status(data.get("status"), "assignment status", "queued"),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignment_id": self.assignment_id,
            "task_id": self.task_id,
            "region": self.region,
            "question": self.question,
            "scope": self.scope,
            "depends_on": list(self.depends_on),
            "memory_request": self.memory_request.to_dict(),
            "expected_output": self.expected_output,
            "status": self.status,
        }


class TaskCoordinationBoard:
    """Thread-safe task and assignment registry without private context."""

    def __init__(self, *, max_tasks: int = 256, max_assignments: int = 1024) -> None:
        self._max_tasks = _positive_int(max_tasks, "max_tasks")
        self._max_assignments = _positive_int(max_assignments, "max_assignments")
        self._tasks: dict[str, TaskSpec] = {}
        self._assignments: dict[str, dict[str, ExpertAssignment]] = {}
        self._lock = RLock()

    def create_task(self, data: dict[str, Any]) -> dict[str, Any]:
        task = TaskSpec.from_dict(data)
        with self._lock:
            if task.task_id in self._tasks:
                raise ValueError(f"task already exists: {task.task_id}")
            if len(self._tasks) >= self._max_tasks:
                raise RuntimeError("task capacity exceeded")
            self._tasks[task.task_id] = task
        return task.to_dict()

    def delegate(self, task_id: str, data: dict[str, Any]) -> dict[str, Any]:
        assignment = ExpertAssignment.from_dict(task_id, data)
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"unknown task: {task_id}")
            task_assignments = self._assignments.setdefault(task_id, {})
            if assignment.assignment_id in task_assignments:
                raise ValueError(f"assignment already exists: {assignment.assignment_id}")
            total = sum(len(items) for items in self._assignments.values())
            if total >= self._max_assignments:
                raise RuntimeError("assignment capacity exceeded")
            task_assignments[assignment.assignment_id] = assignment
        return assignment.to_dict()

    def set_assignment_status(self, task_id: str, assignment_id: str, status: str) -> dict[str, Any]:
        normalized = _status(status, "assignment status", "working")
        with self._lock:
            assignment = self._assignments.get(task_id, {}).get(assignment_id)
            if assignment is None:
                raise ValueError(f"unknown assignment: {assignment_id}")
            updated = replace(assignment, status=normalized)
            self._assignments[task_id][assignment_id] = updated
        return updated.to_dict()

    def status(self, task_id: str) -> dict[str, Any]:
        task_id = _required_text(task_id, "task_id", max_length=200)
        with self._lock:
            task = self._tasks.get(task_id)
            assignments = list(self._assignments.get(task_id, {}).values())
        if task is None:
            raise ValueError(f"unknown task: {task_id}")
        return {
            "task": task.to_dict(),
            "assignments": [assignment.to_dict() for assignment in assignments],
            "assignment_count": len(assignments),
            "contains_context_content": False,
        }

    def clear(self, task_id: str) -> dict[str, Any]:
        task_id = _required_text(task_id, "task_id", max_length=200)
        with self._lock:
            removed_task = self._tasks.pop(task_id, None) is not None
            removed_assignments = len(self._assignments.pop(task_id, {}))
        return {
            "task_id": task_id,
            "removed_task": removed_task,
            "removed_assignments": removed_assignments,
        }


__all__ = [
    "AssignmentStatus",
    "ExpertAssignment",
    "MemoryRequest",
    "TaskCoordinationBoard",
    "TaskSpec",
    "TaskStatus",
]
