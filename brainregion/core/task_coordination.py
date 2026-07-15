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
EvidenceWakeReason = Literal["explicit_recall", "expert_request", "task_focus_change"]
EvidenceWakeSource = Literal["main_brain", "region_expert", "runtime_policy", "mcp_request"]

_STATUSES = frozenset({"queued", "working", "done", "blocked", "cancelled"})
_EVIDENCE_WAKE_REASONS = frozenset(
    {"explicit_recall", "expert_request", "task_focus_change"}
)
_EVIDENCE_WAKE_SOURCES = frozenset(
    {"main_brain", "region_expert", "runtime_policy", "mcp_request"}
)


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


def _bounded_positive_int(value: Any, name: str, *, maximum: int) -> int:
    result = _positive_int(value, name)
    if result > maximum:
        raise ValueError(f"{name} cannot exceed {maximum}")
    return result


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


@dataclass(frozen=True)
class EvidenceWakeRequest:
    """Content-free request to expose evidence to one exact expert assignment."""

    request_id: str
    task_id: str
    assignment_id: str
    region: str
    reason: EvidenceWakeReason
    source: EvidenceWakeSource
    ttl_reads: int
    remaining_reads: int
    created_sequence: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "assignment_id": self.assignment_id,
            "region": self.region,
            "reason": self.reason,
            "source": self.source,
            "ttl_reads": self.ttl_reads,
            "remaining_reads": self.remaining_reads,
            "created_sequence": self.created_sequence,
            "contains_context_content": False,
        }


class TaskCoordinationBoard:
    """Thread-safe task and assignment registry without private context."""

    def __init__(
        self,
        *,
        max_tasks: int = 256,
        max_assignments: int = 1024,
        max_evidence_wakes: int = 4096,
    ) -> None:
        self._max_tasks = _positive_int(max_tasks, "max_tasks")
        self._max_assignments = _positive_int(max_assignments, "max_assignments")
        self._max_evidence_wakes = _positive_int(
            max_evidence_wakes, "max_evidence_wakes"
        )
        self._tasks: dict[str, TaskSpec] = {}
        self._assignments: dict[str, dict[str, ExpertAssignment]] = {}
        self._evidence_wakes: dict[str, dict[str, list[EvidenceWakeRequest]]] = {}
        self._wake_sequence = 0
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

    def assignment(self, task_id: str, assignment_id: str) -> dict[str, Any]:
        """Return one exact assignment contract without private context."""

        task_id = _required_text(task_id, "task_id", max_length=200)
        assignment_id = _required_text(
            assignment_id, "assignment_id", max_length=200
        )
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"unknown task: {task_id}")
            assignment = self._assignments.get(task_id, {}).get(assignment_id)
        if assignment is None:
            raise ValueError(f"unknown assignment: {assignment_id}")
        return assignment.to_dict()

    def evidence_wake_status(
        self, task_id: str, assignment_id: str
    ) -> dict[str, Any]:
        """Inspect pending wakes for one assignment without consuming read TTL."""

        task_id = _required_text(task_id, "task_id", max_length=200)
        assignment_id = _required_text(
            assignment_id, "assignment_id", max_length=200
        )
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"unknown task: {task_id}")
            assignment = self._assignments.get(task_id, {}).get(assignment_id)
            if assignment is None:
                raise ValueError(f"unknown assignment: {assignment_id}")
            requests = list(
                self._evidence_wakes.get(task_id, {}).get(assignment_id, ())
            )
        return {
            "task_id": task_id,
            "assignment_id": assignment_id,
            "region": assignment.region,
            "wakes": [request.to_dict() for request in requests],
            "count": len(requests),
            "contains_context_content": False,
        }

    def request_evidence_wake(
        self,
        task_id: str,
        assignment_id: str,
        *,
        reason: str,
        source: str,
        ttl_reads: int = 1,
    ) -> dict[str, Any]:
        """Register a bounded wake for one assignment without accepting context text."""

        task_id = _required_text(task_id, "task_id", max_length=200)
        assignment_id = _required_text(
            assignment_id, "assignment_id", max_length=200
        )
        normalized_reason = str(reason or "").strip().casefold()
        if normalized_reason not in _EVIDENCE_WAKE_REASONS:
            raise ValueError(
                "reason must be one of " f"{sorted(_EVIDENCE_WAKE_REASONS)}"
            )
        normalized_source = str(source or "").strip().casefold()
        if normalized_source not in _EVIDENCE_WAKE_SOURCES:
            raise ValueError(
                "source must be one of " f"{sorted(_EVIDENCE_WAKE_SOURCES)}"
            )
        ttl_reads = _bounded_positive_int(ttl_reads, "ttl_reads", maximum=32)
        with self._lock:
            assignment = self._assignments.get(task_id, {}).get(assignment_id)
            if assignment is None:
                raise ValueError(f"unknown assignment: {assignment_id}")
            active_wakes = sum(
                len(requests)
                for assignments in self._evidence_wakes.values()
                for requests in assignments.values()
            )
            if active_wakes >= self._max_evidence_wakes:
                raise RuntimeError("evidence wake capacity exceeded")
            self._wake_sequence += 1
            request = EvidenceWakeRequest(
                request_id=f"wake-{self._wake_sequence:08d}",
                task_id=task_id,
                assignment_id=assignment_id,
                region=assignment.region,
                reason=normalized_reason,  # type: ignore[arg-type]
                source=normalized_source,  # type: ignore[arg-type]
                ttl_reads=ttl_reads,
                remaining_reads=ttl_reads,
                created_sequence=self._wake_sequence,
            )
            self._evidence_wakes.setdefault(task_id, {}).setdefault(
                assignment_id, []
            ).append(request)
        return request.to_dict()

    def consume_evidence_wakes(
        self, task_id: str, assignment_id: str
    ) -> dict[str, Any]:
        """Deliver and age only wakes owned by the exact task/assignment pair."""

        task_id = _required_text(task_id, "task_id", max_length=200)
        assignment_id = _required_text(
            assignment_id, "assignment_id", max_length=200
        )
        with self._lock:
            assignment = self._assignments.get(task_id, {}).get(assignment_id)
            if assignment is None:
                raise ValueError(f"unknown assignment: {assignment_id}")
            requests = self._evidence_wakes.get(task_id, {}).get(assignment_id, [])
            delivered: list[EvidenceWakeRequest] = []
            surviving: list[EvidenceWakeRequest] = []
            for request in requests:
                updated = replace(
                    request, remaining_reads=max(0, request.remaining_reads - 1)
                )
                delivered.append(updated)
                if updated.remaining_reads > 0:
                    surviving.append(updated)
            task_wakes = self._evidence_wakes.get(task_id)
            if task_wakes is not None:
                if surviving:
                    task_wakes[assignment_id] = surviving
                else:
                    task_wakes.pop(assignment_id, None)
                if not task_wakes:
                    self._evidence_wakes.pop(task_id, None)
        return {
            "task_id": task_id,
            "assignment_id": assignment_id,
            "region": assignment.region,
            "deliveries": [request.to_dict() for request in delivered],
            "count": len(delivered),
            "contains_context_content": False,
        }

    def clear_evidence_wakes(
        self, task_id: str, *, assignment_id: str = ""
    ) -> dict[str, Any]:
        """Unload pending wake metadata for one task or exact assignment."""

        task_id = _required_text(task_id, "task_id", max_length=200)
        assignment_id = _optional_text(
            assignment_id, "assignment_id", max_length=200
        )
        with self._lock:
            if assignment_id:
                task_wakes = self._evidence_wakes.get(task_id, {})
                removed = len(task_wakes.pop(assignment_id, []))
                if not task_wakes:
                    self._evidence_wakes.pop(task_id, None)
            else:
                removed = sum(
                    len(requests)
                    for requests in self._evidence_wakes.pop(task_id, {}).values()
                )
        return {
            "task_id": task_id,
            "assignment_id": assignment_id,
            "removed_evidence_wakes": removed,
        }

    def status(self, task_id: str) -> dict[str, Any]:
        task_id = _required_text(task_id, "task_id", max_length=200)
        with self._lock:
            task = self._tasks.get(task_id)
            assignments = list(self._assignments.get(task_id, {}).values())
            evidence_wakes = [
                request
                for requests in self._evidence_wakes.get(task_id, {}).values()
                for request in requests
            ]
        if task is None:
            raise ValueError(f"unknown task: {task_id}")
        return {
            "task": task.to_dict(),
            "assignments": [assignment.to_dict() for assignment in assignments],
            "assignment_count": len(assignments),
            "evidence_wakes": [request.to_dict() for request in evidence_wakes],
            "evidence_wake_count": len(evidence_wakes),
            "contains_context_content": False,
        }

    def clear(self, task_id: str) -> dict[str, Any]:
        task_id = _required_text(task_id, "task_id", max_length=200)
        with self._lock:
            removed_task = self._tasks.pop(task_id, None) is not None
            removed_assignments = len(self._assignments.pop(task_id, {}))
            removed_evidence_wakes = sum(
                len(requests)
                for requests in self._evidence_wakes.pop(task_id, {}).values()
            )
        return {
            "task_id": task_id,
            "removed_task": removed_task,
            "removed_assignments": removed_assignments,
            "removed_evidence_wakes": removed_evidence_wakes,
        }


__all__ = [
    "AssignmentStatus",
    "EvidenceWakeReason",
    "EvidenceWakeRequest",
    "EvidenceWakeSource",
    "ExpertAssignment",
    "MemoryRequest",
    "TaskCoordinationBoard",
    "TaskSpec",
    "TaskStatus",
]
