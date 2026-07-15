"""Wake-gated expert runner for one exact task assignment.

The runner owns the delivery lifecycle around an existing RegionExpertEngine.
It never stores private context: evidence remains in CognitiveWorkspace, wake
metadata remains in TaskCoordinationBoard, and only a validated RegionReport
crosses back to the main-brain surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .cognitive_workspace import CognitiveWorkspace, WorkspaceView
from .region_expert import RegionExpertEngine, RegionExpertResult
from .region_reporting import RegionCoordinationBoard
from .task_coordination import TaskCoordinationBoard

AssignmentExpertState = Literal["sleeping", "waiting_context", "awake", "blocked"]


def _assignment_task(assignment: dict[str, Any]) -> str:
    question = str(assignment.get("question") or "").strip()
    scope = str(assignment.get("scope") or "").strip()
    if scope:
        return f"{question}\n\nDELEGATED SCOPE:\n{scope}"
    return question


def _pending_provider_reads(wake_status: dict[str, Any]) -> int:
    return sum(
        max(0, int(wake.get("remaining_reads") or 0))
        for wake in wake_status.get("wakes", ())
    )


@dataclass(frozen=True)
class AssignmentExpertResult:
    state: AssignmentExpertState
    task_id: str
    assignment_id: str
    region: str
    model: str
    endpoint_id: str | None
    expert_result: RegionExpertResult | None
    wake_deliveries: tuple[dict[str, Any], ...] = ()
    pending_wake_requests: int = 0
    pending_provider_reads: int = 0

    @classmethod
    def sleeping(
        cls,
        *,
        assignment: dict[str, Any],
        model: str,
        endpoint_id: str | None = None,
        pending_wake_requests: int = 0,
        pending_provider_reads: int = 0,
    ) -> "AssignmentExpertResult":
        return cls(
            state="sleeping",
            task_id=str(assignment["task_id"]),
            assignment_id=str(assignment["assignment_id"]),
            region=str(assignment["region"]),
            model=str(model),
            endpoint_id=endpoint_id,
            expert_result=None,
            pending_wake_requests=pending_wake_requests,
            pending_provider_reads=pending_provider_reads,
        )

    def to_dict(self) -> dict[str, Any]:
        if self.expert_result is None:
            output: dict[str, Any] = {
                "ok": True,
                "task_id": self.task_id,
                "region": self.region,
                "model": self.model,
                "endpoint_id": self.endpoint_id,
                "published_report": None,
                "context": {
                    "blocks_used": 0,
                    "estimated_tokens": 0,
                    "private_context_returned": False,
                },
                "usage": {},
                "cost_usd": None,
                "cost_source": None,
                "parse_ok": True,
                "error": "",
                "model_called": False,
            }
        else:
            output = self.expert_result.to_dict()
        reasons = sorted(
            {
                str(delivery.get("reason") or "")
                for delivery in self.wake_deliveries
                if delivery.get("reason")
            }
        )
        sources = sorted(
            {
                str(delivery.get("source") or "")
                for delivery in self.wake_deliveries
                if delivery.get("source")
            }
        )
        output["assignment_id"] = self.assignment_id
        output["assignment_lifecycle"] = {
            "state": self.state,
            "wake_required": True,
            "wake_delivered": bool(self.wake_deliveries),
            "wake_delivery_count": len(self.wake_deliveries),
            "wake_request_ids": [
                str(delivery.get("request_id") or "")
                for delivery in self.wake_deliveries
            ],
            "wake_reasons": reasons,
            "wake_sources": sources,
            "pending_wake_requests": self.pending_wake_requests,
            "pending_provider_reads": self.pending_provider_reads,
            "contains_context_content": False,
            "authorization_boundary": False,
        }
        return output


class AssignmentExpertRunner:
    """Run an expert only while its exact assignment evidence view is awake."""

    def __init__(
        self,
        *,
        engine: RegionExpertEngine,
        tasks: TaskCoordinationBoard,
        workspace: CognitiveWorkspace,
        coordination: RegionCoordinationBoard,
    ) -> None:
        self.engine = engine
        self.tasks = tasks
        self.workspace = workspace
        self.coordination = coordination

    async def run(
        self,
        *,
        task_id: str,
        assignment_id: str,
        model: str,
        endpoint_id: str | None = None,
        max_context_tokens: int = 2000,
        max_blocks: int = 12,
        max_tokens: int = 1200,
        temperature: float = 0.1,
        effort: str | None = None,
        evidence_view: WorkspaceView | None = None,
    ) -> AssignmentExpertResult:
        assignment = self.tasks.assignment(task_id, assignment_id)
        wake_status = self.tasks.evidence_wake_status(task_id, assignment_id)
        if wake_status["count"] == 0:
            return AssignmentExpertResult.sleeping(
                assignment=assignment,
                model=model,
                endpoint_id=endpoint_id,
            )

        if evidence_view is not None and (
            evidence_view.task_id != task_id
            or evidence_view.consumer != "region"
            or evidence_view.region != assignment["region"]
            or evidence_view.assignment_id != assignment_id
        ):
            raise ValueError(
                "assignment evidence view must match task_id, region, and assignment_id"
            )
        view = evidence_view or self.workspace.read(
            task_id,
            consumer="region",
            region=assignment["region"],
            assignment_id=assignment_id,
            max_context_tokens=max_context_tokens,
            max_blocks=max_blocks,
        )
        if not view.blocks:
            expert_result = await self.engine.run(
                workspace=self.workspace,
                coordination=self.coordination,
                task_id=task_id,
                region=assignment["region"],
                task=_assignment_task(assignment),
                model=model,
                assignment_id=assignment_id,
                endpoint_id=endpoint_id,
                max_context_tokens=max_context_tokens,
                max_blocks=max_blocks,
                max_tokens=max_tokens,
                temperature=temperature,
                effort=effort,
                view=view,
            )
            return AssignmentExpertResult(
                state="waiting_context",
                task_id=task_id,
                assignment_id=assignment_id,
                region=assignment["region"],
                model=model,
                endpoint_id=endpoint_id,
                expert_result=expert_result,
                pending_wake_requests=int(wake_status["count"]),
                pending_provider_reads=_pending_provider_reads(wake_status),
            )

        delivered = self.tasks.consume_evidence_wakes(task_id, assignment_id)
        deliveries = tuple(delivered["deliveries"])
        if not deliveries:
            pending = self.tasks.evidence_wake_status(task_id, assignment_id)
            return AssignmentExpertResult.sleeping(
                assignment=assignment,
                model=model,
                endpoint_id=endpoint_id,
                pending_wake_requests=int(pending["count"]),
                pending_provider_reads=_pending_provider_reads(pending),
            )

        expert_result = await self.engine.run(
            workspace=self.workspace,
            coordination=self.coordination,
            task_id=task_id,
            region=assignment["region"],
            task=_assignment_task(assignment),
            model=model,
            assignment_id=assignment_id,
            endpoint_id=endpoint_id,
            max_context_tokens=max_context_tokens,
            max_blocks=max_blocks,
            max_tokens=max_tokens,
            temperature=temperature,
            effort=effort,
            view=view,
        )
        pending = self.tasks.evidence_wake_status(task_id, assignment_id)
        return AssignmentExpertResult(
            state="awake",
            task_id=task_id,
            assignment_id=assignment_id,
            region=assignment["region"],
            model=model,
            endpoint_id=endpoint_id,
            expert_result=expert_result,
            wake_deliveries=deliveries,
            pending_wake_requests=int(pending["count"]),
            pending_provider_reads=_pending_provider_reads(pending),
        )


__all__ = [
    "AssignmentExpertResult",
    "AssignmentExpertRunner",
    "AssignmentExpertState",
]
