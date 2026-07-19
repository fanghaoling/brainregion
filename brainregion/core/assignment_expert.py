"""Wake-gated expert runner for one exact task assignment.

The runner owns the delivery lifecycle around an existing RegionExpertEngine.
It never stores private context: evidence remains in CognitiveWorkspace, wake
metadata remains in TaskCoordinationBoard, and only a validated RegionReport
crosses back to the main-brain surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .activation import ActivationPlan
from .context import ContextProvider, ContextQuery
from .context_loader import ActivatedContext, ContextLoadRecord, fit_context_blocks
from .cognitive_workspace import CognitiveWorkspace, WorkspaceView
from .region_expert import RegionExpertEngine, RegionExpertResult
from .region_reporting import RegionContextReceipt, RegionCoordinationBoard
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


def _retrieval_trace(
    status: str = "disabled",
    *,
    provider: str = "",
    blocks_staged: int = 0,
    estimated_tokens: int = 0,
    truncated: bool = False,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "status": status,
        "provider": provider,
        "blocks_staged": blocks_staged,
        "estimated_tokens": estimated_tokens,
        "truncated": truncated,
        "reason": reason,
        "contains_context_content": False,
        "models_called": False,
    }


def _has_memory_request(assignment: dict[str, Any]) -> bool:
    request = assignment.get("memory_request") or {}
    return bool(
        str(request.get("query") or "").strip()
        or str(request.get("purpose") or "").strip()
        or request.get("regions")
        or request.get("selectors")
    )


@dataclass(frozen=True)
class AssignmentContextPreparation:
    task_id: str
    assignment_id: str
    region: str
    view: WorkspaceView
    context_retrieval: dict[str, Any] = field(default_factory=_retrieval_trace)


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
    context_retrieval: dict[str, Any] = field(default_factory=_retrieval_trace)

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
                    "truncated": False,
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
        output["context_retrieval"] = dict(self.context_retrieval)
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
        engine: RegionExpertEngine | None,
        tasks: TaskCoordinationBoard,
        workspace: CognitiveWorkspace,
        coordination: RegionCoordinationBoard,
        memory_provider: ContextProvider | None = None,
    ) -> None:
        self.engine = engine
        self.tasks = tasks
        self.workspace = workspace
        self.coordination = coordination
        self.memory_provider = memory_provider

    def _retrieve_private_context(
        self,
        *,
        assignment: dict[str, Any],
        max_context_tokens: int,
        max_blocks: int,
    ) -> dict[str, Any]:
        if self.memory_provider is None:
            return _retrieval_trace()
        if not _has_memory_request(assignment):
            return _retrieval_trace("skipped", reason="memory_request_empty")

        request = assignment["memory_request"]
        if str(request.get("target_region") or "").casefold() != str(
            assignment["region"]
        ).casefold():
            return _retrieval_trace("skipped", reason="target_region_mismatch")
        query_text = str(request.get("query") or request.get("purpose") or "").strip()
        if not query_text:
            query_text = str(assignment["question"])
        requested_tokens = min(
            max(1, int(request.get("max_context_tokens") or 1600)),
            max(1, int(max_context_tokens)),
        )
        requested_blocks = min(
            max(1, int(request.get("top_k") or 5)),
            max(1, int(max_blocks)),
        )
        query = ContextQuery(
            text=query_text,
            regions=(
                frozenset(str(region).strip().casefold() for region in request["regions"])
                if request.get("regions")
                else frozenset({str(assignment["region"])})
            ),
            top_k=requested_blocks,
            selectors=tuple(str(value) for value in request.get("selectors") or ()),
        )
        try:
            retrieved = self.memory_provider.retrieve(query)
            fitted, used, truncated = fit_context_blocks(
                list(retrieved.blocks),
                max_tokens=requested_tokens,
                max_blocks=requested_blocks,
            )
            load = ContextLoadRecord(
                skill_id="assignment-memory",
                region=str(assignment["region"]),
                status="loaded" if fitted else "empty",
                provider=str(retrieved.provider),
                selectors=query.selectors,
                requested_tokens=requested_tokens,
                estimated_tokens=used,
                blocks_loaded=len(fitted),
                truncated=truncated,
                reason="" if fitted else "provider_returned_no_matching_context",
                provider_meta=dict(retrieved.meta),
            )
        except Exception as exc:  # one retrieval failure must not consume the wake
            fitted = []
            used = 0
            truncated = False
            load = ContextLoadRecord(
                skill_id="assignment-memory",
                region=str(assignment["region"]),
                status="failed",
                selectors=query.selectors,
                requested_tokens=requested_tokens,
                reason=f"provider_error:{type(exc).__name__}",
            )

        activated = ActivatedContext(
            activation=ActivationPlan(
                decisions=(),
                woken_regions=tuple(
                    dict.fromkeys(("memory", str(assignment["region"])))
                ),
                context_requests=(),
                trace={"strategy": "assignment_memory_request_v1", "models_called": False},
            ),
            blocks=tuple(fitted),
            loads=(load,),
            trace={
                "strategy": "assignment_memory_request_v1",
                "models_called": False,
                "blocks_loaded": len(fitted),
                "estimated_tokens": used,
                "truncated": truncated,
            },
        )
        try:
            delivery = self.workspace.stage(
                activated,
                task_id=str(assignment["task_id"]),
                audience="region",
                target_region=str(assignment["region"]),
                assignment_id=str(assignment["assignment_id"]),
            )
            evidence_refs = (
                tuple(delivery.entry["evidence_refs"])
                if delivery.entry is not None
                else ()
            )
            self.coordination.record_receipt(
                RegionContextReceipt.from_activated(
                    activated,
                    task_id=str(assignment["task_id"]),
                    region=str(assignment["region"]),
                    assignment_id=str(assignment["assignment_id"]),
                    evidence_refs=evidence_refs,
                )
            )
        except Exception:  # preserve the wake and avoid a half-staged private delivery
            if fitted:
                self.workspace.clear(
                    str(assignment["task_id"]),
                    assignment_id=str(assignment["assignment_id"]),
                )
            return _retrieval_trace(
                "failed",
                provider=load.provider,
                reason="private_stage_failed",
            )
        return _retrieval_trace(
            load.status,
            provider=load.provider,
            blocks_staged=len(fitted),
            estimated_tokens=used,
            truncated=truncated,
            reason=("provider_error" if load.status == "failed" else load.reason),
        )

    def prepare_context(
        self,
        *,
        task_id: str,
        assignment_id: str,
        max_context_tokens: int = 2000,
        max_blocks: int = 12,
        evidence_view: WorkspaceView | None = None,
    ) -> AssignmentContextPreparation:
        """Prepare one exact private view without consuming wake or calling a model."""
        assignment = self.tasks.assignment(task_id, assignment_id)
        wake_status = self.tasks.evidence_wake_status(task_id, assignment_id)
        if wake_status["count"] == 0:
            raise RuntimeError("cannot prepare context for a sleeping assignment")
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
        context_retrieval = _retrieval_trace()
        if not view.blocks:
            context_retrieval = self._retrieve_private_context(
                assignment=assignment,
                max_context_tokens=max_context_tokens,
                max_blocks=max_blocks,
            )
            if context_retrieval["status"] == "loaded":
                view = self.workspace.read(
                    task_id,
                    consumer="region",
                    region=assignment["region"],
                    assignment_id=assignment_id,
                    max_context_tokens=max_context_tokens,
                    max_blocks=max_blocks,
                )
        return AssignmentContextPreparation(
            task_id=task_id,
            assignment_id=assignment_id,
            region=str(assignment["region"]),
            view=view,
            context_retrieval=context_retrieval,
        )

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
        context_preparation: AssignmentContextPreparation | None = None,
    ) -> AssignmentExpertResult:
        assignment = self.tasks.assignment(task_id, assignment_id)
        wake_status = self.tasks.evidence_wake_status(task_id, assignment_id)
        if wake_status["count"] == 0:
            return AssignmentExpertResult.sleeping(
                assignment=assignment,
                model=model,
                endpoint_id=endpoint_id,
            )
        if self.engine is None:
            raise RuntimeError("expert engine is required to run an awake assignment")

        if context_preparation is not None and evidence_view is not None:
            raise ValueError("context_preparation and evidence_view are mutually exclusive")
        preparation = context_preparation or self.prepare_context(
            task_id=task_id,
            assignment_id=assignment_id,
            max_context_tokens=max_context_tokens,
            max_blocks=max_blocks,
            evidence_view=evidence_view,
        )
        if (
            preparation.task_id != task_id
            or preparation.assignment_id != assignment_id
            or preparation.region != assignment["region"]
            or preparation.view.task_id != task_id
            or preparation.view.consumer != "region"
            or preparation.view.region != assignment["region"]
            or preparation.view.assignment_id != assignment_id
        ):
            raise ValueError(
                "assignment context preparation must match task_id, region, and assignment_id"
            )
        view = preparation.view
        context_retrieval = dict(preparation.context_retrieval)
        if not view.blocks:
            self.tasks.set_assignment_status(task_id, assignment_id, "working")
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
            self._sync_result_status(task_id, assignment_id, expert_result)
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
                context_retrieval=context_retrieval,
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

        self.tasks.set_assignment_status(task_id, assignment_id, "working")
        try:
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
        except Exception:
            self.tasks.set_assignment_status(task_id, assignment_id, "blocked")
            raise
        self._sync_result_status(task_id, assignment_id, expert_result)
        pending = self.tasks.evidence_wake_status(task_id, assignment_id)
        return AssignmentExpertResult(
            state="awake" if expert_result.ok else "blocked",
            task_id=task_id,
            assignment_id=assignment_id,
            region=assignment["region"],
            model=model,
            endpoint_id=endpoint_id,
            expert_result=expert_result,
            wake_deliveries=deliveries,
            pending_wake_requests=int(pending["count"]),
            pending_provider_reads=_pending_provider_reads(pending),
            context_retrieval=context_retrieval,
        )

    def _sync_result_status(
        self,
        task_id: str,
        assignment_id: str,
        result: RegionExpertResult,
    ) -> None:
        published = result.published_report or {}
        report = published.get("report") or {}
        if report.get("state"):
            self.tasks.apply_assignment_report(
                task_id, assignment_id, str(report["state"])
            )
        elif not result.ok:
            self.tasks.set_assignment_status(task_id, assignment_id, "blocked")


__all__ = [
    "AssignmentContextPreparation",
    "AssignmentExpertResult",
    "AssignmentExpertRunner",
    "AssignmentExpertState",
]
