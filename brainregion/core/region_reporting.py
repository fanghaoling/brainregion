"""Observable region context receipts, reports, and deterministic escalation.

The coordination board contains no private ContextBlock contents and no model
chain of thought. It records retrieval coverage, grounded region conclusions,
and whether the runtime should continue, request context, or notify the main
brain.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

from .context_loader import ActivatedContext

ContextState = Literal["ready", "partial", "insufficient", "conflicted", "stale"]
ReportState = Literal["working", "done", "blocked", "needs_decision"]
DecisionScope = Literal["routine", "task", "architecture", "user", "cross_region"]
RiskLevel = Literal["low", "medium", "high"]
MemoryImpact = Literal["none", "supporting", "decision_changing", "contradictory"]
EscalationAction = Literal["continue", "request_context", "notify_main"]

_CONTEXT_STATES = frozenset({"ready", "partial", "insufficient", "conflicted", "stale"})
_REPORT_STATES = frozenset({"working", "done", "blocked", "needs_decision"})
_DECISION_SCOPES = frozenset({"routine", "task", "architecture", "user", "cross_region"})
_RISK_LEVELS = frozenset({"low", "medium", "high"})
_MEMORY_IMPACTS = frozenset({"none", "supporting", "decision_changing", "contradictory"})


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


def _choice(value: Any, name: str, choices: frozenset[str], default: str) -> str:
    text = str(value or default).strip().casefold()
    if text not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}")
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


def _meta_count(meta: dict[str, Any], *keys: str) -> int:
    total = 0
    for key in keys:
        value = meta.get(key, 0)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value > 0:
            total += int(value)
    return total


@dataclass(frozen=True)
class RegionContextReceipt:
    task_id: str
    region: str
    state: ContextState
    requested_selectors: tuple[str, ...]
    confirmed_selectors: tuple[str, ...]
    selector_coverage: Literal["not_requested", "unverified", "partial", "verified"]
    provider_statuses: tuple[dict[str, Any], ...]
    blocks_loaded: int
    estimated_tokens: int
    evidence_refs: tuple[str, ...]
    stale_candidates_removed: int
    conflicts: int
    warnings: tuple[str, ...]
    assignment_id: str = ""

    @classmethod
    def from_activated(
        cls,
        activated: ActivatedContext,
        *,
        task_id: str,
        region: str,
        evidence_refs: tuple[str, ...] = (),
        assignment_id: str = "",
    ) -> "RegionContextReceipt":
        task_id = _required_text(task_id, "task_id", max_length=200)
        region = _required_text(region, "region", max_length=200).casefold()
        assignment_id = _optional_text(
            assignment_id, "assignment_id", max_length=200
        )
        context_loads = tuple(load for load in activated.loads if load.reason != "activation_mode_not_context")
        requested = tuple(dict.fromkeys(selector for load in context_loads for selector in load.selectors))
        confirmed_values: list[str] = []
        for block in activated.blocks:
            raw = block.metadata.get("selectors", ())
            if isinstance(raw, (list, tuple, set)):
                for selector in raw:
                    text = str(selector or "").strip().casefold()
                    if text and text not in confirmed_values:
                        confirmed_values.append(text)
        confirmed = tuple(confirmed_values)
        if not requested:
            coverage = "not_requested"
        elif not confirmed:
            coverage = "unverified"
        elif set(requested) <= set(confirmed):
            coverage = "verified"
        else:
            coverage = "partial"

        provider_statuses = tuple(
            {
                "skill_id": load.skill_id,
                "provider": load.provider,
                "status": load.status,
                "blocks_loaded": load.blocks_loaded,
                "reason": load.reason,
            }
            for load in context_loads
        )
        stale_removed = sum(
            _meta_count(load.provider_meta, "removed_expired", "stale_candidates_removed") for load in context_loads
        )
        stale_returned = sum(_meta_count(load.provider_meta, "stale_returned") for load in context_loads)
        conflicts = sum(_meta_count(load.provider_meta, "conflicts", "conflicting_records") for load in context_loads)
        loaded = [load for load in context_loads if load.status == "loaded"]
        degraded = [load for load in context_loads if load.status in {"failed", "empty", "skipped"}]
        if conflicts:
            state: ContextState = "conflicted"
        elif stale_returned:
            state = "stale"
        elif not activated.blocks:
            state = "insufficient"
        elif degraded:
            state = "partial"
        elif loaded:
            state = "ready"
        else:
            state = "insufficient"

        warnings: list[str] = []
        if stale_removed:
            warnings.append("stale_candidates_removed")
        if coverage == "unverified":
            warnings.append("selector_coverage_unverified")
        if any(load.status == "failed" for load in context_loads):
            warnings.append("provider_failure_isolated")
        return cls(
            task_id=task_id,
            region=region,
            state=state,
            requested_selectors=requested,
            confirmed_selectors=confirmed,
            selector_coverage=coverage,
            provider_statuses=provider_statuses,
            blocks_loaded=len(activated.blocks),
            estimated_tokens=int(activated.trace.get("estimated_tokens") or 0),
            evidence_refs=tuple(dict.fromkeys(evidence_refs)),
            stale_candidates_removed=stale_removed,
            conflicts=conflicts,
            warnings=tuple(warnings),
            assignment_id=assignment_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "region": self.region,
            "assignment_id": self.assignment_id,
            "state": self.state,
            "requested_selectors": list(self.requested_selectors),
            "confirmed_selectors": list(self.confirmed_selectors),
            "selector_coverage": self.selector_coverage,
            "provider_statuses": [dict(status) for status in self.provider_statuses],
            "blocks_loaded": self.blocks_loaded,
            "estimated_tokens": self.estimated_tokens,
            "evidence_refs": list(self.evidence_refs),
            "stale_candidates_removed": self.stale_candidates_removed,
            "conflicts": self.conflicts,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class RegionReport:
    report_id: str
    task_id: str
    region: str
    state: ReportState
    summary: str
    implication: str
    recommended_action: str
    uncertainty: str
    evidence_refs: tuple[str, ...]
    context_state: ContextState
    decision_scope: DecisionScope
    risk: RiskLevel
    memory_impact: MemoryImpact
    reversible: bool
    repeated_failure: bool
    requires_user_choice: bool
    needs_more_context: bool
    assignment_id: str = ""
    covered_scope: str = ""
    unresolved_questions: tuple[str, ...] = ()
    conflicts_with: tuple[str, ...] = ()
    recommended_followups: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, task_id: str, data: dict[str, Any]) -> "RegionReport":
        if not isinstance(data, dict):
            raise ValueError("report must be an object")
        known = {
            "region",
            "state",
            "summary",
            "implication",
            "recommended_action",
            "uncertainty",
            "evidence_refs",
            "context_state",
            "decision_scope",
            "risk",
            "memory_impact",
            "reversible",
            "repeated_failure",
            "requires_user_choice",
            "needs_more_context",
            "assignment_id",
            "covered_scope",
            "unresolved_questions",
            "conflicts_with",
            "recommended_followups",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"report unknown field(s): {sorted(unknown)}")
        for name in (
            "reversible",
            "repeated_failure",
            "requires_user_choice",
            "needs_more_context",
        ):
            if name in data and not isinstance(data[name], bool):
                raise ValueError(f"report {name} must be a boolean")
        return cls(
            report_id=f"report-{uuid4().hex[:12]}",
            task_id=_required_text(task_id, "task_id", max_length=200),
            region=_required_text(data.get("region"), "region", max_length=200).casefold(),
            state=_choice(data.get("state"), "state", _REPORT_STATES, "working"),  # type: ignore[arg-type]
            summary=_required_text(data.get("summary"), "summary"),
            implication=_optional_text(data.get("implication"), "implication"),
            recommended_action=_optional_text(data.get("recommended_action"), "recommended_action"),
            uncertainty=_optional_text(data.get("uncertainty"), "uncertainty"),
            evidence_refs=_string_tuple(data.get("evidence_refs"), "evidence_refs"),
            context_state=_choice(data.get("context_state"), "context_state", _CONTEXT_STATES, "ready"),  # type: ignore[arg-type]
            decision_scope=_choice(data.get("decision_scope"), "decision_scope", _DECISION_SCOPES, "routine"),  # type: ignore[arg-type]
            risk=_choice(data.get("risk"), "risk", _RISK_LEVELS, "low"),  # type: ignore[arg-type]
            memory_impact=_choice(data.get("memory_impact"), "memory_impact", _MEMORY_IMPACTS, "none"),  # type: ignore[arg-type]
            reversible=data.get("reversible", True),
            repeated_failure=data.get("repeated_failure", False),
            requires_user_choice=data.get("requires_user_choice", False),
            needs_more_context=data.get("needs_more_context", False),
            assignment_id=_optional_text(
                data.get("assignment_id"), "assignment_id", max_length=200
            ),
            covered_scope=_optional_text(data.get("covered_scope"), "covered_scope"),
            unresolved_questions=_string_tuple(
                data.get("unresolved_questions"), "unresolved_questions"
            ),
            conflicts_with=_string_tuple(data.get("conflicts_with"), "conflicts_with"),
            recommended_followups=_string_tuple(
                data.get("recommended_followups"), "recommended_followups"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "task_id": self.task_id,
            "region": self.region,
            "assignment_id": self.assignment_id,
            "state": self.state,
            "summary": self.summary,
            "implication": self.implication,
            "recommended_action": self.recommended_action,
            "uncertainty": self.uncertainty,
            "evidence_refs": list(self.evidence_refs),
            "context_state": self.context_state,
            "decision_scope": self.decision_scope,
            "risk": self.risk,
            "memory_impact": self.memory_impact,
            "reversible": self.reversible,
            "repeated_failure": self.repeated_failure,
            "requires_user_choice": self.requires_user_choice,
            "needs_more_context": self.needs_more_context,
            "covered_scope": self.covered_scope,
            "unresolved_questions": list(self.unresolved_questions),
            "conflicts_with": list(self.conflicts_with),
            "recommended_followups": list(self.recommended_followups),
        }


@dataclass(frozen=True)
class EscalationDecision:
    action: EscalationAction
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "reasons": list(self.reasons)}


class EscalationPolicy:
    """Deterministic policy: routine autonomy, explicit context requests, hard escalation."""

    @staticmethod
    def evaluate(report: RegionReport) -> EscalationDecision:
        reasons: list[str] = []
        if report.state in {"blocked", "needs_decision"}:
            reasons.append(f"region_{report.state}")
        if report.memory_impact in {"decision_changing", "contradictory"}:
            reasons.append(f"memory_{report.memory_impact}")
        if report.context_state in {"conflicted", "stale"}:
            reasons.append(f"context_{report.context_state}")
        if report.decision_scope in {"architecture", "user", "cross_region"}:
            reasons.append(f"scope_{report.decision_scope}")
        if report.risk == "high":
            reasons.append("risk_high")
        if not report.reversible:
            reasons.append("irreversible_action")
        if report.repeated_failure:
            reasons.append("repeated_failure")
        if report.requires_user_choice:
            reasons.append("requires_user_choice")
        if reasons:
            return EscalationDecision("notify_main", tuple(reasons))
        if report.context_state == "insufficient" or report.needs_more_context:
            context_reasons = []
            if report.context_state == "insufficient":
                context_reasons.append("context_insufficient")
            if report.needs_more_context:
                context_reasons.append("expert_requested_more_context")
            return EscalationDecision("request_context", tuple(context_reasons))
        return EscalationDecision("continue", ("within_delegated_scope",))


@dataclass(frozen=True)
class PublishedRegionReport:
    report: RegionReport
    decision: EscalationDecision

    def to_dict(self) -> dict[str, Any]:
        return {"report": self.report.to_dict(), "decision": self.decision.to_dict()}


class RegionCoordinationBoard:
    """Thread-safe task board for receipts and region reports, never private context."""

    def __init__(self, *, max_reports: int = 512) -> None:
        if isinstance(max_reports, bool) or not isinstance(max_reports, int) or max_reports <= 0:
            raise ValueError("max_reports must be a positive integer")
        self._max_reports = max_reports
        self._receipts: dict[str, dict[tuple[str, str], RegionContextReceipt]] = {}
        self._reports: dict[str, list[PublishedRegionReport]] = {}
        self._lock = RLock()

    def record_receipt(self, receipt: RegionContextReceipt) -> dict[str, Any]:
        with self._lock:
            self._receipts.setdefault(receipt.task_id, {})[
                (receipt.region, receipt.assignment_id)
            ] = receipt
        return receipt.to_dict()

    def publish(self, task_id: str, report_data: dict[str, Any]) -> dict[str, Any]:
        report = RegionReport.from_dict(task_id, report_data)
        published = PublishedRegionReport(report, EscalationPolicy.evaluate(report))
        with self._lock:
            total = sum(len(reports) for reports in self._reports.values())
            if total >= self._max_reports:
                raise RuntimeError("region report capacity exceeded")
            self._reports.setdefault(report.task_id, []).append(published)
        return published.to_dict()

    def reports(
        self, task_id: str, *, assignment_id: str | None = None
    ) -> dict[str, Any]:
        """Return validated public reports, optionally for one assignment."""
        task_id = _required_text(task_id, "task_id", max_length=200)
        if assignment_id is not None:
            assignment_id = _optional_text(
                assignment_id, "assignment_id", max_length=200
            )
        with self._lock:
            reports = list(self._reports.get(task_id, ()))
        if assignment_id is not None:
            reports = [
                report for report in reports
                if report.report.assignment_id == assignment_id
            ]
        return {
            "task_id": task_id,
            "assignment_id": assignment_id,
            "reports": [report.to_dict() for report in reports],
            "count": len(reports),
            "contains_private_context": False,
        }

    def status(self, task_id: str) -> dict[str, Any]:
        task_id = _required_text(task_id, "task_id", max_length=200)
        with self._lock:
            receipts = list(self._receipts.get(task_id, {}).values())
            reports = list(self._reports.get(task_id, ()))
        latest_by_region: dict[tuple[str, str], PublishedRegionReport] = {}
        for report in reports:
            latest_by_region[(report.report.region, report.report.assignment_id)] = report
        return {
            "task_id": task_id,
            "context_receipts": [receipt.to_dict() for receipt in receipts],
            "region_statuses": [
                {
                    "region": published.report.region,
                    "assignment_id": published.report.assignment_id,
                    "state": published.report.state,
                    "context_state": published.report.context_state,
                    "decision": published.decision.to_dict(),
                    "needs_main_attention": published.decision.action == "notify_main",
                }
                for published in latest_by_region.values()
            ],
            "contains_private_context": False,
        }

    def inbox(self, task_id: str) -> dict[str, Any]:
        task_id = _required_text(task_id, "task_id", max_length=200)
        with self._lock:
            reports = list(self._reports.get(task_id, ()))
        escalated = [report for report in reports if report.decision.action == "notify_main"]
        return {
            "task_id": task_id,
            "reports": [report.to_dict() for report in escalated],
            "count": len(escalated),
            "contains_private_context": False,
        }

    def clear(
        self, task_id: str, *, assignment_id: str | None = None
    ) -> dict[str, Any]:
        task_id = _required_text(task_id, "task_id", max_length=200)
        if assignment_id is not None:
            assignment_id = _optional_text(
                assignment_id, "assignment_id", max_length=200
            )
        with self._lock:
            if assignment_id is None:
                removed_receipts = len(self._receipts.pop(task_id, {}))
                removed_reports = len(self._reports.pop(task_id, ()))
            else:
                receipts = self._receipts.get(task_id, {})
                removed_receipts = sum(
                    1 for key in receipts if key[1] == assignment_id
                )
                retained_receipts = {
                    key: receipt for key, receipt in receipts.items()
                    if key[1] != assignment_id
                }
                reports = self._reports.get(task_id, [])
                retained_reports = [
                    report for report in reports
                    if report.report.assignment_id != assignment_id
                ]
                removed_reports = len(reports) - len(retained_reports)
                if retained_receipts:
                    self._receipts[task_id] = retained_receipts
                else:
                    self._receipts.pop(task_id, None)
                if retained_reports:
                    self._reports[task_id] = retained_reports
                else:
                    self._reports.pop(task_id, None)
        return {
            "task_id": task_id,
            "assignment_id": assignment_id,
            "removed_receipts": removed_receipts,
            "removed_reports": removed_reports,
        }


__all__ = [
    "EscalationDecision",
    "EscalationPolicy",
    "PublishedRegionReport",
    "RegionContextReceipt",
    "RegionCoordinationBoard",
    "RegionReport",
]
