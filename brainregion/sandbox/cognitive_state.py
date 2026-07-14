"""Compact, evidence-linked state for main-brain work across model turns."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

_HYPOTHESIS_STATES = frozenset({"open", "supported", "rejected"})
_ATTEMPT_OUTCOMES = frozenset({"unknown", "failed", "succeeded"})
_CHECKPOINT_REASONS = frozenset(
    {"periodic", "repeated_target", "tool_error", "verification_failed", "workspace_effect"}
)
_STRATEGIC_UPDATE_FIELDS = {
    "current_subgoal",
    "hypotheses_upsert",
    "blocker",
    "next_action",
    "verification_gap",
}


def _text(value: Any, name: str, *, max_length: int, required: bool = False) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise ValueError(f"{name} cannot be empty")
    if len(text) > max_length:
        raise ValueError(f"{name} cannot exceed {max_length} characters")
    return text


def _strict_fields(data: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{name} unknown field(s): {sorted(unknown)}")


def _evidence_refs(value: Any, name: str, valid_refs: set[str], *, required: bool) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    refs = tuple(dict.fromkeys(_text(item, name, max_length=200, required=True) for item in value))
    if required and not refs:
        raise ValueError(f"{name} cannot be empty")
    invalid = sorted(set(refs) - valid_refs)
    if invalid:
        raise ValueError(f"{name} contains unavailable reference(s): {invalid}")
    return refs


@dataclass(frozen=True)
class CognitiveFact:
    fact_id: str
    statement: str
    evidence_refs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "statement": self.statement,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class CognitiveHypothesis:
    hypothesis_id: str
    statement: str
    status: str
    evidence_refs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "status": self.status,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class CognitiveAttempt:
    summary: str
    outcome: str
    evidence_refs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "outcome": self.outcome,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class MainCognitiveState:
    revision: int = 0
    current_subgoal: str = "Inspect the workspace and identify a grounded next action."
    facts: tuple[CognitiveFact, ...] = ()
    hypotheses: tuple[CognitiveHypothesis, ...] = ()
    attempts: tuple[CognitiveAttempt, ...] = ()
    blocker: str = ""
    next_action: str = "Inspect relevant source and tests."
    verification_gap: str = "Objective checks have not passed yet."
    update_attempts: int = 0
    update_failures: int = 0
    update_errors: tuple[str, ...] = field(default_factory=tuple)

    def apply_update(self, raw: dict[str, Any], *, valid_evidence_refs: set[str]) -> "MainCognitiveState":
        if not isinstance(raw, dict):
            raise ValueError("cognitive_update must be an object")
        _strict_fields(
            raw,
            {
                "current_subgoal",
                "facts_upsert",
                "facts_remove",
                "hypotheses_upsert",
                "attempts_add",
                "blocker",
                "next_action",
                "verification_gap",
            },
            "cognitive_update",
        )
        facts = {fact.fact_id: fact for fact in self.facts}
        hypotheses = {item.hypothesis_id: item for item in self.hypotheses}
        attempts = list(self.attempts)

        for fact_id in _id_list(raw.get("facts_remove", []), "facts_remove"):
            facts.pop(fact_id, None)
        for item in _object_list(raw.get("facts_upsert", []), "facts_upsert", max_items=8):
            _strict_fields(item, {"fact_id", "statement", "evidence_refs"}, "fact")
            fact = CognitiveFact(
                fact_id=_text(item.get("fact_id"), "fact_id", max_length=80, required=True),
                statement=_text(item.get("statement"), "fact statement", max_length=400, required=True),
                evidence_refs=_evidence_refs(
                    item.get("evidence_refs"), "fact evidence_refs", valid_evidence_refs, required=True
                ),
            )
            facts[fact.fact_id] = fact
        for item in _object_list(raw.get("hypotheses_upsert", []), "hypotheses_upsert", max_items=6):
            _strict_fields(
                item,
                {"hypothesis_id", "statement", "status", "evidence_refs"},
                "hypothesis",
            )
            status = _text(item.get("status"), "hypothesis status", max_length=20, required=True)
            if status not in _HYPOTHESIS_STATES:
                raise ValueError(f"hypothesis status must be one of {sorted(_HYPOTHESIS_STATES)}")
            hypothesis = CognitiveHypothesis(
                hypothesis_id=_text(
                    item.get("hypothesis_id"), "hypothesis_id", max_length=80, required=True
                ),
                statement=_text(
                    item.get("statement"), "hypothesis statement", max_length=400, required=True
                ),
                status=status,
                evidence_refs=_evidence_refs(
                    item.get("evidence_refs", []),
                    "hypothesis evidence_refs",
                    valid_evidence_refs,
                    required=False,
                ),
            )
            hypotheses[hypothesis.hypothesis_id] = hypothesis
        for item in _object_list(raw.get("attempts_add", []), "attempts_add", max_items=4):
            _strict_fields(item, {"summary", "outcome", "evidence_refs"}, "attempt")
            outcome = _text(item.get("outcome"), "attempt outcome", max_length=20, required=True)
            if outcome not in _ATTEMPT_OUTCOMES:
                raise ValueError(f"attempt outcome must be one of {sorted(_ATTEMPT_OUTCOMES)}")
            attempts.append(
                CognitiveAttempt(
                    summary=_text(item.get("summary"), "attempt summary", max_length=400, required=True),
                    outcome=outcome,
                    evidence_refs=_evidence_refs(
                        item.get("evidence_refs", []),
                        "attempt evidence_refs",
                        valid_evidence_refs,
                        required=False,
                    ),
                )
            )

        if len(facts) > 8:
            raise ValueError("cognitive state cannot contain more than 8 facts")
        if len(hypotheses) > 6:
            raise ValueError("cognitive state cannot contain more than 6 hypotheses")
        attempts = attempts[-6:]
        return replace(
            self,
            revision=self.revision + 1,
            current_subgoal=_optional_update(
                raw, "current_subgoal", self.current_subgoal, max_length=400
            ),
            facts=tuple(facts.values()),
            hypotheses=tuple(hypotheses.values()),
            attempts=tuple(attempts),
            blocker=_optional_update(raw, "blocker", self.blocker, max_length=400),
            next_action=_optional_update(raw, "next_action", self.next_action, max_length=400),
            verification_gap=_optional_update(
                raw, "verification_gap", self.verification_gap, max_length=400
            ),
            update_attempts=self.update_attempts + 1,
        )

    def record_failed_update(self, error: str) -> "MainCognitiveState":
        bounded = _text(error, "cognitive update error", max_length=300, required=True)
        return replace(
            self,
            update_attempts=self.update_attempts + 1,
            update_failures=self.update_failures + 1,
            update_errors=(*self.update_errors[-4:], bounded),
        )

    def apply_strategic_update(
        self, raw: dict[str, Any], *, valid_evidence_refs: set[str]
    ) -> "MainCognitiveState":
        if not isinstance(raw, dict):
            raise ValueError("cognitive_update must be an object")
        _strict_fields(raw, _STRATEGIC_UPDATE_FIELDS, "strategic cognitive_update")
        return self.apply_update(raw, valid_evidence_refs=valid_evidence_refs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "current_subgoal": self.current_subgoal,
            "facts": [fact.to_dict() for fact in self.facts],
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "blocker": self.blocker,
            "next_action": self.next_action,
            "verification_gap": self.verification_gap,
            "update_attempts": self.update_attempts,
            "update_failures": self.update_failures,
        }

    def public_metrics(self) -> dict[str, Any]:
        error_categories: dict[str, int] = {}
        for error in self.update_errors:
            category = _error_category(error)
            error_categories[category] = error_categories.get(category, 0) + 1
        return {
            "enabled": True,
            "mode": "model_managed",
            "revision": self.revision,
            "facts": len(self.facts),
            "hypotheses": len(self.hypotheses),
            "open_hypotheses": sum(item.status == "open" for item in self.hypotheses),
            "attempts": len(self.attempts),
            "update_attempts": self.update_attempts,
            "update_failures": self.update_failures,
            "recent_update_error_categories": error_categories,
            "has_blocker": bool(self.blocker),
            "has_verification_gap": bool(self.verification_gap),
            "contains_state_content": False,
            "contains_reasoning": False,
        }


@dataclass(frozen=True)
class RuntimeCognitiveEvent:
    step: int
    operation: str
    target_kind: str
    target_label: str
    target_fingerprint: str
    target_is_new: bool
    workspace_effect: bool
    verification_passed: bool | None
    error: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "operation": self.operation,
            "target_kind": self.target_kind,
            "target_label": self.target_label,
            "target_fingerprint": self.target_fingerprint,
            "target_is_new": self.target_is_new,
            "workspace_effect": self.workspace_effect,
            "verification_passed": self.verification_passed,
            "error": self.error,
        }


@dataclass(frozen=True)
class RuntimeCognitiveState:
    """Objective tool-event state plus sparse model-authored strategic checkpoints."""

    revision: int = 0
    recent_events: tuple[RuntimeCognitiveEvent, ...] = ()
    workspace_effects: int = 0
    verification_runs: int = 0
    last_verification_passed: bool | None = None
    pending_verification: bool = False
    tool_errors: int = 0
    last_workspace_effect_revision: int | None = None
    last_checkpoint_revision: int = 0
    checkpoint_count: int = 0
    checkpoint_reason_counts: tuple[tuple[str, int], ...] = ()
    strategy: MainCognitiveState = field(default_factory=MainCognitiveState)

    def observe(
        self,
        *,
        step: int,
        operation: str,
        target_kind: str = "",
        target_label: str = "",
        target_fingerprint: str = "",
        target_is_new: bool = False,
        workspace_effect: bool = False,
        verification_passed: bool | None = None,
        error: bool = False,
    ) -> "RuntimeCognitiveState":
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("runtime cognitive event step must be a non-negative integer")
        operation = _text(operation, "runtime operation", max_length=80, required=True)
        target_kind = _text(target_kind, "runtime target_kind", max_length=40)
        target_label = _text(target_label, "runtime target_label", max_length=300)
        target_fingerprint = _text(
            target_fingerprint, "runtime target_fingerprint", max_length=80
        )
        revision = self.revision + 1
        event = RuntimeCognitiveEvent(
            step=step,
            operation=operation,
            target_kind=target_kind,
            target_label=target_label,
            target_fingerprint=target_fingerprint,
            target_is_new=bool(target_is_new),
            workspace_effect=bool(workspace_effect),
            verification_passed=verification_passed,
            error=bool(error),
        )
        return replace(
            self,
            revision=revision,
            recent_events=(*self.recent_events, event)[-6:],
            workspace_effects=self.workspace_effects + int(workspace_effect),
            verification_runs=self.verification_runs + int(verification_passed is not None),
            last_verification_passed=(
                verification_passed
                if verification_passed is not None
                else self.last_verification_passed
            ),
            pending_verification=(
                True
                if workspace_effect
                else False
                if verification_passed is not None
                else self.pending_verification
            ),
            tool_errors=self.tool_errors + int(error),
            last_workspace_effect_revision=(
                revision if workspace_effect else self.last_workspace_effect_revision
            ),
        )

    def checkpoint_reason(self, *, period: int = 3, min_interval: int = 2) -> str | None:
        if isinstance(period, bool) or not isinstance(period, int) or period <= 0:
            raise ValueError("checkpoint period must be a positive integer")
        if isinstance(min_interval, bool) or not isinstance(min_interval, int) or min_interval <= 0:
            raise ValueError("checkpoint min_interval must be a positive integer")
        distance = self.revision - self.last_checkpoint_revision
        if distance <= 0 or not self.recent_events:
            return None
        event = self.recent_events[-1]
        if event.verification_passed is False:
            return "verification_failed"
        if event.error:
            return "tool_error"
        if distance >= min_interval and event.target_fingerprint and not event.target_is_new:
            return "repeated_target"
        if distance >= min_interval and event.workspace_effect:
            return "workspace_effect"
        if distance >= period:
            return "periodic"
        return None

    def complete_checkpoint(
        self,
        reason: str,
        raw_update: dict[str, Any] | None,
        *,
        valid_evidence_refs: set[str],
    ) -> tuple["RuntimeCognitiveState", str | None]:
        if reason not in _CHECKPOINT_REASONS:
            raise ValueError(f"unknown runtime checkpoint reason: {reason!r}")
        error: str | None = None
        if raw_update is None:
            error = "missing cognitive_update at runtime checkpoint"
            strategy = self.strategy.record_failed_update(error)
        else:
            try:
                strategy = self.strategy.apply_strategic_update(
                    raw_update,
                    valid_evidence_refs=valid_evidence_refs,
                )
            except ValueError as exc:
                error = str(exc)[:300]
                strategy = self.strategy.record_failed_update(error)
        counts = dict(self.checkpoint_reason_counts)
        counts[reason] = counts.get(reason, 0) + 1
        return (
            replace(
                self,
                last_checkpoint_revision=self.revision,
                checkpoint_count=self.checkpoint_count + 1,
                checkpoint_reason_counts=tuple(sorted(counts.items())),
                strategy=strategy,
            ),
            error,
        )

    @property
    def steps_since_workspace_effect(self) -> int:
        if self.last_workspace_effect_revision is None:
            return self.revision
        return self.revision - self.last_workspace_effect_revision

    def prompt_dict(self, *, reason: str) -> dict[str, Any]:
        if reason not in _CHECKPOINT_REASONS:
            raise ValueError(f"unknown runtime checkpoint reason: {reason!r}")
        return {
            "mode": "runtime_checkpoint",
            "checkpoint_reason": reason,
            "objective": {
                "revision": self.revision,
                "completed_actions": self.revision,
                "workspace_effects": self.workspace_effects,
                "verification_runs": self.verification_runs,
                "last_verification_passed": self.last_verification_passed,
                "pending_verification": self.pending_verification,
                "tool_errors": self.tool_errors,
                "steps_since_workspace_effect": self.steps_since_workspace_effect,
                "recent_events": [event.to_dict() for event in self.recent_events[-4:]],
            },
            "strategy": {
                "revision": self.strategy.revision,
                "current_subgoal": self.strategy.current_subgoal,
                "hypotheses": [item.to_dict() for item in self.strategy.hypotheses],
                "blocker": self.strategy.blocker,
                "next_action": self.strategy.next_action,
                "verification_gap": self.strategy.verification_gap,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.prompt_dict(reason="periodic"),
            "checkpoint_reason": None,
            "last_checkpoint_revision": self.last_checkpoint_revision,
            "checkpoint_count": self.checkpoint_count,
            "checkpoint_reason_counts": dict(self.checkpoint_reason_counts),
            "strategy_update_attempts": self.strategy.update_attempts,
            "strategy_update_failures": self.strategy.update_failures,
        }

    def public_metrics(self) -> dict[str, Any]:
        strategy_metrics = self.strategy.public_metrics()
        return {
            "enabled": True,
            "mode": "runtime_checkpoint",
            "objective_revision": self.revision,
            "checkpoint_count": self.checkpoint_count,
            "checkpoint_reason_counts": dict(self.checkpoint_reason_counts),
            "workspace_effects": self.workspace_effects,
            "verification_runs": self.verification_runs,
            "last_verification_passed": self.last_verification_passed,
            "pending_verification": self.pending_verification,
            "tool_errors": self.tool_errors,
            "strategic_revision": self.strategy.revision,
            "hypotheses": len(self.strategy.hypotheses),
            "open_hypotheses": sum(
                item.status == "open" for item in self.strategy.hypotheses
            ),
            "update_attempts": self.strategy.update_attempts,
            "update_failures": self.strategy.update_failures,
            "recent_update_error_categories": strategy_metrics[
                "recent_update_error_categories"
            ],
            "has_blocker": bool(self.strategy.blocker),
            "has_verification_gap": bool(self.strategy.verification_gap),
            "contains_state_content": False,
            "contains_reasoning": False,
        }


def _object_list(value: Any, name: str, *, max_items: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{name} must be an array of objects")
    if len(value) > max_items:
        raise ValueError(f"{name} cannot contain more than {max_items} items")
    return value


def _id_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    if len(value) > 8:
        raise ValueError(f"{name} cannot contain more than 8 items")
    return tuple(_text(item, name, max_length=80, required=True) for item in value)


def _optional_update(raw: dict[str, Any], key: str, current: str, *, max_length: int) -> str:
    return _text(raw[key], key, max_length=max_length) if key in raw else current


def _error_category(error: str) -> str:
    normalized = str(error or "").casefold()
    if normalized.startswith("missing cognitive_update"):
        return "missing_update"
    if "unavailable reference" in normalized:
        return "unavailable_evidence"
    if "unknown field" in normalized:
        return "unknown_field"
    if "must be" in normalized or "cannot be empty" in normalized:
        return "invalid_shape"
    if "cannot exceed" in normalized or "more than" in normalized:
        return "limit_exceeded"
    return "invalid_update"


__all__ = [
    "CognitiveAttempt",
    "CognitiveFact",
    "CognitiveHypothesis",
    "MainCognitiveState",
    "RuntimeCognitiveEvent",
    "RuntimeCognitiveState",
]
