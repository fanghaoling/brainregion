"""Episode-local, evidence-gated hypothesis ledger for ARC experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


_LIVE_STATUSES = frozenset({"open", "candidate", "supported"})
_FRAME_CHANGE_VALUES = frozenset({"changed", "unchanged"})
_STATE_VALUES = frozenset({"", "NOT_FINISHED", "WIN", "GAME_OVER"})


def _bounded_text(value: Any, name: str, *, max_length: int, required: bool = True) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise ValueError(f"{name} cannot be empty")
    if len(text) > max_length:
        raise ValueError(f"{name} cannot exceed {max_length} characters")
    return text


def _strict_fields(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{name} contains unknown field(s): {unknown}")


@dataclass(frozen=True)
class PredictionSpec:
    frame_change: str
    level_delta: int
    state: str = ""

    @classmethod
    def from_dict(cls, raw: Any) -> "PredictionSpec":
        if not isinstance(raw, dict):
            raise ValueError("epistemic predicts must be an object")
        _strict_fields(raw, {"frame_change", "level_delta", "state"}, "epistemic predicts")
        frame_change = _bounded_text(
            raw.get("frame_change"), "epistemic predicts.frame_change", max_length=20
        ).lower()
        if frame_change not in _FRAME_CHANGE_VALUES:
            raise ValueError(
                "epistemic predicts.frame_change must be 'changed' or 'unchanged'"
            )
        level_delta = raw.get("level_delta")
        if isinstance(level_delta, bool) or not isinstance(level_delta, int):
            raise ValueError("epistemic predicts.level_delta must be an integer")
        if not 0 <= level_delta <= 100:
            raise ValueError("epistemic predicts.level_delta must be between 0 and 100")
        state = _bounded_text(
            raw.get("state", ""),
            "epistemic predicts.state",
            max_length=20,
            required=False,
        ).upper()
        if state not in _STATE_VALUES:
            raise ValueError(
                "epistemic predicts.state must be empty, NOT_FINISHED, WIN, or GAME_OVER"
            )
        return cls(frame_change=frame_change, level_delta=level_delta, state=state)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_change": self.frame_change,
            "level_delta": self.level_delta,
            "state": self.state,
        }


@dataclass(frozen=True)
class PreparedPrediction:
    hypothesis_id: str
    rule: str
    scope: str
    replaces: str
    action: str
    predicts: PredictionSpec
    is_new: bool


@dataclass
class EpistemicHypothesis:
    hypothesis_id: str
    rule: str
    scope: str
    status: str = "open"
    support_count: int = 0
    contradiction_count: int = 0
    replaces: str = ""
    superseded_by: str = ""
    insight_candidate: bool = False

    @property
    def fingerprint(self) -> str:
        payload = f"{self.hypothesis_id}\0{self.scope}\0{self.rule}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class EpistemicLedger:
    """Keep hypotheses local to one episode and promote only by observed predictions."""

    def __init__(self, *, verification_threshold: int = 2, max_hypotheses: int = 6) -> None:
        if verification_threshold < 2:
            raise ValueError("verification_threshold must be at least 2")
        if max_hypotheses < 1:
            raise ValueError("max_hypotheses must be positive")
        self.verification_threshold = int(verification_threshold)
        self.max_hypotheses = int(max_hypotheses)
        self.hypotheses: dict[str, EpistemicHypothesis] = {}
        self.predictions = 0
        self.predictions_matched = 0
        self.surprises = 0
        self.insight_candidates = 0
        self.verified_insights = 0
        self.false_insights = 0
        self.supersessions = 0
        self._last_evaluation: dict[str, Any] | None = None

    def reset(self) -> None:
        self.hypotheses.clear()
        self.predictions = 0
        self.predictions_matched = 0
        self.surprises = 0
        self.insight_candidates = 0
        self.verified_insights = 0
        self.false_insights = 0
        self.supersessions = 0
        self._last_evaluation = None

    def prepare(self, raw: Any, *, action: str) -> PreparedPrediction:
        """Validate a model-authored claim without mutating the ledger."""

        if not isinstance(raw, dict):
            raise ValueError("act args.epistemic must be an object when the ledger is enabled")
        _strict_fields(
            raw,
            {"hypothesis_id", "rule", "scope", "replaces", "predicts"},
            "epistemic update",
        )
        hypothesis_id = _bounded_text(
            raw.get("hypothesis_id"), "epistemic hypothesis_id", max_length=80
        )
        rule = _bounded_text(raw.get("rule"), "epistemic rule", max_length=400)
        scope = _bounded_text(raw.get("scope"), "epistemic scope", max_length=120)
        replaces = _bounded_text(
            raw.get("replaces", ""), "epistemic replaces", max_length=80, required=False
        )
        normalized_action = _bounded_text(action, "epistemic action", max_length=80).lower()
        predicts = PredictionSpec.from_dict(raw.get("predicts"))

        existing = self.hypotheses.get(hypothesis_id)
        if existing is not None:
            if existing.rule != rule or existing.scope != scope:
                raise ValueError(
                    "an existing epistemic hypothesis_id cannot silently change rule or scope"
                )
            if existing.status not in _LIVE_STATUSES:
                raise ValueError(
                    "a refuted or superseded hypothesis requires a new hypothesis_id"
                )
            if replaces and replaces != existing.replaces:
                raise ValueError("an existing epistemic hypothesis cannot change replaces")
        elif len(self.hypotheses) >= self.max_hypotheses:
            raise ValueError("epistemic ledger hypothesis limit reached")

        if replaces:
            if replaces == hypothesis_id:
                raise ValueError("an epistemic hypothesis cannot replace itself")
            if replaces not in self.hypotheses:
                raise ValueError("epistemic replaces must reference an existing hypothesis")

        return PreparedPrediction(
            hypothesis_id=hypothesis_id,
            rule=rule,
            scope=scope,
            replaces=replaces,
            action=normalized_action,
            predicts=predicts,
            is_new=existing is None,
        )

    def resolve(
        self,
        prepared: PreparedPrediction,
        *,
        frame_changed: bool,
        level_delta: int,
        state: str,
    ) -> dict[str, Any]:
        """Commit one successful environment transition and verify its prediction."""

        hypothesis = self.hypotheses.get(prepared.hypothesis_id)
        if hypothesis is None:
            hypothesis = EpistemicHypothesis(
                hypothesis_id=prepared.hypothesis_id,
                rule=prepared.rule,
                scope=prepared.scope,
                replaces=prepared.replaces,
            )
            self.hypotheses[hypothesis.hypothesis_id] = hypothesis

        actual_state = _bounded_text(state, "epistemic actual state", max_length=20).upper()
        expected_changed = prepared.predicts.frame_change == "changed"
        matched = (
            bool(frame_changed) == expected_changed
            and int(level_delta) == prepared.predicts.level_delta
            and (not prepared.predicts.state or actual_state == prepared.predicts.state)
        )
        self.predictions += 1
        if matched:
            self.predictions_matched += 1
            hypothesis.support_count += 1
            if hypothesis.replaces and not hypothesis.insight_candidate:
                hypothesis.insight_candidate = True
                self.insight_candidates += 1
            if hypothesis.support_count >= self.verification_threshold:
                newly_verified = hypothesis.status != "supported"
                hypothesis.status = "supported"
                if newly_verified and hypothesis.insight_candidate:
                    self.verified_insights += 1
                self._apply_supersession(hypothesis)
            else:
                hypothesis.status = "candidate" if hypothesis.replaces else "open"
        else:
            self.surprises += 1
            hypothesis.contradiction_count += 1
            if hypothesis.insight_candidate and hypothesis.status != "supported":
                self.false_insights += 1
            hypothesis.status = "refuted"

        evaluation = {
            "hypothesis_id": hypothesis.hypothesis_id,
            "hypothesis_fingerprint": hypothesis.fingerprint,
            "action": prepared.action,
            "matched": matched,
            "expected": prepared.predicts.to_dict(),
            "actual": {
                "frame_change": "changed" if frame_changed else "unchanged",
                "level_delta": int(level_delta),
                "state": actual_state,
            },
            "status": hypothesis.status,
        }
        self._last_evaluation = evaluation
        return dict(evaluation)

    def _apply_supersession(self, hypothesis: EpistemicHypothesis) -> None:
        if not hypothesis.replaces:
            return
        old = self.hypotheses[hypothesis.replaces]
        if old.status == "superseded" and old.superseded_by == hypothesis.hypothesis_id:
            return
        old.status = "superseded"
        old.superseded_by = hypothesis.hypothesis_id
        self.supersessions += 1

    def working_view(self) -> dict[str, Any]:
        """Return bounded live claims plus content-free tombstones for suppressed claims."""

        active = [
            {
                "hypothesis_id": item.hypothesis_id,
                "rule": item.rule,
                "scope": item.scope,
                "status": item.status,
                "support_count": item.support_count,
                "contradiction_count": item.contradiction_count,
            }
            for item in self.hypotheses.values()
            if item.status in _LIVE_STATUSES
        ]
        suppressed = [
            {
                "hypothesis_id": item.hypothesis_id,
                "hypothesis_fingerprint": item.fingerprint,
                "status": item.status,
                "superseded_by": item.superseded_by,
            }
            for item in self.hypotheses.values()
            if item.status not in _LIVE_STATUSES
        ]
        return {
            "policy": "prediction_gated_episode_v1",
            "active_hypotheses": active,
            "suppressed_hypotheses": suppressed,
            "last_evaluation": dict(self._last_evaluation or {}),
        }

    def public_metrics(self) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        for hypothesis in self.hypotheses.values():
            status_counts[hypothesis.status] = status_counts.get(hypothesis.status, 0) + 1
        return {
            "enabled": True,
            "policy": "prediction_gated_episode_v1",
            "verification_threshold": self.verification_threshold,
            "hypotheses": len(self.hypotheses),
            "status_counts": dict(sorted(status_counts.items())),
            "predictions": self.predictions,
            "predictions_matched": self.predictions_matched,
            "prediction_accuracy": (
                round(self.predictions_matched / self.predictions, 4)
                if self.predictions
                else None
            ),
            "surprises": self.surprises,
            "insight_candidates": self.insight_candidates,
            "verified_insights": self.verified_insights,
            "false_insights": self.false_insights,
            "supersessions": self.supersessions,
            "contains_rule_content": False,
            "contains_reasoning": False,
            "persistent": False,
        }


def disabled_epistemic_metrics() -> dict[str, Any]:
    return {
        "enabled": False,
        "policy": "disabled",
        "contains_rule_content": False,
        "contains_reasoning": False,
        "persistent": False,
    }


__all__ = [
    "EpistemicLedger",
    "PredictionSpec",
    "PreparedPrediction",
    "disabled_epistemic_metrics",
]
