"""Bounded, deduplicated runtime evidence for one epistemic episode."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from brainregion.core.context_loader import estimate_context_tokens

_MISMATCH_FIELDS = frozenset({"change_scale", "level_delta", "state"})
_CHANGE_SCALES = frozenset({"none", "local", "regional", "global"})
_STATES = frozenset({"NOT_FINISHED", "WIN", "GAME_OVER"})
_ACTION_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
)


@dataclass
class EvidenceEvent:
    event_id: str
    action: str
    actual: dict[str, Any]
    first_step: int
    last_step: int
    observations: int = 0
    prediction_matches: int = 0
    prediction_mismatches: int = 0
    mismatch_fields: set[str] = field(default_factory=set)

    def observe(self, evidence: dict[str, Any], *, step: int) -> None:
        self.last_step = int(step)
        self.observations += 1
        matched = evidence.get("matched")
        if matched is True:
            self.prediction_matches += 1
        elif matched is False:
            self.prediction_mismatches += 1
        self.mismatch_fields.update(evidence.get("mismatch_fields") or ())

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "action": self.action,
            "actual": dict(self.actual),
            "observations": self.observations,
            "prediction_matches": self.prediction_matches,
            "prediction_mismatches": self.prediction_mismatches,
            "mismatch_fields": sorted(self.mismatch_fields),
            "first_step": self.first_step,
            "last_step": self.last_step,
        }


class EpistemicEvidenceWorkspace:
    """Upsert objective transitions by stable identity and cap episode context."""

    def __init__(self, *, max_events: int = 8) -> None:
        if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events <= 0:
            raise ValueError("max_events must be a positive integer")
        self.max_events = max_events
        self._events: dict[str, EvidenceEvent] = {}
        self.observations = 0
        self.deduplicated_observations = 0
        self.evicted_events = 0
        self.evicted_observations = 0

    def record(self, raw: Any, *, step: int) -> str:
        evidence = sanitize_objective_evidence(raw)
        action = evidence.get("action")
        actual = evidence.get("actual")
        if not isinstance(action, str) or not isinstance(actual, dict) or not actual:
            return ""
        event_id = _event_id(action, actual)
        event = self._events.get(event_id)
        if event is None:
            if len(self._events) >= self.max_events:
                self._evict_oldest()
            event = EvidenceEvent(
                event_id=event_id,
                action=action,
                actual=dict(actual),
                first_step=int(step),
                last_step=int(step),
            )
            self._events[event_id] = event
        else:
            self.deduplicated_observations += 1
        event.observe(evidence, step=step)
        self.observations += 1
        return event_id

    def model_view(self) -> dict[str, Any]:
        events = sorted(
            self._events.values(),
            key=lambda item: (item.last_step, item.first_step, item.event_id),
        )
        return {
            "policy": "deduplicated_objective_evidence_v1",
            "events": [event.to_model_dict() for event in events],
        }

    def attention_view(
        self,
        *,
        current_action: str = "",
        focus_lineage: tuple[str, ...] = (),
        max_events: int = 4,
    ) -> dict[str, Any]:
        """Select a bounded provider view without removing episode evidence."""

        if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events <= 0:
            raise ValueError("attention max_events must be a positive integer")
        events = list(self._events.values())
        if not events:
            return {
                "policy": "attention_selected_objective_evidence_v1",
                "selection": {
                    "candidate_events": 0,
                    "selected_events": 0,
                    "omitted_events": 0,
                },
                "events": [],
            }

        latest_by_action: dict[str, EvidenceEvent] = {}
        for event in events:
            latest = latest_by_action.get(event.action)
            if latest is None or _recency_key(event) > _recency_key(latest):
                latest_by_action[event.action] = event

        lineage_actions = tuple(
            action
            for action in dict.fromkeys(str(item or "") for item in focus_lineage)
            if action and action != current_action
        )
        ranked: list[tuple[int, tuple[int, int, str], EvidenceEvent, set[str]]] = []
        for event in events:
            reasons: set[str] = set()
            score = 0
            if event.prediction_mismatches > event.prediction_matches:
                reasons.add("unresolved_contradiction")
                score += 400
            if current_action and event.action == current_action:
                reasons.add("current_action")
                score += 200
            if (
                event.action in lineage_actions
                and latest_by_action.get(event.action) is event
            ):
                reasons.add("focus_lineage")
                score += 100
            if reasons:
                ranked.append((score, _recency_key(event), event, reasons))

        if not ranked:
            latest = max(events, key=_recency_key)
            ranked.append((1, _recency_key(latest), latest, {"recent_fallback"}))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = ranked[:max_events]
        selected.sort(key=lambda item: item[1])
        rendered_events = []
        for _, _, event, reasons in selected:
            rendered = event.to_model_dict()
            rendered["attention_reasons"] = sorted(reasons)
            rendered_events.append(rendered)
        return {
            "policy": "attention_selected_objective_evidence_v1",
            "selection": {
                "candidate_events": len(events),
                "selected_events": len(rendered_events),
                "omitted_events": len(events) - len(rendered_events),
            },
            "events": rendered_events,
        }

    def contains(self, event_id: str) -> bool:
        return str(event_id or "") in self._events

    def public_metrics(self) -> dict[str, Any]:
        rendered = json.dumps(
            self.model_view(), ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
        return {
            "policy": "deduplicated_objective_evidence_v1",
            "max_events": self.max_events,
            "events": len(self._events),
            "observations": self.observations,
            "deduplicated_observations": self.deduplicated_observations,
            "evicted_events": self.evicted_events,
            "evicted_observations": self.evicted_observations,
            "estimated_tokens": estimate_context_tokens(rendered),
            "contains_event_content": False,
            "contains_reasoning": False,
            "persistent": False,
        }

    def _evict_oldest(self) -> None:
        oldest = min(
            self._events.values(),
            key=lambda item: (item.last_step, item.first_step, item.event_id),
        )
        self._events.pop(oldest.event_id)
        self.evicted_events += 1
        self.evicted_observations += oldest.observations


def sanitize_objective_evidence(raw: Any) -> dict[str, Any]:
    """Keep runtime facts only; model predictions and free-form text never pass."""

    if not isinstance(raw, dict):
        return {}
    evidence: dict[str, Any] = {}
    action = raw.get("action")
    if (
        isinstance(action, str)
        and 0 < len(action) <= 64
        and all(character in _ACTION_CHARACTERS for character in action)
    ):
        evidence["action"] = action
    if isinstance(raw.get("matched"), bool):
        evidence["matched"] = raw["matched"]

    mismatches = raw.get("mismatch_fields")
    if isinstance(mismatches, list):
        evidence["mismatch_fields"] = sorted(
            {
                item
                for item in mismatches
                if isinstance(item, str) and item in _MISMATCH_FIELDS
            }
        )

    actual = raw.get("actual")
    if isinstance(actual, dict):
        clean_actual: dict[str, Any] = {}
        change_scale = actual.get("change_scale")
        if isinstance(change_scale, str) and change_scale in _CHANGE_SCALES:
            clean_actual["change_scale"] = change_scale
        for key in ("changed_cells", "total_cells"):
            value = actual.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                clean_actual[key] = max(0, min(1_000_000, value))
        level_delta = actual.get("level_delta")
        if isinstance(level_delta, int) and not isinstance(level_delta, bool):
            clean_actual["level_delta"] = max(
                -1_000_000, min(1_000_000, level_delta)
            )
        state = actual.get("state")
        if isinstance(state, str) and state in _STATES:
            clean_actual["state"] = state
        if clean_actual:
            evidence["actual"] = clean_actual
    return evidence


def _event_id(action: str, actual: dict[str, Any]) -> str:
    payload = json.dumps(
        {"action": action, "actual": actual},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "evidence-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _recency_key(event: EvidenceEvent) -> tuple[int, int, str]:
    return event.last_step, event.first_step, event.event_id


__all__ = [
    "EpistemicEvidenceWorkspace",
    "EvidenceEvent",
    "sanitize_objective_evidence",
]
