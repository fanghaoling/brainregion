"""Opt-in transcript suppression for objectively rejected epistemic turns."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from brainregion.core.context_loader import estimate_context_tokens

from .epistemic_ledger import EpistemicLedger

EpistemicTranscriptMode = Literal["full", "suppress", "evidence"]

_METADATA_KEY = "_brainregion_epistemic_turn"
_SUPPRESSED_STATUSES = frozenset({"refuted", "superseded", "rejected"})
_MISMATCH_FIELDS = frozenset({"change_scale", "level_delta", "state"})
_CHANGE_SCALES = frozenset({"none", "local", "regional", "global"})
_STATES = frozenset({"NOT_FINISHED", "WIN", "GAME_OVER"})
_ACTION_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
)


@dataclass
class EpistemicTranscriptLifecycle:
    mode: EpistemicTranscriptMode = "full"
    ledger: EpistemicLedger | None = field(default=None, repr=False)
    compaction_passes: int = 0
    marked_turns: int = 0
    suppressed_turns: int = 0
    evidence_receipts: int = 0
    body_characters_removed: int = 0
    body_estimated_tokens_removed: int = 0
    estimated_input_tokens_avoided: int = 0
    suppressed_by_status: dict[str, int] = field(default_factory=dict)
    _seen_turn_ids: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in {"full", "suppress", "evidence"}:
            raise ValueError(
                "epistemic transcript lifecycle must be 'full', 'suppress', or 'evidence'"
            )
        if self.mode != "full" and self.ledger is None:
            raise ValueError("epistemic transcript lifecycle requires an EpistemicLedger")

    def mark(
        self,
        message: dict[str, Any],
        *,
        hypothesis_id: str,
        step: int,
        rejected: bool = False,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        """Attach provider-invisible identity to one model-authored epistemic turn."""

        clean_id = str(hypothesis_id or "").strip()
        if not clean_id:
            return
        turn_id = f"step:{int(step)}:{_fingerprint(clean_id)}"
        message[_METADATA_KEY] = {
            "turn_id": turn_id,
            "hypothesis_id": clean_id,
            "step": int(step),
            "rejected": bool(rejected),
            "evidence": _objective_evidence(evidence),
            "compacted": False,
        }
        self._seen_turn_ids.add(turn_id)
        self.marked_turns = len(self._seen_turn_ids)

    def apply(self, messages: list[dict[str, Any]], *, next_step: int) -> None:
        """Unload rejected rule text before the next provider request."""

        del next_step
        records = self.observe(messages)
        if self.mode == "full":
            return
        self.compaction_passes += 1
        for record in records:
            if record["compacted"]:
                continue
            status = self._status(record)
            if status not in _SUPPRESSED_STATUSES:
                continue
            message = record["message"]
            original = str(message.get("content") or "")
            receipt = _receipt(record, status, mode=self.mode)
            original_tokens = estimate_context_tokens(original)
            receipt_tokens = estimate_context_tokens(receipt)
            message["content"] = receipt
            metadata = message[_METADATA_KEY]
            metadata["compacted"] = True
            metadata["status"] = status
            metadata["estimated_tokens_removed"] = max(0, original_tokens - receipt_tokens)
            self.suppressed_turns += 1
            if self.mode == "evidence" and record["evidence"]:
                self.evidence_receipts += 1
            self.body_characters_removed += max(0, len(original) - len(receipt))
            self.body_estimated_tokens_removed += max(0, original_tokens - receipt_tokens)
            self.suppressed_by_status[status] = self.suppressed_by_status.get(status, 0) + 1

        active = self.observe(messages)
        self.estimated_input_tokens_avoided += sum(
            record["estimated_tokens_removed"] for record in active if record["compacted"]
        )

    def observe(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records = []
        for message in messages:
            metadata = message.get(_METADATA_KEY)
            if not isinstance(metadata, dict):
                continue
            turn_id = str(metadata.get("turn_id") or "")
            self._seen_turn_ids.add(turn_id)
            records.append(
                {
                    "message": message,
                    "turn_id": turn_id,
                    "hypothesis_id": str(metadata.get("hypothesis_id") or ""),
                    "step": int(metadata.get("step") or 0),
                    "rejected": bool(metadata.get("rejected")),
                    "evidence": _objective_evidence(metadata.get("evidence")),
                    "compacted": bool(metadata.get("compacted")),
                    "estimated_tokens_removed": int(
                        metadata.get("estimated_tokens_removed") or 0
                    ),
                }
            )
        self.marked_turns = len(self._seen_turn_ids)
        return records

    def _status(self, record: dict[str, Any]) -> str:
        if record["rejected"]:
            return "rejected"
        if self.ledger is None:
            return ""
        hypothesis = self.ledger.hypotheses.get(record["hypothesis_id"])
        return hypothesis.status if hypothesis is not None else ""

    def public_metrics(self) -> dict[str, Any]:
        policy = {
            "full": "none",
            "suppress": "objective_belief_suppression_v1",
            "evidence": "objective_evidence_receipt_v1",
        }[self.mode]
        receipt_mode = {
            "full": "none",
            "suppress": "status_only",
            "evidence": "objective_evidence",
        }[self.mode]
        return {
            "mode": self.mode,
            "enabled": self.mode != "full",
            "policy": policy,
            "receipt_mode": receipt_mode,
            "marked_turns": self.marked_turns,
            "compaction_passes": self.compaction_passes,
            "suppressed_turns": self.suppressed_turns,
            "evidence_receipts": self.evidence_receipts,
            "suppressed_by_status": dict(sorted(self.suppressed_by_status.items())),
            "body_characters_removed": self.body_characters_removed,
            "body_estimated_tokens_removed": self.body_estimated_tokens_removed,
            "estimated_input_tokens_avoided": self.estimated_input_tokens_avoided,
            "contains_rule_content": False,
            "contains_reasoning": False,
            "contains_objective_evidence": self.evidence_receipts > 0,
            "persistent": False,
        }


def _fingerprint(hypothesis_id: str) -> str:
    return hashlib.sha256(hypothesis_id.encode("utf-8")).hexdigest()[:12]


def _receipt(
    record: dict[str, Any],
    status: str,
    *,
    mode: EpistemicTranscriptMode,
) -> str:
    if mode == "evidence":
        evidence = json.dumps(
            record["evidence"] or None,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return (
            f'<epistemic_evidence_receipt step="{record["step"]}" '
            f'hypothesis="{_fingerprint(record["hypothesis_id"])}" status="{status}">\n'
            f"objective_evidence={evidence}\n"
            "The model-authored rule, prediction, and reasoning were unloaded. "
            "Treat this bounded runtime evidence as observation data, not as a rule.\n"
            "</epistemic_evidence_receipt>"
        )
    return (
        f'<epistemic_turn_receipt step="{record["step"]}" '
        f'hypothesis="{_fingerprint(record["hypothesis_id"])}" status="{status}">\n'
        "The prior model-authored rule and reasoning were unloaded after objective runtime rejection. "
        "Use the current epistemic ledger instead of reconstructing them.\n"
        "</epistemic_turn_receipt>"
    )


def _objective_evidence(raw: Any) -> dict[str, Any]:
    """Keep a small allow-listed projection of runtime feedback, never model predictions."""

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
        clean_mismatches = sorted(
            {
                item
                for item in mismatches
                if isinstance(item, str) and item in _MISMATCH_FIELDS
            }
        )
        evidence["mismatch_fields"] = clean_mismatches

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


__all__ = ["EpistemicTranscriptLifecycle", "EpistemicTranscriptMode"]
