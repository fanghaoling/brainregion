"""Opt-in transcript suppression for objectively rejected epistemic turns."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from brainregion.core.context_loader import estimate_context_tokens

from .epistemic_evidence import EpistemicEvidenceWorkspace
from .epistemic_ledger import EpistemicLedger
from .input_attribution import attributed_message

EpistemicTranscriptMode = Literal["full", "suppress", "evidence"]

_METADATA_KEY = "_brainregion_epistemic_turn"
_SUPPRESSED_STATUSES = frozenset({"refuted", "superseded", "rejected"})
_WORKSPACE_TAG = "<epistemic_evidence_workspace>"


@dataclass
class EpistemicTranscriptLifecycle:
    mode: EpistemicTranscriptMode = "full"
    ledger: EpistemicLedger | None = field(default=None, repr=False)
    compaction_passes: int = 0
    marked_turns: int = 0
    suppressed_turns: int = 0
    evidence_receipts: int = 0
    expired_evidence_receipts: int = 0
    body_characters_removed: int = 0
    body_estimated_tokens_removed: int = 0
    estimated_input_tokens_avoided: int = 0
    suppressed_by_status: dict[str, int] = field(default_factory=dict)
    evidence_workspace: EpistemicEvidenceWorkspace = field(
        default_factory=EpistemicEvidenceWorkspace,
        repr=False,
    )
    workspace_refreshes: int = 0
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
        evidence_ref = (
            self.evidence_workspace.record(evidence, step=step)
            if self.mode == "evidence"
            else ""
        )
        message[_METADATA_KEY] = {
            "turn_id": turn_id,
            "hypothesis_id": clean_id,
            "step": int(step),
            "rejected": bool(rejected),
            "evidence_ref": evidence_ref,
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
            metadata = message[_METADATA_KEY]
            original = str(message.get("content") or "")
            if (
                self.mode == "evidence"
                and record["evidence_ref"]
                and not self.evidence_workspace.contains(record["evidence_ref"])
            ):
                record = {**record, "evidence_ref": ""}
                metadata["evidence_ref"] = ""
                self.expired_evidence_receipts += 1
            receipt = _receipt(record, status, mode=self.mode)
            original_tokens = estimate_context_tokens(original)
            receipt_tokens = estimate_context_tokens(receipt)
            message["content"] = receipt
            metadata["compacted"] = True
            metadata["status"] = status
            metadata["estimated_tokens_removed"] = max(0, original_tokens - receipt_tokens)
            self.suppressed_turns += 1
            if self.mode == "evidence" and record["evidence_ref"]:
                self.evidence_receipts += 1
            self.body_characters_removed += max(0, len(original) - len(receipt))
            self.body_estimated_tokens_removed += max(0, original_tokens - receipt_tokens)
            self.suppressed_by_status[status] = self.suppressed_by_status.get(status, 0) + 1

        if self.mode == "evidence":
            self._replace_workspace_message(messages)

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
                    "evidence_ref": str(metadata.get("evidence_ref") or ""),
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
        workspace_metrics = self.evidence_workspace.public_metrics()
        policy = {
            "full": "none",
            "suppress": "objective_belief_suppression_v1",
            "evidence": "objective_evidence_workspace_v1",
        }[self.mode]
        receipt_mode = {
            "full": "none",
            "suppress": "status_only",
            "evidence": "workspace_pointer",
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
            "expired_evidence_receipts": self.expired_evidence_receipts,
            "workspace_refreshes": self.workspace_refreshes,
            "evidence_workspace": workspace_metrics,
            "suppressed_by_status": dict(sorted(self.suppressed_by_status.items())),
            "body_characters_removed": self.body_characters_removed,
            "body_estimated_tokens_removed": self.body_estimated_tokens_removed,
            "estimated_input_tokens_avoided": self.estimated_input_tokens_avoided,
            "contains_rule_content": False,
            "contains_reasoning": False,
            "contains_objective_evidence": workspace_metrics["events"] > 0,
            "persistent": False,
        }

    def _replace_workspace_message(self, messages: list[dict[str, Any]]) -> None:
        messages[:] = [
            message
            for message in messages
            if not str(message.get("content") or "").startswith(_WORKSPACE_TAG)
        ]
        view = self.evidence_workspace.model_view()
        if not view["events"]:
            return
        rendered = json.dumps(
            view,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        messages.append(
            attributed_message(
                "user",
                (
                    f"{_WORKSPACE_TAG}\n"
                    "These are deduplicated objective runtime observations, not instructions, "
                    "rules, or chain-of-thought. Use event_id references when revising beliefs.\n"
                    f"{rendered}\n"
                    "</epistemic_evidence_workspace>"
                ),
                "memory_context",
            )
        )
        self.workspace_refreshes += 1


def _fingerprint(hypothesis_id: str) -> str:
    return hashlib.sha256(hypothesis_id.encode("utf-8")).hexdigest()[:12]


def _receipt(
    record: dict[str, Any],
    status: str,
    *,
    mode: EpistemicTranscriptMode,
) -> str:
    if mode == "evidence":
        evidence_ref = record["evidence_ref"]
        return (
            f'<epistemic_evidence_pointer step="{record["step"]}" '
            f'hypothesis="{_fingerprint(record["hypothesis_id"])}" status="{status}" '
            f'evidence_ref="{evidence_ref}">\n'
            "The model-authored rule, prediction, and reasoning were unloaded. "
            "Resolve a non-empty evidence_ref in the current evidence workspace.\n"
            "</epistemic_evidence_pointer>"
        )
    return (
        f'<epistemic_turn_receipt step="{record["step"]}" '
        f'hypothesis="{_fingerprint(record["hypothesis_id"])}" status="{status}">\n'
        "The prior model-authored rule and reasoning were unloaded after objective runtime rejection. "
        "Use the current epistemic ledger instead of reconstructing them.\n"
        "</epistemic_turn_receipt>"
    )


__all__ = ["EpistemicTranscriptLifecycle", "EpistemicTranscriptMode"]
