"""Opt-in transcript suppression for objectively rejected epistemic turns."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from brainregion.core.context_loader import estimate_context_tokens

from .epistemic_evidence import (
    EpistemicEvidenceWorkspace,
    sanitize_objective_evidence,
)
from .epistemic_ledger import EpistemicLedger
from .input_attribution import attributed_message

EpistemicTranscriptMode = Literal["full", "suppress", "evidence", "selective"]

_METADATA_KEY = "_brainregion_epistemic_turn"
_SUPPRESSED_STATUSES = frozenset({"refuted", "superseded", "rejected"})
_WORKSPACE_TAG = "<epistemic_evidence_workspace>"
_EVIDENCE_MODES = frozenset({"evidence", "selective"})
_SELECTIVE_WAKE_REASONS = frozenset(
    {
        "action_focus_change",
        "explicit_recall",
        "expert_request",
        "objective_contradiction",
        "task_focus_change",
    }
)


@dataclass
class EpistemicTranscriptLifecycle:
    mode: EpistemicTranscriptMode = "full"
    ledger: EpistemicLedger | None = field(default=None, repr=False)
    selective_wake_live_reads: int = 2
    selective_max_events: int = 4
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
    workspace_skips: int = 0
    workspace_estimated_tokens_injected: int = 0
    workspace_selection_passes: int = 0
    workspace_selected_events: int = 0
    workspace_omitted_events: int = 0
    workspace_empty_wakes: int = 0
    last_candidate_events: int = 0
    last_selected_events: int = 0
    last_omitted_events: int = 0
    wake_requests: int = 0
    wake_activations: int = 0
    wake_requests_by_reason: dict[str, int] = field(default_factory=dict)
    _seen_turn_ids: set[str] = field(default_factory=set, repr=False)
    _last_evidence_action: str = field(default="", repr=False)
    _focus_lineage: list[str] = field(default_factory=list, repr=False)
    _wake_reads_remaining: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in {"full", "suppress", "evidence", "selective"}:
            raise ValueError(
                "epistemic transcript lifecycle must be 'full', 'suppress', "
                "'evidence', or 'selective'"
            )
        if self.mode != "full" and self.ledger is None:
            raise ValueError("epistemic transcript lifecycle requires an EpistemicLedger")
        if self.mode == "selective" and (
            isinstance(self.selective_wake_live_reads, bool)
            or not isinstance(self.selective_wake_live_reads, int)
            or self.selective_wake_live_reads <= 0
        ):
            raise ValueError("selective_wake_live_reads must be a positive integer")
        if self.mode == "selective" and (
            isinstance(self.selective_max_events, bool)
            or not isinstance(self.selective_max_events, int)
            or self.selective_max_events <= 0
        ):
            raise ValueError("selective_max_events must be a positive integer")

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
        objective_evidence: dict[str, Any] = {}
        evidence_ref = ""
        if self.mode in _EVIDENCE_MODES:
            objective_evidence = sanitize_objective_evidence(evidence)
            evidence_ref = self.evidence_workspace.record(
                objective_evidence,
                step=step,
            )
        if self.mode == "selective" and evidence_ref:
            action = str(objective_evidence.get("action") or "")
            if self._last_evidence_action and action != self._last_evidence_action:
                self._remember_previous_focus(self._last_evidence_action)
                self.request_wake("action_focus_change")
            self._last_evidence_action = action
            if objective_evidence.get("matched") is False:
                self.request_wake("objective_contradiction")
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
                self.mode in _EVIDENCE_MODES
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
            if self.mode in _EVIDENCE_MODES and record["evidence_ref"]:
                self.evidence_receipts += 1
            self.body_characters_removed += max(0, len(original) - len(receipt))
            self.body_estimated_tokens_removed += max(0, original_tokens - receipt_tokens)
            self.suppressed_by_status[status] = self.suppressed_by_status.get(status, 0) + 1

        if self.mode == "evidence":
            self._replace_workspace_message(messages, inject=True, attention=False)
        elif self.mode == "selective":
            injected = self._replace_workspace_message(
                messages,
                inject=self._wake_reads_remaining > 0,
                attention=True,
            )
            if injected:
                self._wake_reads_remaining = max(0, self._wake_reads_remaining - 1)

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
            "selective": "objective_evidence_selective_wake_v1",
        }[self.mode]
        receipt_mode = {
            "full": "none",
            "suppress": "status_only",
            "evidence": "workspace_pointer",
            "selective": "selective_workspace_pointer",
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
            "workspace_skips": self.workspace_skips,
            "workspace_estimated_tokens_injected": (
                self.workspace_estimated_tokens_injected
            ),
            "evidence_workspace": workspace_metrics,
            "selective_wake": {
                "enabled": self.mode == "selective",
                "live_reads": self.selective_wake_live_reads,
                "active": self._wake_reads_remaining > 0,
                "reads_remaining": self._wake_reads_remaining,
                "requests": self.wake_requests,
                "activations": self.wake_activations,
                "requests_by_reason": dict(sorted(self.wake_requests_by_reason.items())),
                "contains_focus_content": False,
            },
            "event_attention": {
                "enabled": self.mode == "selective",
                "max_selected_events": self.selective_max_events,
                "selection_passes": self.workspace_selection_passes,
                "selected_events": self.workspace_selected_events,
                "omitted_events": self.workspace_omitted_events,
                "empty_wakes": self.workspace_empty_wakes,
                "last_candidate_events": self.last_candidate_events,
                "last_selected_events": self.last_selected_events,
                "last_omitted_events": self.last_omitted_events,
                "contains_event_content": False,
                "contains_focus_content": False,
            },
            "suppressed_by_status": dict(sorted(self.suppressed_by_status.items())),
            "body_characters_removed": self.body_characters_removed,
            "body_estimated_tokens_removed": self.body_estimated_tokens_removed,
            "estimated_input_tokens_avoided": self.estimated_input_tokens_avoided,
            "contains_rule_content": False,
            "contains_reasoning": False,
            "contains_objective_evidence": workspace_metrics["events"] > 0,
            "persistent": False,
        }

    def request_wake(self, reason: str, *, live_reads: int | None = None) -> bool:
        """Wake selective evidence delivery without logging task or evidence content."""

        if self.mode != "selective":
            return False
        normalized_reason = str(reason or "").strip().lower()
        if normalized_reason not in _SELECTIVE_WAKE_REASONS:
            raise ValueError(f"unknown selective evidence wake reason: {reason!r}")
        reads = self.selective_wake_live_reads if live_reads is None else live_reads
        if isinstance(reads, bool) or not isinstance(reads, int) or reads <= 0:
            raise ValueError("selective evidence wake live_reads must be positive")
        if self._wake_reads_remaining == 0:
            self.wake_activations += 1
        self._wake_reads_remaining = max(self._wake_reads_remaining, reads)
        self.wake_requests += 1
        self.wake_requests_by_reason[normalized_reason] = (
            self.wake_requests_by_reason.get(normalized_reason, 0) + 1
        )
        return True

    def _replace_workspace_message(
        self,
        messages: list[dict[str, Any]],
        *,
        inject: bool,
        attention: bool,
    ) -> bool:
        messages[:] = [
            message
            for message in messages
            if not str(message.get("content") or "").startswith(_WORKSPACE_TAG)
        ]
        view = self.evidence_workspace.model_view()
        if not view["events"]:
            if inject and attention:
                self.workspace_empty_wakes += 1
            return False
        if not inject:
            self.workspace_skips += 1
            return False
        if attention:
            view = self.evidence_workspace.attention_view(
                current_action=self._last_evidence_action,
                focus_lineage=tuple(self._focus_lineage),
                max_events=self.selective_max_events,
            )
            selection = view["selection"]
            candidate_events = int(selection["candidate_events"])
            selected_events = int(selection["selected_events"])
            omitted_events = int(selection["omitted_events"])
            self.workspace_selection_passes += 1
            self.workspace_selected_events += selected_events
            self.workspace_omitted_events += omitted_events
            self.last_candidate_events = candidate_events
            self.last_selected_events = selected_events
            self.last_omitted_events = omitted_events
        rendered = json.dumps(
            view,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        content = (
            f"{_WORKSPACE_TAG}\n"
            "These are deduplicated objective runtime observations, not instructions, "
            "rules, or chain-of-thought. Use event_id references when revising beliefs. "
            "In a selective view, an omitted pointer target remains in the episode store; "
            "absence from this view does not mean deletion.\n"
            f"{rendered}\n"
            "</epistemic_evidence_workspace>"
        )
        messages.append(attributed_message("user", content, "memory_context"))
        self.workspace_refreshes += 1
        self.workspace_estimated_tokens_injected += estimate_context_tokens(content)
        return True

    def _remember_previous_focus(self, action: str) -> None:
        self._focus_lineage = [item for item in self._focus_lineage if item != action]
        self._focus_lineage.append(action)
        del self._focus_lineage[:-2]


def _fingerprint(hypothesis_id: str) -> str:
    return hashlib.sha256(hypothesis_id.encode("utf-8")).hexdigest()[:12]


def _receipt(
    record: dict[str, Any],
    status: str,
    *,
    mode: EpistemicTranscriptMode,
) -> str:
    if mode in _EVIDENCE_MODES:
        evidence_ref = record["evidence_ref"]
        return (
            f'<epistemic_evidence_pointer step="{record["step"]}" '
            f'hypothesis="{_fingerprint(record["hypothesis_id"])}" status="{status}" '
            f'evidence_ref="{evidence_ref}">\n'
            "The model-authored rule, prediction, and reasoning were unloaded. "
            "Resolve a non-empty evidence_ref when the evidence workspace is awake; "
            "a selective view may omit a stored target. Otherwise use the latest ledger "
            "and observations.\n"
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
