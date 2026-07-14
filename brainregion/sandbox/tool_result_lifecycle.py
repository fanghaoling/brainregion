"""Deterministic lifecycle for tool-result messages in sandbox transcripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from brainregion.core.context_loader import estimate_context_tokens

from .input_attribution import attributed_message

ToolResultLifecycleMode = Literal["full", "compact"]

_METADATA_KEY = "_brainregion_tool_result"
_COMPACTABLE_TOOLS = frozenset(
    {
        "search_text",
        "read_text",
        "inspect_file",
        "list_allowed_roots",
        "apply_text_patch",
        "workspace_run_check",
    }
)


def tool_result_message(
    content: str,
    *,
    tool: str,
    step: int,
    target_kind: str = "",
    target_fingerprint: str = "",
    error: bool = False,
) -> dict[str, Any]:
    """Create a provider-safe tool-result message with internal lifecycle metadata."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("tool result step must be a non-negative integer")
    tool = str(tool or "").strip()
    if not tool:
        raise ValueError("tool result tool cannot be empty")
    message = attributed_message("user", str(content or ""), "tool_transcript")
    message[_METADATA_KEY] = {
        "result_id": f"step:{step}:{tool}",
        "tool": tool,
        "step": step,
        "target_kind": str(target_kind or "")[:40],
        "target_fingerprint": str(target_fingerprint or "")[:80],
        "error": bool(error),
        "compacted": False,
    }
    return message


@dataclass
class ToolResultLifecycle:
    mode: ToolResultLifecycleMode = "full"
    live_read_results: int = 3
    compaction_passes: int = 0
    compacted_results: int = 0
    body_characters_removed: int = 0
    body_estimated_tokens_removed: int = 0
    estimated_input_tokens_avoided: int = 0
    compacted_by_tool: dict[str, int] = field(default_factory=dict)
    _seen_result_ids: set[str] = field(default_factory=set, repr=False)
    _active_receipts: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in {"full", "compact"}:
            raise ValueError("tool result lifecycle mode must be 'full' or 'compact'")
        if (
            isinstance(self.live_read_results, bool)
            or not isinstance(self.live_read_results, int)
            or self.live_read_results < 0
        ):
            raise ValueError("live_read_results must be a non-negative integer")

    def apply(self, messages: list[dict[str, Any]], *, next_step: int) -> None:
        """Unload eligible result bodies before the next main-model request."""
        if isinstance(next_step, bool) or not isinstance(next_step, int) or next_step < 0:
            raise ValueError("tool result lifecycle next_step must be a non-negative integer")
        records = self.observe(messages)
        if self.mode == "full":
            return
        self.compaction_passes += 1

        live_reads = {
            record["result_id"]
            for record in sorted(
                (
                    record
                    for record in records
                    if record["tool"] == "read_text" and not record["compacted"]
                ),
                key=lambda record: (record["step"], record["result_id"]),
                reverse=True,
            )[: self.live_read_results]
        }
        latest_patch = max(
            (record["step"] for record in records if record["tool"] == "apply_text_patch"),
            default=-1,
        )
        latest_check = max(
            (record["step"] for record in records if record["tool"] == "workspace_run_check"),
            default=-1,
        )

        for record in records:
            if record["compacted"] or not self._eligible(
                record,
                next_step=next_step,
                live_reads=live_reads,
                latest_patch=latest_patch,
                latest_check=latest_check,
            ):
                continue
            message = record["message"]
            original = str(message.get("content") or "")
            receipt = _receipt(record)
            original_tokens = estimate_context_tokens(original)
            receipt_tokens = estimate_context_tokens(receipt)
            if receipt_tokens >= original_tokens:
                continue
            message["content"] = receipt
            metadata = message[_METADATA_KEY]
            metadata["compacted"] = True
            metadata["original_characters"] = len(original)
            metadata["receipt_characters"] = len(receipt)
            metadata["estimated_tokens_removed"] = original_tokens - receipt_tokens
            self.compacted_results += 1
            self.body_characters_removed += len(original) - len(receipt)
            self.body_estimated_tokens_removed += original_tokens - receipt_tokens
            tool = record["tool"]
            self.compacted_by_tool[tool] = self.compacted_by_tool.get(tool, 0) + 1

        active = self.observe(messages)
        self.estimated_input_tokens_avoided += sum(
            record["estimated_tokens_removed"] for record in active if record["compacted"]
        )

    def observe(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Refresh content-free inventory without changing transcript messages."""
        records = _records(messages)
        self._seen_result_ids.update(record["result_id"] for record in records)
        self._active_receipts = sum(record["compacted"] for record in records)
        return records

    def _eligible(
        self,
        record: dict[str, Any],
        *,
        next_step: int,
        live_reads: set[str],
        latest_patch: int,
        latest_check: int,
    ) -> bool:
        tool = record["tool"]
        age = next_step - record["step"]
        if tool not in _COMPACTABLE_TOOLS or age <= 1:
            return False
        if record["error"] and age <= 2:
            return False
        if tool == "read_text" and record["result_id"] in live_reads:
            return False
        if tool == "apply_text_patch" and record["step"] > latest_check:
            return False
        if tool == "workspace_run_check" and record["step"] > latest_patch:
            return False
        return True

    def public_metrics(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "enabled": self.mode == "compact",
            "policy": "evidence_pinned_receipt_v1",
            "live_read_results": self.live_read_results,
            "tool_results_observed": len(self._seen_result_ids),
            "compaction_passes": self.compaction_passes,
            "compacted_results": self.compacted_results,
            "active_receipts": self._active_receipts,
            "body_characters_removed": self.body_characters_removed,
            "body_estimated_tokens_removed": self.body_estimated_tokens_removed,
            "estimated_input_tokens_avoided": self.estimated_input_tokens_avoided,
            "compacted_by_tool": dict(sorted(self.compacted_by_tool.items())),
            "contains_result_content": False,
            "contains_reasoning": False,
        }


def _records(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for message in messages:
        metadata = message.get(_METADATA_KEY)
        if not isinstance(metadata, dict):
            continue
        records.append(
            {
                "message": message,
                "result_id": str(metadata.get("result_id") or ""),
                "tool": str(metadata.get("tool") or ""),
                "step": int(metadata.get("step") or 0),
                "target_kind": str(metadata.get("target_kind") or ""),
                "error": bool(metadata.get("error")),
                "compacted": bool(metadata.get("compacted")),
                "estimated_tokens_removed": int(
                    metadata.get("estimated_tokens_removed") or 0
                ),
            }
        )
    return records


def _receipt(record: dict[str, Any]) -> str:
    return (
        f'<tool_result_receipt tool="{record["tool"]}" step="{record["step"]}" '
        f'target_kind="{record["target_kind"]}">\n'
        "The earlier result body was unloaded after its guaranteed consumer turn. "
        "Re-run the tool if exact evidence is needed.\n"
        "</tool_result_receipt>"
    )


__all__ = [
    "ToolResultLifecycle",
    "ToolResultLifecycleMode",
    "tool_result_message",
]
