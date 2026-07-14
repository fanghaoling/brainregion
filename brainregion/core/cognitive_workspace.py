"""Task-scoped cognitive workspace with explicit context delivery boundaries.

The workspace stores retrieved evidence and task state, not model chain of
thought. Entries are process-local, bounded, and expire only when the runtime
advances the owning task. Region-private blocks are never returned in a main
brain view.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

from .activation import ActivationPlan
from .context import ContextBlock
from .context_loader import (
    ActivatedContext,
    ContextLoadRecord,
    context_block_to_dict,
    estimate_context_tokens,
    fit_context_blocks,
)

ContextAudience = Literal["main", "shared", "region"]
WorkspaceConsumer = Literal["main", "region"]

_AUDIENCES = frozenset({"main", "shared", "region"})
_CONSUMERS = frozenset({"main", "region"})
_EVIDENCE_KEYS = ("id", "sha", "path", "file", "issue_id", "source")


def _identifier(value: str, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} cannot be empty")
    if len(text) > 200:
        raise ValueError(f"{name} cannot exceed 200 characters")
    return text


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _clone_block(block: ContextBlock) -> ContextBlock:
    return ContextBlock(
        source=block.source,
        title=block.title,
        content=block.content,
        framing=block.framing,
        metadata=dict(block.metadata),
    )


def _evidence_refs(blocks: tuple[ContextBlock, ...]) -> tuple[str, ...]:
    refs: list[str] = []
    for block in blocks:
        for key in _EVIDENCE_KEYS:
            value = block.metadata.get(key)
            if value not in (None, ""):
                ref = f"{block.source}:{key}:{value}"
                if ref not in refs:
                    refs.append(ref)
    return tuple(refs)


@dataclass
class WorkspaceEntry:
    entry_id: str
    task_id: str
    audience: ContextAudience
    target_region: str
    assignment_id: str
    blocks: tuple[ContextBlock, ...]
    source_skills: tuple[str, ...]
    source_regions: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    remaining_steps: int
    estimated_tokens: int

    def to_public_dict(self) -> dict[str, Any]:
        """Return routing metadata without leaking private block contents."""
        return {
            "entry_id": self.entry_id,
            "task_id": self.task_id,
            "audience": self.audience,
            "target_region": self.target_region,
            "assignment_id": self.assignment_id,
            "source_skills": list(self.source_skills),
            "source_regions": list(self.source_regions),
            "evidence_refs": list(self.evidence_refs),
            "remaining_steps": self.remaining_steps,
            "blocks": len(self.blocks),
            "estimated_tokens": self.estimated_tokens,
        }


@dataclass(frozen=True)
class WorkspaceDelivery:
    status: Literal["staged", "empty"]
    task_id: str
    entry: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "task_id": self.task_id, "entry": self.entry}


@dataclass(frozen=True)
class WorkspaceView:
    task_id: str
    consumer: WorkspaceConsumer
    region: str
    assignment_id: str
    entry_ids: tuple[str, ...]
    blocks: tuple[ContextBlock, ...]
    trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "consumer": self.consumer,
            "region": self.region,
            "assignment_id": self.assignment_id,
            "entry_ids": list(self.entry_ids),
            "context_blocks": [context_block_to_dict(block) for block in self.blocks],
            "trace": dict(self.trace),
        }


class CognitiveWorkspace:
    """In-memory, task-scoped context store with audience-filtered views."""

    def __init__(self, *, max_entries: int = 256) -> None:
        self._max_entries = _positive_int(max_entries, "max_entries")
        self._entries: dict[str, list[WorkspaceEntry]] = {}
        self._lock = RLock()

    def stage(
        self,
        activated: ActivatedContext,
        *,
        task_id: str,
        audience: ContextAudience,
        target_region: str = "",
        assignment_id: str = "",
        ttl_steps: int = 3,
    ) -> WorkspaceDelivery:
        """Store activated blocks and return only a public delivery receipt."""
        task_id = _identifier(task_id, "task_id")
        audience = str(audience or "").strip().casefold()  # type: ignore[assignment]
        if audience not in _AUDIENCES:
            raise ValueError(f"audience must be one of {sorted(_AUDIENCES)}")
        target_region = str(target_region or "").strip().casefold()
        if audience == "region" and not target_region:
            raise ValueError("target_region is required for region audience")
        if audience != "region" and target_region:
            raise ValueError("target_region is only valid for region audience")
        assignment_id = str(assignment_id or "").strip()
        if assignment_id and audience != "region":
            raise ValueError("assignment_id is only valid for region audience")
        if len(assignment_id) > 200:
            raise ValueError("assignment_id cannot exceed 200 characters")
        ttl_steps = _positive_int(ttl_steps, "ttl_steps")

        blocks = tuple(_clone_block(block) for block in activated.blocks)
        if not blocks:
            return WorkspaceDelivery(status="empty", task_id=task_id, entry=None)
        source_skills = tuple(load.skill_id for load in activated.loads if load.status == "loaded")
        source_regions = tuple(dict.fromkeys(load.region for load in activated.loads if load.status == "loaded"))
        estimated_tokens = sum(
            estimate_context_tokens(block.title) + estimate_context_tokens(block.content) for block in blocks
        )
        entry = WorkspaceEntry(
            entry_id=f"ctx-{uuid4().hex[:12]}",
            task_id=task_id,
            audience=audience,  # type: ignore[arg-type]
            target_region=target_region,
            assignment_id=assignment_id,
            blocks=blocks,
            source_skills=source_skills,
            source_regions=source_regions,
            evidence_refs=_evidence_refs(blocks),
            remaining_steps=ttl_steps,
            estimated_tokens=estimated_tokens,
        )
        with self._lock:
            total = sum(len(entries) for entries in self._entries.values())
            if total >= self._max_entries:
                raise RuntimeError("cognitive workspace entry capacity exceeded")
            self._entries.setdefault(task_id, []).append(entry)
        return WorkspaceDelivery(status="staged", task_id=task_id, entry=entry.to_public_dict())

    def publish(
        self,
        blocks: tuple[ContextBlock, ...] | list[ContextBlock],
        *,
        task_id: str,
        source_region: str,
        audience: ContextAudience = "shared",
        target_region: str = "",
        assignment_id: str = "",
        source_skill: str = "",
        ttl_steps: int = 3,
    ) -> WorkspaceDelivery:
        """Publish region-produced context through the normal workspace boundary.

        Functional regions use this entry point after the host has executed their
        bounded tool requests. The synthetic activation record preserves source
        provenance while ``stage`` remains the sole storage and visibility path.
        """
        source_region = _identifier(source_region, "source_region").casefold()
        source_skill = str(source_skill or f"runtime-{source_region}").strip()
        if not source_skill or len(source_skill) > 200:
            raise ValueError("source_skill must contain 1..200 characters")
        normalized_blocks = tuple(blocks)
        if any(not isinstance(block, ContextBlock) for block in normalized_blocks):
            raise ValueError("blocks must contain only ContextBlock values")
        activated = ActivatedContext(
            activation=ActivationPlan(
                decisions=(),
                woken_regions=(source_region,),
                context_requests=(),
                trace={"models_called": False, "source": "region_publish"},
            ),
            blocks=normalized_blocks,
            loads=(
                ContextLoadRecord(
                    skill_id=source_skill,
                    region=source_region,
                    status="loaded" if normalized_blocks else "empty",
                    provider="region_runtime",
                    blocks_loaded=len(normalized_blocks),
                ),
            ),
            trace={"models_called": False, "source": "region_publish"},
        )
        return self.stage(
            activated,
            task_id=task_id,
            audience=audience,
            target_region=target_region,
            assignment_id=assignment_id,
            ttl_steps=ttl_steps,
        )

    def read(
        self,
        task_id: str,
        *,
        consumer: WorkspaceConsumer,
        region: str = "",
        assignment_id: str = "",
        max_context_tokens: int = 2000,
        max_blocks: int = 12,
    ) -> WorkspaceView:
        """Return only blocks visible to the requested main or region consumer."""
        task_id = _identifier(task_id, "task_id")
        consumer = str(consumer or "").strip().casefold()  # type: ignore[assignment]
        if consumer not in _CONSUMERS:
            raise ValueError(f"consumer must be one of {sorted(_CONSUMERS)}")
        region = str(region or "").strip().casefold()
        if consumer == "region" and not region:
            raise ValueError("region is required for region consumer")
        if consumer == "main" and region:
            raise ValueError("region is only valid for region consumer")
        assignment_id = str(assignment_id or "").strip()
        if consumer == "main" and assignment_id:
            raise ValueError("assignment_id is only valid for region consumer")
        if len(assignment_id) > 200:
            raise ValueError("assignment_id cannot exceed 200 characters")
        if isinstance(max_context_tokens, bool) or not isinstance(max_context_tokens, int) or max_context_tokens < 0:
            raise ValueError("max_context_tokens must be a non-negative integer")
        if isinstance(max_blocks, bool) or not isinstance(max_blocks, int) or max_blocks < 0:
            raise ValueError("max_blocks must be a non-negative integer")

        with self._lock:
            entries = list(self._entries.get(task_id, ()))
        visible = [
            entry
            for entry in entries
            if entry.audience == "shared"
            or (consumer == "main" and entry.audience == "main")
            or (
                consumer == "region"
                and entry.audience == "region"
                and entry.target_region == region
                and entry.assignment_id == assignment_id
            )
        ]
        candidates = [block for entry in visible for block in entry.blocks]
        blocks, estimated_tokens, truncated = fit_context_blocks(
            candidates,
            max_tokens=max_context_tokens,
            max_blocks=max_blocks,
        )
        return WorkspaceView(
            task_id=task_id,
            consumer=consumer,  # type: ignore[arg-type]
            region=region,
            assignment_id=assignment_id,
            entry_ids=tuple(entry.entry_id for entry in visible),
            blocks=tuple(blocks),
            trace={
                "strategy": "cognitive_workspace_view_v1",
                "models_called": False,
                "visible_entries": len(visible),
                "candidate_blocks": len(candidates),
                "blocks_loaded": len(blocks),
                "estimated_tokens": estimated_tokens,
                "truncated": truncated,
            },
        )

    def inspect(self, task_id: str) -> dict[str, Any]:
        """Inspect delivery metadata for debugging without returning block contents."""
        task_id = _identifier(task_id, "task_id")
        with self._lock:
            entries = list(self._entries.get(task_id, ()))
        return {
            "task_id": task_id,
            "entries": [entry.to_public_dict() for entry in entries],
            "count": len(entries),
            "contains_context_content": False,
        }

    def advance(self, task_id: str, *, steps: int = 1) -> dict[str, Any]:
        """Advance task time and remove entries whose explicit TTL reaches zero."""
        task_id = _identifier(task_id, "task_id")
        steps = _positive_int(steps, "steps")
        with self._lock:
            entries = self._entries.get(task_id, [])
            for entry in entries:
                entry.remaining_steps -= steps
            active = [entry for entry in entries if entry.remaining_steps > 0]
            expired = len(entries) - len(active)
            if active:
                self._entries[task_id] = active
            else:
                self._entries.pop(task_id, None)
        return {
            "task_id": task_id,
            "advanced_steps": steps,
            "active_entries": len(active),
            "expired_entries": expired,
        }

    def clear(self, task_id: str, *, assignment_id: str = "") -> dict[str, Any]:
        """Unload one assignment or all process-local context for a task."""
        task_id = _identifier(task_id, "task_id")
        assignment_id = str(assignment_id or "").strip()
        if len(assignment_id) > 200:
            raise ValueError("assignment_id cannot exceed 200 characters")
        with self._lock:
            if assignment_id:
                entries = self._entries.get(task_id, [])
                active = [entry for entry in entries if entry.assignment_id != assignment_id]
                removed = len(entries) - len(active)
                if active:
                    self._entries[task_id] = active
                else:
                    self._entries.pop(task_id, None)
            else:
                removed = len(self._entries.pop(task_id, ()))
        return {"task_id": task_id, "removed_entries": removed}


__all__ = [
    "CognitiveWorkspace",
    "ContextAudience",
    "WorkspaceConsumer",
    "WorkspaceDelivery",
    "WorkspaceEntry",
    "WorkspaceView",
]
