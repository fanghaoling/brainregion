"""Materialize bounded provider context from a structured activation plan.

This module is the narrow lifecycle seam between deciding that a region should
wake and placing retrieved context in a caller-owned reasoning turn. Retrieval
is deterministic and model-free; returned blocks are never retained here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from .activation import ActivationPlan, ContextRequest
from .context import ContextBlock, ContextQuery
from .skills.registry import SkillRegistry
from .skills.resolver import Resolver, resolve_skill_body

ContextLoadStatus = Literal["loaded", "empty", "skipped", "failed"]


def estimate_context_tokens(text: str) -> int:
    """Estimate tokens conservatively without binding the runtime to one tokenizer."""
    if not text:
        return 0
    ascii_count = sum(ord(char) < 128 for char in text)
    non_ascii_count = len(text) - ascii_count
    return non_ascii_count + math.ceil(ascii_count / 4)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    if estimate_context_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_context_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip()


def fit_context_blocks(
    blocks: list[ContextBlock],
    *,
    max_tokens: int,
    max_blocks: int,
) -> tuple[list[ContextBlock], int, bool]:
    fitted: list[ContextBlock] = []
    used = 0
    truncated = False
    for block in blocks:
        if len(fitted) >= max_blocks or used >= max_tokens:
            truncated = True
            break
        title_tokens = estimate_context_tokens(block.title)
        remaining = max_tokens - used
        if title_tokens >= remaining:
            truncated = True
            break
        content_budget = remaining - title_tokens
        content = _truncate_to_tokens(block.content, content_budget)
        was_truncated = content != block.content
        if not content and block.content:
            truncated = True
            break
        metadata = dict(block.metadata)
        if was_truncated:
            metadata["activation_truncated"] = True
        fitted_block = ContextBlock(
            source=block.source,
            title=block.title,
            content=content,
            framing=block.framing,
            metadata=metadata,
        )
        fitted.append(fitted_block)
        used += title_tokens + estimate_context_tokens(content)
        truncated = truncated or was_truncated
        if was_truncated:
            break
    if len(fitted) < len(blocks):
        truncated = True
    return fitted, used, truncated


def context_block_to_dict(block: ContextBlock) -> dict[str, Any]:
    return {
        "source": block.source,
        "title": block.title,
        "content": block.content,
        "framing": block.framing,
        "metadata": dict(block.metadata),
    }


@dataclass(frozen=True)
class ContextLoadRecord:
    skill_id: str
    region: str
    status: ContextLoadStatus
    provider: str = ""
    selectors: tuple[str, ...] = ()
    requested_tokens: int = 0
    estimated_tokens: int = 0
    blocks_loaded: int = 0
    truncated: bool = False
    reason: str = ""
    provider_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "region": self.region,
            "status": self.status,
            "provider": self.provider,
            "selectors": list(self.selectors),
            "requested_tokens": self.requested_tokens,
            "estimated_tokens": self.estimated_tokens,
            "blocks_loaded": self.blocks_loaded,
            "truncated": self.truncated,
            "reason": self.reason,
            "provider_meta": dict(self.provider_meta),
        }


@dataclass(frozen=True)
class ActivatedContext:
    activation: ActivationPlan
    blocks: tuple[ContextBlock, ...]
    loads: tuple[ContextLoadRecord, ...]
    trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "activation": self.activation.to_dict(),
            "context_blocks": [context_block_to_dict(block) for block in self.blocks],
            "loads": [load.to_dict() for load in self.loads],
            "trace": dict(self.trace),
        }


def _skipped(request: ContextRequest, reason: str) -> ContextLoadRecord:
    return ContextLoadRecord(
        skill_id=request.skill_id,
        region=request.region,
        status="skipped",
        selectors=request.selectors,
        requested_tokens=request.max_tokens,
        reason=reason,
    )


def load_activation_context(
    activation: ActivationPlan,
    *,
    query_text: str,
    skill_registry: SkillRegistry,
    resolvers: dict[str, Resolver],
    scope_regions: frozenset[str] | None = None,
    top_k: int = 5,
    max_blocks: int = 12,
) -> ActivatedContext:
    """Resolve provider Skills selected by an activation plan within its budgets."""
    query_text = (query_text or "").strip()
    if not query_text:
        raise ValueError("query_text cannot be empty")
    for value, name in ((top_k, "top_k"), (max_blocks, "max_blocks")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    scopes = scope_regions
    if scopes is None:
        scopes = frozenset(activation.woken_regions)
    remaining_blocks = max_blocks
    loaded_blocks: list[ContextBlock] = []
    loads: list[ContextLoadRecord] = []
    providers_called: list[str] = []

    for request in activation.context_requests:
        if request.activation_mode != "context":
            loads.append(_skipped(request, "activation_mode_not_context"))
            continue
        manifest = skill_registry.get(request.skill_id)
        if manifest is None:
            loads.append(_skipped(request, "skill_not_registered"))
            continue
        if remaining_blocks <= 0:
            loads.append(_skipped(request, "block_budget_exceeded"))
            continue

        query = ContextQuery(
            text=query_text,
            regions=scopes,
            top_k=top_k,
            selectors=request.selectors,
        )
        try:
            result = resolve_skill_body(manifest, query, resolvers=resolvers)
        except Exception as exc:  # one provider must not block other activated context
            loads.append(
                ContextLoadRecord(
                    skill_id=request.skill_id,
                    region=request.region,
                    status="failed",
                    selectors=request.selectors,
                    requested_tokens=request.max_tokens,
                    reason=f"{type(exc).__name__}: {exc}"[:240],
                )
            )
            continue

        providers_called.append(result.provider)
        fitted, used, truncated = fit_context_blocks(
            result.blocks,
            max_tokens=request.max_tokens,
            max_blocks=remaining_blocks,
        )
        loaded_blocks.extend(fitted)
        remaining_blocks -= len(fitted)
        loads.append(
            ContextLoadRecord(
                skill_id=request.skill_id,
                region=request.region,
                status="loaded" if fitted else "empty",
                provider=result.provider,
                selectors=request.selectors,
                requested_tokens=request.max_tokens,
                estimated_tokens=used,
                blocks_loaded=len(fitted),
                truncated=truncated,
                reason="" if fitted else "provider_returned_no_matching_context",
                provider_meta=dict(result.meta),
            )
        )

    estimated_tokens = sum(load.estimated_tokens for load in loads)
    return ActivatedContext(
        activation=activation,
        blocks=tuple(loaded_blocks),
        loads=tuple(loads),
        trace={
            "strategy": "activation_context_load_v1",
            "models_called": False,
            "providers_called": providers_called,
            "blocks_loaded": len(loaded_blocks),
            "estimated_tokens": estimated_tokens,
            "max_blocks": max_blocks,
            "remaining_blocks": remaining_blocks,
            "scope_regions": sorted(scopes),
            "retained_by_runtime": False,
        },
    )


__all__ = [
    "ActivatedContext",
    "ContextLoadRecord",
    "ContextLoadStatus",
    "context_block_to_dict",
    "fit_context_blocks",
    "estimate_context_tokens",
    "load_activation_context",
]
