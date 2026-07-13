"""Metadata-only authorization for exporting expert workspace context.

The policy never rewrites ContextBlocks. ``off`` is a true bypass, ``audit``
observes without changing the call, and ``enforce`` either allows the original
blocks or denies the model call before prompt construction.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .context import ContextBlock

_MODES = frozenset({"off", "audit", "enforce"})
_TRUST_LEVELS = frozenset({"external", "trusted", "local"})
_SENSITIVITIES = ("public", "project", "private", "secret")
_SENSITIVITY_RANK = {name: rank for rank, name in enumerate(_SENSITIVITIES)}
_DEFAULT_SOURCE_SENSITIVITY = {
    "git": "project",
    "memory": "private",
    "public_test_fixture": "public",
}
_ALLOWED_BY_TRUST = {
    "external": frozenset({"public"}),
    "trusted": frozenset({"public", "project", "private"}),
    "local": frozenset(_SENSITIVITIES),
}


def context_export_mode(policy: dict[str, Any] | None) -> str:
    if policy is None:
        return "off"
    if not isinstance(policy, dict):
        raise ValueError("context_export_policy must be an object")
    mode = str(policy.get("mode") or "off").strip().casefold()
    if mode not in _MODES:
        raise ValueError("context_export_policy.mode must be off, audit, or enforce")
    return mode


def endpoint_context_trust(
    endpoint_id: str | None,
    endpoints: dict[str, Any] | None,
    policy: dict[str, Any] | None,
) -> str:
    policy = policy or {}
    overrides = policy.get("endpoint_trust") or {}
    if not isinstance(overrides, dict):
        raise ValueError("context_export_policy.endpoint_trust must be an object")
    endpoint = (endpoints or {}).get(endpoint_id, {}) if endpoint_id else {}
    trust = overrides.get(endpoint_id) if endpoint_id else overrides.get("official")
    if trust is None and isinstance(endpoint, dict):
        trust = endpoint.get("context_trust")
    trust = str(trust or "external").strip().casefold()
    if trust not in _TRUST_LEVELS:
        raise ValueError("context trust must be external, trusted, or local")
    return trust


@dataclass(frozen=True)
class ContextExportDecision:
    mode: str
    evaluated: bool
    action: str
    allowed: bool
    endpoint_trust: str
    highest_sensitivity: str | None = None
    block_counts: tuple[tuple[str, int], ...] = ()
    denied_sensitivities: tuple[str, ...] = ()
    reason: str = ""

    @property
    def permits_call(self) -> bool:
        return self.mode != "enforce" or self.allowed

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "evaluated": self.evaluated,
            "action": self.action,
            "allowed": self.allowed,
            "endpoint_trust": self.endpoint_trust,
            "highest_sensitivity": self.highest_sensitivity,
            "block_counts": dict(self.block_counts),
            "denied_sensitivities": list(self.denied_sensitivities),
            "reason": self.reason,
            "context_modified": False,
        }


def bypass_context_export() -> ContextExportDecision:
    """Return the off-mode result without inspecting any ContextBlock."""
    return ContextExportDecision(
        mode="off",
        evaluated=False,
        action="bypass",
        allowed=True,
        endpoint_trust="uninspected",
        reason="policy_off",
    )


def _source_sensitivity(policy: dict[str, Any]) -> dict[str, str]:
    configured = policy.get("source_sensitivity") or {}
    if not isinstance(configured, dict):
        raise ValueError("context_export_policy.source_sensitivity must be an object")
    merged = dict(_DEFAULT_SOURCE_SENSITIVITY)
    for source, sensitivity in configured.items():
        normalized = str(sensitivity or "").strip().casefold()
        if normalized not in _SENSITIVITY_RANK:
            raise ValueError("context sensitivity must be public, project, private, or secret")
        merged[str(source)] = normalized
    return merged


def _classify_block(
    block: ContextBlock,
    *,
    source_sensitivity: dict[str, str],
    default_sensitivity: str,
) -> str:
    explicit = block.metadata.get("sensitivity")
    sensitivity = explicit if explicit is not None else source_sensitivity.get(block.source)
    normalized = str(sensitivity or default_sensitivity).strip().casefold()
    if normalized not in _SENSITIVITY_RANK:
        raise ValueError("ContextBlock sensitivity must be public, project, private, or secret")
    return normalized


def evaluate_context_export(
    blocks: tuple[ContextBlock, ...],
    *,
    policy: dict[str, Any] | None,
    endpoint_trust: str,
) -> ContextExportDecision:
    mode = context_export_mode(policy)
    if mode == "off":
        return bypass_context_export()
    endpoint_trust = str(endpoint_trust or "").strip().casefold()
    if endpoint_trust not in _TRUST_LEVELS:
        raise ValueError("context trust must be external, trusted, or local")
    policy = policy or {}
    default_sensitivity = str(policy.get("default_sensitivity") or "private").strip().casefold()
    if default_sensitivity not in _SENSITIVITY_RANK:
        raise ValueError("context_export_policy.default_sensitivity must be public, project, private, or secret")
    sources = _source_sensitivity(policy)
    counts = Counter(
        _classify_block(
            block,
            source_sensitivity=sources,
            default_sensitivity=default_sensitivity,
        )
        for block in blocks
    )
    highest = max(counts, key=_SENSITIVITY_RANK.get) if counts else None
    allowed_levels = _ALLOWED_BY_TRUST[endpoint_trust]
    denied = tuple(level for level in _SENSITIVITIES if counts[level] and level not in allowed_levels)
    allowed = not denied
    action = "allow" if allowed else ("would_deny" if mode == "audit" else "deny")
    return ContextExportDecision(
        mode=mode,
        evaluated=True,
        action=action,
        allowed=allowed,
        endpoint_trust=endpoint_trust,
        highest_sensitivity=highest,
        block_counts=tuple((level, counts[level]) for level in _SENSITIVITIES if counts[level]),
        denied_sensitivities=denied,
        reason="policy_allowed" if allowed else "context_exceeds_endpoint_trust",
    )


__all__ = [
    "ContextExportDecision",
    "bypass_context_export",
    "context_export_mode",
    "endpoint_context_trust",
    "evaluate_context_export",
]
