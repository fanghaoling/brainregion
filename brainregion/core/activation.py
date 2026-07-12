"""Structured region activation contracts and deterministic wake planning.

The hard gate answers two separate questions without calling a model:

1. Is this skill/region eligible to wake for the current structured signal?
2. Which bounded context selectors should be loaded if it wakes?

Ambiguous matches return ``defer``. A cheap semantic gate can resolve those in
a later phase without changing the deterministic prerequisite/deny boundary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal

WakeAction = Literal["wake", "skip", "defer"]
ActivationMode = Literal["context", "advisory", "action"]
CostTier = Literal["free", "low", "medium", "high"]

_ACTIVATION_MODES = frozenset({"context", "advisory", "action"})
_COST_TIERS = frozenset({"free", "low", "medium", "high"})
_COST_RANK = {"free": 0, "low": 1, "medium": 2, "high": 3}


def _strings(value: Any, field_name: str) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    values = value if isinstance(value, (list, tuple, set)) else [value]
    out: list[str] = []
    for item in values:
        text = str(item).strip().casefold()
        if text and text not in out:
            out.append(text)
    return tuple(out)


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"activation {field_name} must be an object")
    return value


def _intersection(left: tuple[str, ...], right: tuple[str, ...]) -> list[str]:
    return sorted(set(left) & set(right))


@dataclass(frozen=True)
class ActivationSignal:
    """Small structured facts visible to a sleeping gate."""

    task_intents: tuple[str, ...] = ()
    events: tuple[str, ...] = ()
    target_apps: tuple[str, ...] = ()
    running_apps: tuple[str, ...] = ()
    available_tools: tuple[str, ...] = ()
    available_capabilities: tuple[str, ...] = ()
    cooldowns: dict[str, int] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ActivationSignal":
        data = data or {}
        if not isinstance(data, dict):
            raise ValueError("activation signal must be an object")
        attributes = data.get("attributes") or {}
        if not isinstance(attributes, dict):
            raise ValueError("activation signal attributes must be an object")
        raw_cooldowns = data.get("cooldowns") or {}
        if not isinstance(raw_cooldowns, dict):
            raise ValueError("activation signal cooldowns must be an object")
        cooldowns: dict[str, int] = {}
        for skill_id, remaining in raw_cooldowns.items():
            if isinstance(remaining, bool) or not isinstance(remaining, int) or remaining < 0:
                raise ValueError("activation signal cooldown values must be non-negative integers")
            cooldowns[str(skill_id).strip().casefold()] = remaining
        return cls(
            task_intents=_strings(data.get("task_intents"), "task_intents"),
            events=_strings(data.get("events"), "events"),
            target_apps=_strings(data.get("target_apps"), "target_apps"),
            running_apps=_strings(data.get("running_apps"), "running_apps"),
            available_tools=_strings(data.get("available_tools"), "available_tools"),
            available_capabilities=_strings(
                data.get("available_capabilities"), "available_capabilities"
            ),
            cooldowns=cooldowns,
            attributes=dict(attributes),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_intents": list(self.task_intents),
            "events": list(self.events),
            "target_apps": list(self.target_apps),
            "running_apps": list(self.running_apps),
            "available_tools": list(self.available_tools),
            "available_capabilities": list(self.available_capabilities),
            "cooldowns": dict(self.cooldowns),
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class ActivationContract:
    """Typed activation metadata attached to one existing SkillManifest."""

    skill_id: str
    region: str
    capabilities: tuple[str, ...] = ()
    wake_task_intents: tuple[str, ...] = ()
    wake_events: tuple[str, ...] = ()
    wake_target_apps: tuple[str, ...] = ()
    deny_task_intents: tuple[str, ...] = ()
    deny_events: tuple[str, ...] = ()
    deny_target_apps: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    require_running_app: bool = False
    context_selectors: tuple[str, ...] = ()
    max_context_tokens: int = 1200
    activation_mode: ActivationMode = "context"
    cost_tier: CostTier = "low"
    min_signal_groups: int = 1
    confidence_threshold: float = 0.0
    cooldown_steps: int = 0

    @classmethod
    def from_dict(
        cls,
        *,
        skill_id: str,
        region: str,
        data: dict[str, Any],
    ) -> "ActivationContract":
        if not isinstance(data, dict):
            raise ValueError(f"skill {skill_id!r} activation metadata must be an object")
        skill_id = str(skill_id).strip()
        region = str(region).strip()
        if not skill_id or not region:
            raise ValueError("activation contract skill_id and region must be non-empty")

        wake = _mapping(data.get("wake_when"), "wake_when")
        deny = _mapping(data.get("do_not_wake_when"), "do_not_wake_when")
        context = _mapping(data.get("context"), "context")
        known_root = {
            "capabilities", "wake_when", "do_not_wake_when", "context",
            "context_selectors", "max_context_tokens", "activation_mode", "cost_tier",
            "min_signal_groups", "confidence_threshold", "cooldown_steps",
        }
        known_wake = {
            "task_intents", "events", "target_apps", "required_tools", "required_capabilities",
        }
        known_deny = {
            "task_intents", "events", "target_apps", "missing_capabilities", "app_not_running",
        }
        for scope, values, known in (
            ("activation", data, known_root),
            ("wake_when", wake, known_wake),
            ("do_not_wake_when", deny, known_deny),
            ("context", context, {"selectors", "max_tokens"}),
        ):
            unknown = set(values) - known
            if unknown:
                raise ValueError(
                    f"skill {skill_id!r} activation {scope} unknown field(s): {sorted(unknown)}"
                )
        if "app_not_running" in deny and not isinstance(deny["app_not_running"], bool):
            raise ValueError(
                f"skill {skill_id!r} activation app_not_running must be a boolean"
            )

        mode = str(data.get("activation_mode") or "context").strip().casefold()
        if mode not in _ACTIVATION_MODES:
            raise ValueError(
                f"skill {skill_id!r} activation_mode {mode!r} not in {sorted(_ACTIVATION_MODES)}"
            )
        cost_tier = str(data.get("cost_tier") or "low").strip().casefold()
        if cost_tier not in _COST_TIERS:
            raise ValueError(
                f"skill {skill_id!r} cost_tier {cost_tier!r} not in {sorted(_COST_TIERS)}"
            )

        max_tokens = data.get("max_context_tokens", context.get("max_tokens", 1200))
        min_groups = data.get("min_signal_groups", 1)
        cooldown = data.get("cooldown_steps", 0)
        for value, name, lo, hi in (
            (max_tokens, "max_context_tokens", 1, 1_000_000),
            (min_groups, "min_signal_groups", 1, 3),
            (cooldown, "cooldown_steps", 0, 1_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
                raise ValueError(f"skill {skill_id!r} activation {name} must be in [{lo}, {hi}]")

        threshold = data.get("confidence_threshold", 0.0)
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise ValueError(
                f"skill {skill_id!r} activation confidence_threshold must be in [0, 1]"
            )

        required_caps = list(_strings(wake.get("required_capabilities"), "required_capabilities"))
        for capability in _strings(deny.get("missing_capabilities"), "missing_capabilities"):
            if capability not in required_caps:
                required_caps.append(capability)

        configured_groups = sum(bool(wake.get(key)) for key in ("task_intents", "events", "target_apps"))
        if configured_groups == 0:
            raise ValueError(f"skill {skill_id!r} activation requires at least one wake signal group")
        if min_groups > configured_groups:
            raise ValueError(
                f"skill {skill_id!r} min_signal_groups={min_groups} exceeds "
                f"configured signal groups={configured_groups}"
            )

        return cls(
            skill_id=skill_id,
            region=region,
            capabilities=_strings(data.get("capabilities"), "capabilities"),
            wake_task_intents=_strings(wake.get("task_intents"), "wake.task_intents"),
            wake_events=_strings(wake.get("events"), "wake.events"),
            wake_target_apps=_strings(wake.get("target_apps"), "wake.target_apps"),
            deny_task_intents=_strings(deny.get("task_intents"), "deny.task_intents"),
            deny_events=_strings(deny.get("events"), "deny.events"),
            deny_target_apps=_strings(deny.get("target_apps"), "deny.target_apps"),
            required_tools=_strings(wake.get("required_tools"), "required_tools"),
            required_capabilities=tuple(required_caps),
            require_running_app=bool(deny.get("app_not_running", False)),
            context_selectors=_strings(
                data.get("context_selectors", context.get("selectors")), "context_selectors"
            ),
            max_context_tokens=max_tokens,
            activation_mode=mode,  # type: ignore[arg-type]
            cost_tier=cost_tier,  # type: ignore[arg-type]
            min_signal_groups=min_groups,
            confidence_threshold=float(threshold),
            cooldown_steps=cooldown,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "region": self.region,
            "capabilities": list(self.capabilities),
            "wake_when": {
                "task_intents": list(self.wake_task_intents),
                "events": list(self.wake_events),
                "target_apps": list(self.wake_target_apps),
                "required_tools": list(self.required_tools),
                "required_capabilities": list(self.required_capabilities),
            },
            "do_not_wake_when": {
                "task_intents": list(self.deny_task_intents),
                "events": list(self.deny_events),
                "target_apps": list(self.deny_target_apps),
                "app_not_running": self.require_running_app,
            },
            "context_selectors": list(self.context_selectors),
            "max_context_tokens": self.max_context_tokens,
            "activation_mode": self.activation_mode,
            "cost_tier": self.cost_tier,
            "min_signal_groups": self.min_signal_groups,
            "confidence_threshold": self.confidence_threshold,
            "cooldown_steps": self.cooldown_steps,
        }


@dataclass(frozen=True)
class ContextRequest:
    skill_id: str
    region: str
    selectors: tuple[str, ...]
    max_tokens: int
    activation_mode: ActivationMode
    cost_tier: CostTier
    cooldown_steps: int
    matched_signals: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "region": self.region,
            "selectors": list(self.selectors),
            "max_tokens": self.max_tokens,
            "activation_mode": self.activation_mode,
            "cost_tier": self.cost_tier,
            "cooldown_steps": self.cooldown_steps,
            "matched_signals": dict(self.matched_signals),
        }


@dataclass(frozen=True)
class RegionWakeDecision:
    skill_id: str
    region: str
    action: WakeAction
    score: float
    reasons: tuple[str, ...] = ()
    matched_signals: dict[str, list[str]] = field(default_factory=dict)
    missing_requirements: tuple[str, ...] = ()
    cooldown_remaining: int = 0
    context_request: ContextRequest | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "region": self.region,
            "action": self.action,
            "score": round(self.score, 4),
            "reasons": list(self.reasons),
            "matched_signals": dict(self.matched_signals),
            "missing_requirements": list(self.missing_requirements),
            "cooldown_remaining": self.cooldown_remaining,
            "context_request": self.context_request.to_dict() if self.context_request else None,
        }


def evaluate_activation(
    contract: ActivationContract,
    signal: ActivationSignal,
) -> RegionWakeDecision:
    """Evaluate one contract using only structured facts; never calls a model."""
    denied = {
        "task_intents": _intersection(contract.deny_task_intents, signal.task_intents),
        "events": _intersection(contract.deny_events, signal.events),
        "target_apps": _intersection(contract.deny_target_apps, signal.target_apps),
    }
    denied = {key: value for key, value in denied.items() if value}
    if denied:
        return RegionWakeDecision(
            contract.skill_id,
            contract.region,
            "skip",
            0.0,
            reasons=("forbidden_signal",),
            matched_signals=denied,
        )

    cooldown_remaining = int(signal.cooldowns.get(contract.skill_id.casefold(), 0) or 0)
    if cooldown_remaining > 0:
        return RegionWakeDecision(
            contract.skill_id,
            contract.region,
            "skip",
            0.0,
            reasons=("cooldown_active",),
            cooldown_remaining=cooldown_remaining,
        )

    missing: list[str] = []
    for tool in contract.required_tools:
        if tool not in signal.available_tools:
            missing.append(f"tool:{tool}")
    for capability in contract.required_capabilities:
        if capability not in signal.available_capabilities:
            missing.append(f"capability:{capability}")
    if contract.require_running_app:
        expected_apps = contract.wake_target_apps or signal.target_apps
        if not expected_apps or not set(expected_apps) & set(signal.running_apps):
            missing.append("running_app")
    if missing:
        return RegionWakeDecision(
            contract.skill_id,
            contract.region,
            "skip",
            0.0,
            reasons=("missing_requirement",),
            missing_requirements=tuple(missing),
        )

    configured = {
        "task_intents": contract.wake_task_intents,
        "events": contract.wake_events,
        "target_apps": contract.wake_target_apps,
    }
    matched = {
        "task_intents": _intersection(contract.wake_task_intents, signal.task_intents),
        "events": _intersection(contract.wake_events, signal.events),
        "target_apps": _intersection(contract.wake_target_apps, signal.target_apps),
    }
    configured_groups = sum(bool(value) for value in configured.values())
    matched = {key: value for key, value in matched.items() if value}
    matched_groups = len(matched)
    score = matched_groups / configured_groups if configured_groups else 0.0
    if matched_groups == 0:
        return RegionWakeDecision(
            contract.skill_id,
            contract.region,
            "skip",
            score,
            reasons=("no_positive_signal",),
        )

    if matched_groups < contract.min_signal_groups or score < contract.confidence_threshold:
        return RegionWakeDecision(
            contract.skill_id,
            contract.region,
            "defer",
            score,
            reasons=("ambiguous_match",),
            matched_signals=matched,
        )

    request = ContextRequest(
        skill_id=contract.skill_id,
        region=contract.region,
        selectors=contract.context_selectors,
        max_tokens=contract.max_context_tokens,
        activation_mode=contract.activation_mode,
        cost_tier=contract.cost_tier,
        cooldown_steps=contract.cooldown_steps,
        matched_signals=matched,
    )
    return RegionWakeDecision(
        contract.skill_id,
        contract.region,
        "wake",
        score,
        reasons=("hard_gate_passed",),
        matched_signals=matched,
        context_request=request,
    )


@dataclass(frozen=True)
class ActivationPlan:
    decisions: tuple[RegionWakeDecision, ...]
    woken_regions: tuple[str, ...]
    context_requests: tuple[ContextRequest, ...]
    trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": [decision.to_dict() for decision in self.decisions],
            "woken_regions": list(self.woken_regions),
            "context_requests": [request.to_dict() for request in self.context_requests],
            "trace": dict(self.trace),
        }


def plan_activation(
    contracts: list[ActivationContract],
    signal: ActivationSignal,
    *,
    max_regions: int = 3,
    max_context_tokens: int = 4000,
) -> ActivationPlan:
    """Plan bounded hard-gate activations; ``defer`` remains asleep for now."""
    if isinstance(max_regions, bool) or not isinstance(max_regions, int) or max_regions < 0:
        raise ValueError("max_regions must be a non-negative integer")
    if (
        isinstance(max_context_tokens, bool)
        or not isinstance(max_context_tokens, int)
        or max_context_tokens < 0
    ):
        raise ValueError("max_context_tokens must be a non-negative integer")

    by_skill: dict[str, ActivationContract] = {}
    for contract in contracts:
        if contract.skill_id in by_skill:
            raise ValueError(f"duplicate activation skill_id {contract.skill_id!r}")
        by_skill[contract.skill_id] = contract

    evaluated = [evaluate_activation(contract, signal) for contract in contracts]
    candidates = sorted(
        (decision for decision in evaluated if decision.action == "wake"),
        key=lambda decision: (
            -decision.score,
            _COST_RANK[by_skill[decision.skill_id].cost_tier],
            decision.skill_id,
        ),
    )
    selected_skills: set[str] = set()
    woken_regions: list[str] = []
    requests: list[ContextRequest] = []
    remaining_tokens = max_context_tokens
    budget_reasons: dict[str, str] = {}

    for decision in candidates:
        if decision.region not in woken_regions and len(woken_regions) >= max_regions:
            budget_reasons[decision.skill_id] = "region_budget_exceeded"
            continue
        request = decision.context_request
        if request is None or remaining_tokens <= 0:
            budget_reasons[decision.skill_id] = "context_budget_exceeded"
            continue
        allocated = min(request.max_tokens, remaining_tokens)
        if allocated <= 0:
            continue
        selected_skills.add(decision.skill_id)
        if decision.region not in woken_regions:
            woken_regions.append(decision.region)
        requests.append(replace(request, max_tokens=allocated))
        remaining_tokens -= allocated

    decisions: list[RegionWakeDecision] = []
    for decision in evaluated:
        if decision.action == "wake" and decision.skill_id not in selected_skills:
            reason = budget_reasons.get(decision.skill_id, "context_budget_exceeded")
            decisions.append(
                replace(
                    decision,
                    action="skip",
                    reasons=(reason,),
                    context_request=None,
                    score=decision.score,
                )
            )
        else:
            decisions.append(decision)

    return ActivationPlan(
        decisions=tuple(decisions),
        woken_regions=tuple(woken_regions),
        context_requests=tuple(requests),
        trace={
            "strategy": "activation_contract_hard_gate_v1",
            "models_called": False,
            "contracts_evaluated": len(contracts),
            "deferred": [d.skill_id for d in decisions if d.action == "defer"],
            "max_regions": max_regions,
            "max_context_tokens": max_context_tokens,
            "allocated_context_tokens": sum(r.max_tokens for r in requests),
            "remaining_context_tokens": remaining_tokens,
        },
    )


__all__ = [
    "ActivationContract",
    "ActivationPlan",
    "ActivationSignal",
    "ContextRequest",
    "RegionWakeDecision",
    "evaluate_activation",
    "plan_activation",
]
