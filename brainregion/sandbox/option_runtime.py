"""Generic option-region lifecycle primitives for the sandbox runtime.

An option is a bounded burst of region-owned actions. The region proposes and
observes; the host runtime keeps authority over side effects, budgets and
ground-truth completion. ``CognitiveScheduler`` decides *when* to activate from
auditable event facts and contains no navigation-specific policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class OptionRegion(Protocol):
    """Minimal protocol for a bounded action-producing region."""

    name: str
    access_mode: str

    def next_action(self, observation: Any) -> str | None: ...

    def observe_transition(self, *, action: str, observation: Any, status: str) -> None: ...

    def option_boundary(self, observation: Any, *, actions_executed: int) -> str | None: ...

    def snapshot(self) -> dict[str, Any]: ...


@dataclass
class OptionResult:
    region: str
    actor: str
    access_mode: str
    executed_actions: int
    stop_reason: str
    solved: bool
    final_observation: Any = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    region_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "actor": self.actor,
            "access_mode": self.access_mode,
            "executed_actions": self.executed_actions,
            "stop_reason": self.stop_reason,
            "solved": self.solved,
            "final_observation": self.final_observation,
            "trace": self.trace,
            "region_state": self.region_state,
        }


@dataclass
class ActivationRecord:
    trigger: str
    region: str
    access_mode: str
    executed_actions: int
    actions: list[str | None]
    stop_reason: str
    solved: bool
    confidence: float | None = None
    last_decision: str | None = None

    @classmethod
    def from_result(cls, result: OptionResult, *, trigger: str) -> "ActivationRecord":
        return cls(
            trigger=trigger,
            region=result.region,
            access_mode=result.access_mode,
            executed_actions=result.executed_actions,
            actions=[item.get("action") for item in result.trace],
            stop_reason=result.stop_reason,
            solved=result.solved,
            confidence=result.region_state.get("confidence"),
            last_decision=result.region_state.get("last_decision"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger": self.trigger,
            "region": self.region,
            "access_mode": self.access_mode,
            "executed_actions": self.executed_actions,
            "actions": self.actions,
            "stop_reason": self.stop_reason,
            "solved": self.solved,
            "confidence": self.confidence,
            "last_decision": self.last_decision,
        }


@dataclass(frozen=True)
class ActivationDecision:
    activate: bool
    trigger: str = ""
    reason: str = ""


class CognitiveScheduler:
    """Event-driven activation policy independent of region domain logic."""

    def __init__(self, *, continuous: bool = False) -> None:
        self.continuous = bool(continuous)
        self._last_activation_clock = 0

    def initial(self, *, region_available: bool, action_budget: int) -> ActivationDecision:
        if not region_available:
            return ActivationDecision(False, reason="region_unavailable")
        if action_budget <= 0:
            return ActivationDecision(False, reason="no_action_budget")
        return ActivationDecision(True, trigger="initial", reason="region_first")

    def after_environment_change(
        self,
        *,
        action_clock: int,
        last_actor: str | None,
        solved: bool,
        region_available: bool,
        remaining_actions: int | None,
    ) -> ActivationDecision:
        if not self.continuous:
            return ActivationDecision(False, reason="continuous_disabled")
        if not region_available:
            return ActivationDecision(False, reason="region_unavailable")
        if solved:
            return ActivationDecision(False, reason="already_solved")
        if remaining_actions is not None and remaining_actions <= 0:
            return ActivationDecision(False, reason="no_action_budget")
        if action_clock == self._last_activation_clock:
            return ActivationDecision(False, reason="no_new_environment_action")
        if last_actor != "main":
            return ActivationDecision(False, reason="last_actor_not_main")
        return ActivationDecision(True, trigger="after_main_action", reason="main_changed_environment")

    def mark_activated(self, *, action_clock: int) -> None:
        self._last_activation_clock = max(0, int(action_clock))


def select_region_observation(
    region: OptionRegion,
    *,
    public_observation: Any,
    privileged_observation: Any,
) -> Any:
    """Enforce the region access boundary at one explicit choke point."""
    if region.access_mode == "grounded":
        return public_observation
    if region.access_mode == "oracle":
        return privileged_observation
    raise ValueError(f"unsupported option access_mode: {region.access_mode!r}")


__all__ = [
    "ActivationDecision",
    "ActivationRecord",
    "CognitiveScheduler",
    "OptionRegion",
    "OptionResult",
    "select_region_observation",
]
