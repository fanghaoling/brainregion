"""Deterministic, observable struggle signals for on-demand delegation."""

from __future__ import annotations

from dataclasses import dataclass

from .loop import AdvisoryTriggerState


@dataclass(frozen=True)
class DelegationTriggerDecision:
    activate: bool
    reason: str = ""
    signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class DelegationTriggerPolicy:
    """Wake one expert when progress stalls while enough executor turns remain."""

    min_steps_without_effect: int = 2
    min_remaining_steps: int = 2
    repeated_tool_threshold: int = 2
    repeated_path_threshold: int = 2
    recent_error_threshold: int = 2
    trigger_on_failed_verification: bool = True

    def __post_init__(self) -> None:
        for name in (
            "min_steps_without_effect",
            "min_remaining_steps",
            "repeated_tool_threshold",
            "repeated_path_threshold",
            "recent_error_threshold",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def evaluate(self, state: AdvisoryTriggerState) -> DelegationTriggerDecision:
        if state.remaining_steps < self.min_remaining_steps or state.remaining_cost_usd <= 0:
            return DelegationTriggerDecision(activate=False)

        signals: list[str] = []
        if self.trigger_on_failed_verification and state.last_verification_passed is False:
            signals.append("verification_failed")
        if (
            state.completed_steps >= self.min_steps_without_effect
            and state.steps_since_workspace_effect >= self.min_steps_without_effect
        ):
            signals.append("no_workspace_effect")
        if _tail_repeats(state.recent_tools, self.repeated_tool_threshold):
            signals.append("repeated_tool")
        if _has_repeated_value(state.recent_paths, self.repeated_path_threshold):
            signals.append("repeated_path")
        if state.recent_errors >= self.recent_error_threshold:
            signals.append("repeated_error")

        if not signals:
            return DelegationTriggerDecision(activate=False)
        return DelegationTriggerDecision(
            activate=True,
            reason=signals[0],
            signals=tuple(signals),
        )

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "min_steps_without_effect": self.min_steps_without_effect,
            "min_remaining_steps": self.min_remaining_steps,
            "repeated_tool_threshold": self.repeated_tool_threshold,
            "repeated_path_threshold": self.repeated_path_threshold,
            "recent_error_threshold": self.recent_error_threshold,
            "trigger_on_failed_verification": self.trigger_on_failed_verification,
        }


def _tail_repeats(values: tuple[str, ...], threshold: int) -> bool:
    return len(values) >= threshold and len(set(values[-threshold:])) == 1


def _has_repeated_value(values: tuple[str, ...], threshold: int) -> bool:
    return any(values.count(value) >= threshold for value in set(values))


__all__ = ["DelegationTriggerDecision", "DelegationTriggerPolicy"]
