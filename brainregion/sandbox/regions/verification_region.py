"""Grounded verification option for the code-regime sandbox.

The region does not execute commands. It observes a successful workspace
effect, proposes the symbolic ``run_check`` action, then consumes the bounded
result produced by the host's allow-listed ``workspace_run_check`` driver.
"""
from __future__ import annotations

from typing import Any


class VerificationOptionRegion:
    name = "verification"
    access_mode = "grounded"
    uses_model = False

    def __init__(self) -> None:
        self._pending_effect: str | None = None
        self._verified_effects: set[str] = set()
        self.last_status = "idle"
        self.last_decision = "uninitialized"
        self.confidence = 0.0
        self.last_result: dict[str, Any] = {}

    def next_action(self, observation: Any) -> str | None:
        effect_id = _effect_id(observation)
        if effect_id in self._verified_effects:
            self.last_decision = "effect_already_verified"
            return None
        self._pending_effect = effect_id
        self.last_decision = "run_allowlisted_check"
        self.confidence = 1.0
        return "run_check"

    def observe_transition(self, *, action: str, observation: Any, status: str) -> None:
        if action != "run_check":
            raise ValueError(f"unsupported verification action: {action!r}")
        if self._pending_effect is not None:
            self._verified_effects.add(self._pending_effect)
        self.last_status = status
        self.last_result = _result_summary(observation)
        self.last_decision = "tests_passed" if status == "passed" else "tests_failed"
        self.confidence = 1.0 if status in {"passed", "failed"} else 0.5

    def option_boundary(self, observation: Any, *, actions_executed: int) -> str | None:
        del observation
        return "verification_complete" if actions_executed > 0 else None

    def snapshot(self) -> dict[str, Any]:
        return {
            "policy": "allowlisted_test_after_workspace_effect",
            "access_mode": self.access_mode,
            "verified_effects": len(self._verified_effects),
            "last_status": self.last_status,
            "last_decision": self.last_decision,
            "confidence": self.confidence,
            "last_result": self.last_result,
        }


def _effect_id(observation: Any) -> str:
    if not isinstance(observation, dict):
        raise TypeError("verification observation must be a dict")
    effect_id = str(observation.get("effect_id") or "").strip()
    if not effect_id:
        raise ValueError("verification observation missing effect_id")
    return effect_id


def _result_summary(observation: Any) -> dict[str, Any]:
    if not isinstance(observation, dict):
        return {"status": "invalid_result"}
    return {
        "ok": bool(observation.get("ok", False)),
        "status": observation.get("status"),
        "kind": observation.get("kind"),
        "exit_code": observation.get("exit_code"),
        "duration_ms": observation.get("duration_ms"),
        "timed_out": bool(observation.get("timed_out", False)),
    }


__all__ = ["VerificationOptionRegion"]
