"""TargetingController: deterministic reveal/retry orchestration.

The main brain only expresses a Locator. If resolution finds the target is below
the fold, this controller applies a reveal strategy (default: ``press_key "end"``)
and re-resolves — keeping the End-key out of orchestration logic and the main brain
away from coordinates/keys. The reveal strategy is injectable so non-Unity adapters
can supply their own (PageDown, scroll-drag, ...).

If a reveal action fails (adapter rejects / unsupported), the controller aborts
immediately with ``status=blocked`` and a ``reveal_failed`` reason rather than
silently spinning to ``max_reveals``.
"""

from __future__ import annotations

from collections.abc import Callable

from .contracts import ActionIntent, ActionReceipt
from .locator import Locator
from .perception import PerceptionRegion, ResolutionResult


RevealStrategy = Callable[..., ActionReceipt]


def default_reveal_strategy(session, obs) -> ActionReceipt:
    """Unity-style reveal: press End to scroll the active panel to the bottom."""
    intent = ActionIntent(
        intent_id=f"reveal-{obs.frame.frame_id}",
        session_id=obs.session_id,
        app_id=obs.app_id,
        action="press_key",
        expected_frame_id=obs.frame.frame_id,
        expected_state_sha256=obs.state_sha256,
        key="end",
    )
    return session.perform(intent)


class TargetingController:
    def __init__(
        self,
        *,
        session,
        perception: PerceptionRegion,
        max_reveals: int = 3,
        reveal_strategy: RevealStrategy | None = None,
    ) -> None:
        if max_reveals < 0:
            raise ValueError("max_reveals cannot be negative")
        self._session = session
        self._perception = perception
        self._max_reveals = max_reveals
        self._reveal = reveal_strategy or default_reveal_strategy

    def target(self, locator: Locator) -> ResolutionResult:
        obs = self._session.observe()
        result = self._perception.resolve(locator, obs)
        if result.status == "resolved":
            return result
        # Only below_fold is reveal-recoverable; disabled/hidden/ambiguous/not_found/unsupported
        # are returned as-is for the caller to decide.
        if result.status != "blocked" or result.first is None or result.first.blocker != "below_fold":
            return result

        last = result
        for _ in range(self._max_reveals):
            receipt = self._reveal(self._session, obs)
            if receipt.status != "executed":
                return ResolutionResult(
                    status="blocked",
                    candidates=last.candidates,
                    trace=last.trace,
                    reason=f"reveal_failed:{receipt.reason}",
                )
            obs = self._session.observe()
            last = self._perception.resolve(locator, obs)
            if last.status in {"resolved", "ambiguous", "not_found", "unsupported"}:
                return last
            if last.first is None or last.first.blocker != "below_fold":
                return last  # blocked for a non-fold reason
        return last  # max_reveals exhausted without resolving
