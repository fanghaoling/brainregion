"""TargetingController: deterministic reveal/retry orchestration.

The main brain only expresses a Locator. If resolution finds the target is below
the fold, this controller applies a reveal strategy (default: ``press_key "end"``)
and re-resolves — keeping the End-key out of orchestration logic and the main brain
away from coordinates/keys. The reveal strategy is injectable so non-Unity adapters
can supply their own (PageDown, scroll-drag, ...).

If a reveal action fails (adapter rejects / unsupported), the controller aborts
immediately with ``status=blocked`` and a ``reveal_failed`` reason rather than
silently spinning to ``max_reveals``.

``focus_chain`` (缝 5) drives a focus stack into arbitrarily deep nested windows:
each descriptor is re-resolved against the current observation every step (ids are
per-frame, never cached — 缝 8), and ``NoRegionForPanel(below_fold)`` triggers
reveal-before-focus on the parent panel — keep the parent obs, reveal the nearest
visible ancestor, re-observe, re-resolve the descriptor, retry — all bounded by a
``FocusBudget`` with explicit termination reasons and a semantic (label) path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .adapter import NoRegionForPanel
from .contracts import ActionIntent, ActionReceipt, SceneObservation
from .locator import Locator, PanelAnchor
from .perception import PerceptionRegion, ResolutionResult


RevealStrategy = Callable[..., ActionReceipt]


def default_reveal_strategy(session, obs, *, panel_id: str | None = None) -> ActionReceipt:
    """Unity-style reveal: press End to scroll the active panel to the bottom.

    ``panel_id`` (缝 5) lets a scoped strategy target a specific panel; the default
    ignores it (press End scrolls whatever panel is focused). Injectable for non-Unity
    adapters (PageDown, scroll-drag, ...).
    """
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


FocusOutcome = Literal[
    "focused",
    "focus_depth_exceeded",
    "reveal_budget_exhausted",
    "focus_cycle_detected",
    "panel_not_rediscovered",
    "region_still_unavailable",
]


@dataclass(frozen=True)
class FocusBudget:
    """Bounded resource envelope for a focus chain (缝 5).

    Defaults are generous for real UIs but tight enough to guarantee termination on
    pathological adapters (cycles, panels that never reveal).
    """

    max_focus_depth: int = 8
    max_reveals_per_level: int = 2
    max_total_reveals: int = 6
    max_total_observations: int = 12


@dataclass(frozen=True)
class FocusChainResult:
    """Outcome of ``focus_chain``. ``semantic_path`` carries descriptive labels
    (Inspector → Transform → Position); ids are per-frame so only labels persist across
    the chain (缝 8). ``observation`` is the final focused obs on success (or the deepest
    obs reached on a partial failure).
    """

    observation: SceneObservation | None
    outcome: FocusOutcome
    semantic_path: tuple[str, ...]
    reason: str = ""
    reveals_used: int = 0
    observations_used: int = 0
    depth: int = 0


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

    def focus_chain(
        self,
        anchors,
        *,
        budget: FocusBudget | None = None,
    ) -> FocusChainResult:
        """Drive a focus stack to a deeply-nested panel (缝 5).

        ``anchors`` is an ordered chain of ``PanelAnchor`` descriptors, one per level
        (e.g. Inspector → Transform → Position). Each descriptor is re-resolved against
        the CURRENT observation every step — ids are per-frame and never cached (缝 8).
        When ``observe_focus`` raises ``NoRegionForPanel(below_fold)``, the controller
        keeps the parent obs, runs a scoped reveal on the nearest visible ancestor,
        re-observes, re-resolves the descriptor, and retries — bounded by ``budget``.
        """
        budget = budget or FocusBudget()
        ordered = tuple(anchors)
        if not ordered:
            raise ValueError("focus_chain requires at least one anchor")

        reveals_used = 0
        observations_used = 0
        semantic_path: list[str] = []
        seen_labels: set[str] = set()

        current = self._session.observe()
        observations_used += 1

        for depth, anchor in enumerate(ordered, start=1):
            if depth > budget.max_focus_depth:
                return FocusChainResult(
                    current,
                    "focus_depth_exceeded",
                    tuple(semantic_path),
                    reason=f"depth {depth} exceeds max_focus_depth {budget.max_focus_depth}",
                    reveals_used=reveals_used,
                    observations_used=observations_used,
                    depth=len(semantic_path),
                )

            focused, outcome, reason, reveals_added, obs_added = self._focus_one_level(
                anchor, current, budget, reveals_used, observations_used
            )
            reveals_used += reveals_added
            observations_used += obs_added

            if outcome is not None:
                return FocusChainResult(
                    focused,
                    outcome,
                    tuple(semantic_path),
                    reason=reason,
                    reveals_used=reveals_used,
                    observations_used=observations_used,
                    depth=len(semantic_path),
                )

            root_id = focused.focus_root_panel_id
            label = focused.panel(root_id).label if root_id is not None else ""
            # semantic cycle detection: labels are descriptive, so a revisit means the
            # chain looped (real cycles repeat a label). Same-label siblings are
            # unrealistic in Unity-style UIs, so label-based detection suffices.
            if label in seen_labels:
                return FocusChainResult(
                    focused,
                    "focus_cycle_detected",
                    tuple(semantic_path) + (label,),
                    reason=f"label {label!r} revisited at depth {depth}",
                    reveals_used=reveals_used,
                    observations_used=observations_used,
                    depth=len(semantic_path),
                )
            seen_labels.add(label)
            semantic_path.append(label)
            current = focused

        return FocusChainResult(
            current,
            "focused",
            tuple(semantic_path),
            reveals_used=reveals_used,
            observations_used=observations_used,
            depth=len(semantic_path),
        )

    def _focus_one_level(
        self,
        anchor: PanelAnchor,
        parent_obs: SceneObservation,
        budget: FocusBudget,
        reveals_used: int,
        observations_used: int,
    ) -> tuple[SceneObservation | None, FocusOutcome | None, str, int, int]:
        """Resolve ``anchor`` against ``parent_obs`` and focus it, with reveal-before-focus.

        Returns ``(focused_or_None, outcome_or_None, reason, reveals_added,
        observations_added)``. ``outcome`` is ``None`` on success.
        """
        reveals_added = 0
        obs_added = 0

        # re-resolve descriptor against the parent obs (缝 8: no cached id)
        status, panel_id = self._perception.resolve_panel(anchor, parent_obs)
        if status != "resolved" or panel_id is None:
            return None, "panel_not_rediscovered", f"anchor {status}", 0, 0

        while True:
            try:
                focused = self._session.focus_panel_id(panel_id)
                return focused, None, "", reveals_added, obs_added
            except NoRegionForPanel as exc:
                if exc.reason != "below_fold":
                    return (
                        None,
                        "region_still_unavailable",
                        f"no_region:{exc.reason}",
                        reveals_added,
                        obs_added,
                    )
                # reveal-before-focus on the parent (GPT④): keep parent obs, scoped reveal
                # on the nearest visible ancestor, re-observe, re-resolve descriptor (NOT
                # the cached id — 缝 8), then retry observe_focus.
                if reveals_used + reveals_added >= budget.max_total_reveals:
                    return (
                        None,
                        "reveal_budget_exhausted",
                        "max_total_reveals reached",
                        reveals_added,
                        obs_added,
                    )
                if reveals_added >= budget.max_reveals_per_level:
                    return (
                        None,
                        "reveal_budget_exhausted",
                        "max_reveals_per_level reached",
                        reveals_added,
                        obs_added,
                    )
                receipt = self._reveal(self._session, parent_obs, panel_id=exc.nearest_visible_ancestor_panel_id)
                reveals_added += 1
                if receipt.status != "executed":
                    return (
                        None,
                        "region_still_unavailable",
                        f"reveal_failed:{receipt.reason}",
                        reveals_added,
                        obs_added,
                    )
                if observations_used + obs_added >= budget.max_total_observations:
                    return (
                        None,
                        "reveal_budget_exhausted",
                        "max_total_observations reached",
                        reveals_added,
                        obs_added,
                    )
                parent_obs = self._session.observe()
                obs_added += 1
                status, panel_id = self._perception.resolve_panel(anchor, parent_obs)
                if status != "resolved" or panel_id is None:
                    return (
                        None,
                        "panel_not_rediscovered",
                        f"anchor {status} after reveal",
                        reveals_added,
                        obs_added,
                    )
                # loop retries observe_focus with the freshly-resolved id
