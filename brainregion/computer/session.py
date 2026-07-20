"""Controlled one-action-at-a-time computer-use session."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Protocol

from brainregion.runtime import emit_event

from .adapter import (
    AdapterExecution,
    ComputerUseAdapter,
    FocusableComputerUseAdapter,
    FocusNotSupported,
)
from .contracts import ActionIntent, ActionReceipt, SceneObservation, VerificationStatus
from .locator import PanelAnchor
from .perception import PerceptionRegion


EventSink = Callable[..., object]


def _default_monotonic_ms() -> float:
    return time.monotonic() * 1000.0


class FreshnessPolicy(Protocol):
    """How ``perform()`` obtains its before-observation and freshness gate (G plan S1).

    ``StrictFreshness`` re-observes every action and compares the bound frame/state
    against the freshly observed scene (today's behavior, byte-identical). It honestly
    reflects drift but costs an extra observe per action and rejects on VLM jitter.

    ``BoundFreshness`` binds to ``session._latest`` (no re-observe); the intent is bound
    to that same observation, so frame/state checks constructively pass (bound obs IS the
    gate obs → no mix-and-match). Its honest guard is a TTL on when ``_latest`` was
    observed.
    """

    name: str

    def acquire_before(self, session: ComputerUseSession) -> SceneObservation: ...

    def freshness_rejection(
        self, session: ComputerUseSession, intent: ActionIntent, before: SceneObservation
    ) -> tuple[str, str] | None: ...


class StrictFreshness:
    """Re-observe before each action; reject ``stale`` when bound frame/state drift."""

    name = "strict"

    def acquire_before(self, session: ComputerUseSession) -> SceneObservation:
        return session.observe()

    def freshness_rejection(
        self, session: ComputerUseSession, intent: ActionIntent, before: SceneObservation
    ) -> tuple[str, str] | None:
        if intent.expected_frame_id != before.frame.frame_id:
            return "stale", "frame_precondition_failed"
        if intent.expected_state_sha256 != before.state_sha256:
            return "stale", "state_precondition_failed"
        return None


class BoundFreshness:
    """Bind to ``_latest`` (no re-observe); bound intent shares ``_latest`` so frame/state
    constructively pass. Honest guard = TTL on ``_latest_observed_at``.

    This prevents observation mix-and-match (the bound observation IS the gate
    observation). The TTL only bounds exposure to unobserved external mutation — it does
    NOT prove the UI stayed unchanged (a user can still move a window, a transient can
    close, play-mode can toggle within ``ttl_ms``). Use a tight ``ttl_ms``.
    """

    name = "bound"

    def __init__(self, ttl_ms: float | None = None) -> None:
        self.ttl_ms = ttl_ms

    def acquire_before(self, session: ComputerUseSession) -> SceneObservation:
        if session._latest is None:
            return session.observe()  # seed _latest on the first perform
        return session._latest

    def freshness_rejection(
        self, session: ComputerUseSession, intent: ActionIntent, before: SceneObservation
    ) -> tuple[str, str] | None:
        if self.ttl_ms is None or session._latest_observed_at is None:
            return None
        age_ms = session._monotonic_ms() - session._latest_observed_at
        if age_ms > self.ttl_ms:
            return "stale", "ttl_expired"
        return None


STRICT: FreshnessPolicy = StrictFreshness()


class ComputerUseSession:
    """Own policy, freshness checks, budgets and post-action verification."""

    def __init__(
        self,
        *,
        session_id: str,
        adapter: ComputerUseAdapter,
        allowed_apps: Iterable[str],
        allowed_actions: Iterable[str] = ("click", "type_text", "press_key", "wait", "hover"),
        max_actions: int = 50,
        event_sink: EventSink = emit_event,
        freshness: FreshnessPolicy | None = None,
        ttl_ms: float | None = None,
        monotonic_ms: Callable[[], float] | None = None,
    ) -> None:
        self.session_id = str(session_id or "").strip()
        if not self.session_id:
            raise ValueError("session_id cannot be empty")
        self.adapter = adapter
        self.allowed_apps = frozenset(str(app).strip() for app in allowed_apps if str(app).strip())
        self.allowed_actions = frozenset(str(action).strip().casefold() for action in allowed_actions)
        if not self.allowed_apps:
            raise ValueError("allowed_apps cannot be empty")
        if max_actions <= 0:
            raise ValueError("max_actions must be positive")
        self.max_actions = max_actions
        self._event_sink = event_sink
        self._actions_executed = 0
        self._latest: SceneObservation | None = None
        self._latest_observed_at: float | None = None
        self._monotonic_ms = monotonic_ms or _default_monotonic_ms
        self._freshness: FreshnessPolicy = self._resolve_freshness(freshness, ttl_ms)

    @staticmethod
    def _resolve_freshness(freshness: FreshnessPolicy | None, ttl_ms: float | None) -> FreshnessPolicy:
        if freshness is not None:
            return freshness
        if ttl_ms is not None:
            return BoundFreshness(ttl_ms=ttl_ms)
        return STRICT

    @property
    def actions_executed(self) -> int:
        return self._actions_executed

    def observe(self) -> SceneObservation:
        observation = self.adapter.observe(session_id=self.session_id)
        self._validate_observation(observation)
        self._latest = observation
        self._latest_observed_at = self._monotonic_ms()
        self._emit(
            "computer.observed",
            session_id=self.session_id,
            app_id=observation.app_id,
            frame_id=observation.frame.frame_id,
            state_sha256=observation.state_sha256,
            sequence=observation.sequence,
            element_count=len(observation.elements),
        )
        return observation

    def focus(self, anchor: PanelAnchor) -> SceneObservation:
        """Focus a panel by descriptor (缝 2/8): resolve the anchor against the latest
        observation, then crop via the adapter. The main brain uses descriptors, never a
        raw panel_id. ``FocusNotSupported`` if the adapter lacks the capability;
        ``NoRegionForPanel`` (from the adapter) propagates for the caller to fall back.
        """
        if not isinstance(self.adapter, FocusableComputerUseAdapter):
            raise FocusNotSupported(self.adapter.app_id)
        obs = self._latest if self._latest is not None else self.observe()
        status, panel_id = PerceptionRegion().resolve_panel(anchor, obs)
        if status != "resolved" or panel_id is None:
            raise ValueError(f"focus anchor {status}: cannot resolve to a single panel")
        return self.focus_panel_id(panel_id)

    def focus_panel_id(self, panel_id: str) -> SceneObservation:
        """Lower-level focus by internal ``panel_id`` handle (controller path; 缝 5).

        The main brain uses ``focus(anchor)`` with a descriptor; the TargetingController
        resolves the descriptor itself against a specific parent obs and passes the
        handle here. ``NoRegionForPanel`` from the adapter propagates so the controller
        can run reveal-before-focus.
        """
        if not isinstance(self.adapter, FocusableComputerUseAdapter):
            raise FocusNotSupported(self.adapter.app_id)
        focused = self.adapter.observe_focus(session_id=self.session_id, panel_id=panel_id)
        self._validate_observation(focused)
        self._latest = focused
        self._latest_observed_at = self._monotonic_ms()
        self._emit(
            "computer.focused",
            session_id=self.session_id,
            app_id=focused.app_id,
            frame_id=focused.frame.frame_id,
            focus_root_panel_id=focused.focus_root_panel_id,
            element_count=len(focused.elements),
        )
        return focused

    def perform(self, intent: ActionIntent, *, approved: bool = False) -> ActionReceipt:
        before = self._freshness.acquire_before(self)
        rejection = self._common_rejection(intent, before=before, approved=approved)
        if rejection is None:
            rejection = self._freshness.freshness_rejection(self, intent, before)
        if rejection is not None:
            status, reason = rejection
            return self._receipt(intent, before=before, after=before, status=status, reason=reason)

        try:
            result = self.adapter.execute(intent)
            if not isinstance(result, AdapterExecution):
                result = AdapterExecution(False, "invalid_adapter_result")
        except Exception:
            result = AdapterExecution(False, "adapter_exception")
        self._actions_executed += 1
        try:
            after = self.observe()
        except Exception:
            after = before
            result = AdapterExecution(False, "post_action_observation_failed")
        verification = self._verify(intent, after)
        if not result.succeeded:
            status = "failed"
            reason = result.reason or "adapter_execution_failed"
        elif verification == "failed":
            status = "failed"
            reason = "postcondition_failed"
        else:
            status = "executed"
            reason = result.reason or "action_executed"
        return self._receipt(
            intent,
            before=before,
            after=after,
            status=status,
            reason=reason,
            verification=verification,
        )

    def _common_rejection(
        self,
        intent: ActionIntent,
        *,
        before: SceneObservation,
        approved: bool,
    ) -> tuple[str, str] | None:
        if intent.session_id != self.session_id:
            return "rejected", "session_mismatch"
        if intent.app_id != self.adapter.app_id or intent.app_id not in self.allowed_apps:
            return "rejected", "app_not_allowed"
        if intent.action not in self.allowed_actions:
            return "rejected", "action_not_allowed"
        if self._actions_executed >= self.max_actions:
            return "rejected", "action_budget_exhausted"
        if intent.requires_approval or intent.risk == "high":
            if not approved:
                return "rejected", "approval_required"
        # frame/state freshness delegated to FreshnessPolicy.freshness_rejection (G plan S1).
        if intent.target_id is not None:
            target = before.element(intent.target_id)
            if target is None:
                return "rejected", "target_not_found"
            if not target.visible:
                return "rejected", "target_not_visible"
            if not target.enabled:
                return "rejected", "target_not_enabled"
        return None

    @staticmethod
    def _verify(intent: ActionIntent, after: SceneObservation) -> VerificationStatus:
        if intent.verify_element_id is None:
            return "not_requested"
        element = after.element(intent.verify_element_id)
        if element is None:
            return "failed"
        actual = element.attribute_map()
        return "passed" if all(actual.get(name) == value for name, value in intent.verify_attributes) else "failed"

    def _receipt(
        self,
        intent: ActionIntent,
        *,
        before: SceneObservation,
        after: SceneObservation,
        status: str,
        reason: str,
        verification: VerificationStatus = "not_requested",
    ) -> ActionReceipt:
        receipt = ActionReceipt(
            intent_id=intent.intent_id,
            session_id=self.session_id,
            app_id=before.app_id,
            status=status,  # type: ignore[arg-type]
            reason=reason,
            before_frame=before.frame,
            after_frame=after.frame,
            state_changed=before.state_sha256 != after.state_sha256,
            verification=verification,
            action_index=self._actions_executed,
        )
        self._emit(
            "computer.action_receipt",
            **receipt.to_dict(public=True),
            action=intent.action,
            target_id=intent.target_id,
            risk=intent.risk,
            payload_chars=len(intent.payload),
        )
        return receipt

    def _validate_observation(self, observation: SceneObservation) -> None:
        if observation.session_id != self.session_id:
            raise ValueError("adapter returned an observation for another session")
        if observation.app_id != self.adapter.app_id:
            raise ValueError("adapter observation app_id does not match adapter.app_id")

    def _emit(self, event_type: str, **fields: object) -> None:
        self._event_sink(event_type, **fields)
