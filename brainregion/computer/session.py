"""Controlled one-action-at-a-time computer-use session."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from brainregion.runtime import emit_event

from .adapter import AdapterExecution, ComputerUseAdapter
from .contracts import ActionIntent, ActionReceipt, SceneObservation, VerificationStatus


EventSink = Callable[..., object]


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

    @property
    def actions_executed(self) -> int:
        return self._actions_executed

    def observe(self) -> SceneObservation:
        observation = self.adapter.observe(session_id=self.session_id)
        self._validate_observation(observation)
        self._latest = observation
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

    def perform(self, intent: ActionIntent, *, approved: bool = False) -> ActionReceipt:
        before = self.observe()
        rejection = self._policy_rejection(intent, before=before, approved=approved)
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

    def _policy_rejection(
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
        if intent.expected_frame_id != before.frame.frame_id:
            return "stale", "frame_precondition_failed"
        if intent.expected_state_sha256 != before.state_sha256:
            return "stale", "state_precondition_failed"
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
