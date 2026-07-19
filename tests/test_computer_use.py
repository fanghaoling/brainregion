"""Controlled Computer Use contracts and deterministic host-loop tests."""

from __future__ import annotations

import pytest

from brainregion.computer import (
    ActionIntent,
    ComputerUseSession,
    FrameRef,
    MockComputerUseAdapter,
)


def _intent(observation, **overrides) -> ActionIntent:
    data = {
        "intent_id": "intent-1",
        "session_id": observation.session_id,
        "app_id": observation.app_id,
        "action": "click",
        "target_id": "notifications-toggle",
        "expected_frame_id": observation.frame.frame_id,
        "expected_state_sha256": observation.state_sha256,
    }
    data.update(overrides)
    return ActionIntent.from_dict(data)


def _session(*, max_actions: int = 50, allowed_actions=("click", "type_text", "press_key", "wait")):
    events: list[dict] = []

    def record(event_type: str, **fields):
        events.append({"type": event_type, **fields})

    adapter = MockComputerUseAdapter()
    session = ComputerUseSession(
        session_id="session-1",
        adapter=adapter,
        allowed_apps={adapter.app_id},
        allowed_actions=allowed_actions,
        max_actions=max_actions,
        event_sink=record,
    )
    return session, adapter, events


def test_frame_ref_rejects_inline_image_data_and_redacts_artifact_uri():
    digest = "a" * 64
    with pytest.raises(ValueError, match="inline frame data"):
        FrameRef.from_dict(
            {
                "frame_id": "frame-1",
                "sha256": digest,
                "width": 10,
                "height": 10,
                "artifact_uri": "data:image/png;base64,AAAA",
            }
        )

    frame = FrameRef.from_dict(
        {
            "frame_id": "frame-1",
            "sha256": digest,
            "width": 10,
            "height": 10,
            "artifact_uri": "artifact://frames/frame-1",
            "sensitivity": "private",
        }
    )

    assert frame.to_dict()["artifact_uri"] == "artifact://frames/frame-1"
    assert "artifact_uri" not in frame.to_public_dict()
    assert frame.to_public_dict()["artifact_uri_redacted"] is True

    with pytest.raises(ValueError, match="inline frame data"):
        FrameRef(
            frame_id="frame-2",
            sha256=digest,
            width=10,
            height=10,
            artifact_uri="data:image/png;base64,AAAA",
        )


def test_session_executes_semantic_action_and_verifies_result_without_leaking_payload():
    session, adapter, events = _session()
    observation = session.observe()
    intent = _intent(
        observation,
        action="type_text",
        target_id="username-input",
        payload="private-user-value",
        verify_element_id="username-input",
        verify_attributes={"value": "private-user-value"},
    )

    receipt = session.perform(intent)

    assert receipt.status == "executed"
    assert receipt.state_changed is True
    assert receipt.verification == "passed"
    assert adapter.execution_count == 1
    assert session.actions_executed == 1
    assert "private-user-value" not in repr(events)
    action_event = next(event for event in events if event["type"] == "computer.action_receipt")
    assert action_event["payload_chars"] == len("private-user-value")


def test_out_of_band_change_makes_planned_action_stale_before_adapter_execution():
    session, adapter, _events = _session()
    observation = session.observe()
    intent = _intent(observation)
    adapter.mutate_out_of_band(username="changed elsewhere")

    receipt = session.perform(intent)

    assert receipt.status == "stale"
    assert receipt.reason == "frame_precondition_failed"
    assert receipt.state_changed is False
    assert adapter.execution_count == 0
    assert session.actions_executed == 0


def test_high_risk_action_requires_approval_and_reuses_unchanged_precondition():
    session, adapter, _events = _session()
    observation = session.observe()
    intent = _intent(observation, risk="high")

    denied = session.perform(intent)
    accepted = session.perform(intent, approved=True)

    assert denied.status == "rejected"
    assert denied.reason == "approval_required"
    assert accepted.status == "executed"
    assert adapter.execution_count == 1


def test_action_allowlist_and_budget_are_enforced_by_session_not_adapter():
    session, adapter, _events = _session(max_actions=1, allowed_actions=("click",))
    observation = session.observe()
    disallowed = _intent(
        observation,
        action="type_text",
        target_id="username-input",
        payload="not executed",
    )

    denied = session.perform(disallowed)
    first = session.perform(_intent(session.observe(), intent_id="intent-2"))
    second = session.perform(_intent(session.observe(), intent_id="intent-3"))

    assert denied.reason == "action_not_allowed"
    assert first.status == "executed"
    assert second.status == "rejected"
    assert second.reason == "action_budget_exhausted"
    assert adapter.execution_count == 1


def test_failed_postcondition_is_not_reported_as_success():
    session, adapter, _events = _session()
    observation = session.observe()
    intent = _intent(
        observation,
        verify_element_id="notifications-toggle",
        verify_attributes={"checked": False},
    )

    receipt = session.perform(intent)

    assert receipt.status == "failed"
    assert receipt.reason == "postcondition_failed"
    assert receipt.verification == "failed"
    assert adapter.execution_count == 1


def test_adapter_exception_produces_failed_receipt_and_consumes_budget():
    session, adapter, _events = _session(max_actions=1)
    observation = session.observe()

    def fail(_intent):
        raise RuntimeError("sensitive adapter detail")

    adapter.execute = fail  # type: ignore[method-assign]
    receipt = session.perform(_intent(observation))

    assert receipt.status == "failed"
    assert receipt.reason == "adapter_exception"
    assert session.actions_executed == 1


def test_public_observation_exposes_shape_not_window_or_element_content():
    session, _adapter, _events = _session()
    public = session.observe().to_public_dict()

    assert public["app_id"] == "mock.settings"
    assert public["element_count"] == 3
    assert public["content_redacted"] is True
    assert "window_title" not in public
    assert "elements" not in public
