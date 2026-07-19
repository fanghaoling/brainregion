"""Deterministic in-process app for computer-use contract testing."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .adapter import AdapterExecution
from .contracts import ActionIntent, FrameRef, SceneObservation, UIElement


class MockComputerUseAdapter:
    app_id = "mock.settings"

    def __init__(self) -> None:
        self._sequence = 0
        self._state: dict[str, Any] = {
            "notifications": False,
            "username": "",
            "saved": False,
            "dialog_open": False,
        }
        self.execution_count = 0

    def observe(self, *, session_id: str) -> SceneObservation:
        self._sequence += 1
        digest = self._state_digest()
        return SceneObservation(
            session_id=session_id,
            sequence=self._sequence,
            app_id=self.app_id,
            window_id="settings-main",
            window_title="Mock Settings",
            frame=FrameRef(
                frame_id=f"frame-{digest[:16]}",
                sha256=digest,
                width=800,
                height=600,
                artifact_uri=f"mock://frames/{digest}",
                sensitivity="private",
            ),
            state_sha256=digest,
            elements=self._elements(),
        )

    def execute(self, intent: ActionIntent) -> AdapterExecution:
        self.execution_count += 1
        if intent.action == "click" and intent.target_id == "notifications-toggle":
            self._state["notifications"] = not self._state["notifications"]
            self._state["saved"] = False
            return AdapterExecution(True, "toggle_changed")
        if intent.action == "type_text" and intent.target_id == "username-input":
            self._state["username"] = intent.payload
            self._state["saved"] = False
            return AdapterExecution(True, "text_entered")
        if intent.action == "click" and intent.target_id == "save-button":
            self._state["saved"] = True
            self._state["dialog_open"] = True
            return AdapterExecution(True, "settings_saved")
        if intent.action == "press_key" and intent.key.casefold() == "escape":
            self._state["dialog_open"] = False
            return AdapterExecution(True, "dialog_closed")
        if intent.action == "wait":
            return AdapterExecution(True, "wait_completed")
        return AdapterExecution(False, "unsupported_mock_action")

    def mutate_out_of_band(self, **changes: Any) -> None:
        """Simulate user/app state changes between planning and execution."""
        unknown = set(changes) - set(self._state)
        if unknown:
            raise ValueError(f"unknown mock state: {sorted(unknown)}")
        self._state.update(changes)

    def _state_digest(self) -> str:
        encoded = json.dumps(self._state, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _elements(self) -> tuple[UIElement, ...]:
        elements = [
            UIElement(
                element_id="notifications-toggle",
                role="checkbox",
                label="Notifications",
                attributes=(("checked", bool(self._state["notifications"])),),
            ),
            UIElement(
                element_id="username-input",
                role="text_input",
                label="Username",
                attributes=(("value", str(self._state["username"])),),
            ),
            UIElement(
                element_id="save-button",
                role="button",
                label="Save",
            ),
        ]
        if self._state["dialog_open"]:
            elements.append(
                UIElement(
                    element_id="saved-dialog",
                    role="dialog",
                    label="Settings saved",
                    attributes=(("open", True),),
                )
            )
        return tuple(elements)
