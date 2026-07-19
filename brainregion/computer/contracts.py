"""Strict, content-aware contracts for controlled computer interaction.

The runtime passes screenshot references rather than image bytes.  Raw scene
content and typed values remain private to the adapter/session boundary unless
an explicit export policy allows them to leave the host.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal


Sensitivity = Literal["public", "project", "private", "secret"]
ComputerAction = Literal["click", "type_text", "press_key", "wait"]
ActionRisk = Literal["low", "medium", "high"]
ReceiptStatus = Literal["executed", "rejected", "stale", "failed"]
VerificationStatus = Literal["passed", "failed", "not_requested"]
AttributeValue = str | int | float | bool | None

_SENSITIVITIES = frozenset({"public", "project", "private", "secret"})
_ACTIONS = frozenset({"click", "type_text", "press_key", "wait"})
_RISKS = frozenset({"low", "medium", "high"})
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _required_text(value: Any, name: str, *, max_length: int = 1000) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} cannot be empty")
    if len(text) > max_length:
        raise ValueError(f"{name} cannot exceed {max_length} characters")
    return text


def _identifier(value: Any, name: str) -> str:
    text = _required_text(value, name, max_length=200)
    if any(char.isspace() for char in text):
        raise ValueError(f"{name} cannot contain whitespace")
    return text


def _sha256(value: Any, name: str) -> str:
    text = str(value or "").strip().casefold()
    if not _HEX_SHA256.fullmatch(text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _strict_fields(data: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{name} unknown field(s): {sorted(unknown)}")


def _attributes(value: Any, name: str) -> tuple[tuple[str, AttributeValue], ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    if len(value) > 32:
        raise ValueError(f"{name} cannot contain more than 32 entries")
    output: list[tuple[str, AttributeValue]] = []
    for raw_key, raw_value in value.items():
        key = _identifier(raw_key, f"{name} key")
        if not isinstance(raw_value, (str, int, float, bool, type(None))):
            raise ValueError(f"{name}.{key} must be a JSON scalar")
        if isinstance(raw_value, str) and len(raw_value) > 2000:
            raise ValueError(f"{name}.{key} cannot exceed 2000 characters")
        output.append((key, raw_value))
    return tuple(sorted(output))


@dataclass(frozen=True)
class FrameRef:
    """Reference to a frame owned by the host, never the frame bytes."""

    frame_id: str
    sha256: str
    width: int
    height: int
    artifact_uri: str
    sensitivity: Sensitivity = "private"

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _identifier(self.frame_id, "frame_id"))
        object.__setattr__(self, "sha256", _sha256(self.sha256, "sha256"))
        if self.width <= 0 or self.height <= 0:
            raise ValueError("frame width and height must be positive")
        uri = _required_text(self.artifact_uri, "artifact_uri", max_length=2000)
        if uri.casefold().startswith("data:"):
            raise ValueError("artifact_uri cannot contain inline frame data")
        object.__setattr__(self, "artifact_uri", uri)
        sensitivity = str(self.sensitivity or "").strip().casefold()
        if sensitivity not in _SENSITIVITIES:
            raise ValueError(f"sensitivity must be one of {sorted(_SENSITIVITIES)}")
        object.__setattr__(self, "sensitivity", sensitivity)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameRef":
        if not isinstance(data, dict):
            raise ValueError("frame must be an object")
        _strict_fields(
            data,
            {"frame_id", "sha256", "width", "height", "artifact_uri", "sensitivity"},
            "frame",
        )
        width = int(data.get("width") or 0)
        height = int(data.get("height") or 0)
        if width <= 0 or height <= 0:
            raise ValueError("frame width and height must be positive")
        uri = _required_text(data.get("artifact_uri"), "artifact_uri", max_length=2000)
        if uri.casefold().startswith("data:"):
            raise ValueError("artifact_uri cannot contain inline frame data")
        sensitivity = str(data.get("sensitivity") or "private").strip().casefold()
        if sensitivity not in _SENSITIVITIES:
            raise ValueError(f"sensitivity must be one of {sorted(_SENSITIVITIES)}")
        return cls(
            frame_id=_identifier(data.get("frame_id"), "frame_id"),
            sha256=_sha256(data.get("sha256"), "sha256"),
            width=width,
            height=height,
            artifact_uri=uri,
            sensitivity=sensitivity,  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "artifact_uri": self.artifact_uri,
            "sensitivity": self.sensitivity,
        }

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "sensitivity": self.sensitivity,
            "artifact_uri_redacted": True,
        }


@dataclass(frozen=True)
class UIElement:
    element_id: str
    role: str
    label: str
    enabled: bool = True
    visible: bool = True
    attributes: tuple[tuple[str, AttributeValue], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "element_id", _identifier(self.element_id, "element_id"))
        object.__setattr__(self, "role", _identifier(self.role, "role").casefold())
        object.__setattr__(self, "label", _required_text(self.label, "label", max_length=500))
        object.__setattr__(self, "attributes", _attributes(dict(self.attributes), "attributes"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UIElement":
        if not isinstance(data, dict):
            raise ValueError("element must be an object")
        _strict_fields(data, {"element_id", "role", "label", "enabled", "visible", "attributes"}, "element")
        return cls(
            element_id=_identifier(data.get("element_id"), "element_id"),
            role=_identifier(data.get("role"), "role").casefold(),
            label=_required_text(data.get("label"), "label", max_length=500),
            enabled=bool(data.get("enabled", True)),
            visible=bool(data.get("visible", True)),
            attributes=_attributes(data.get("attributes"), "attributes"),
        )

    def attribute_map(self) -> dict[str, AttributeValue]:
        return dict(self.attributes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "role": self.role,
            "label": self.label,
            "enabled": self.enabled,
            "visible": self.visible,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class SceneObservation:
    session_id: str
    sequence: int
    app_id: str
    window_id: str
    window_title: str
    frame: FrameRef
    state_sha256: str
    elements: tuple[UIElement, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _identifier(self.session_id, "session_id"))
        object.__setattr__(self, "app_id", _identifier(self.app_id, "app_id"))
        object.__setattr__(self, "window_id", _identifier(self.window_id, "window_id"))
        object.__setattr__(self, "window_title", _required_text(self.window_title, "window_title", max_length=1000))
        object.__setattr__(self, "state_sha256", _sha256(self.state_sha256, "state_sha256"))
        if self.sequence < 0:
            raise ValueError("sequence cannot be negative")
        if not isinstance(self.frame, FrameRef):
            raise ValueError("frame must be a FrameRef")
        if any(not isinstance(element, UIElement) for element in self.elements):
            raise ValueError("elements must contain UIElement values")
        ids = [element.element_id for element in self.elements]
        if len(ids) != len(set(ids)):
            raise ValueError("element_id values must be unique within a scene")

    def element(self, element_id: str) -> UIElement | None:
        return next((element for element in self.elements if element.element_id == element_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "sequence": self.sequence,
            "app_id": self.app_id,
            "window_id": self.window_id,
            "window_title": self.window_title,
            "frame": self.frame.to_dict(),
            "state_sha256": self.state_sha256,
            "elements": [element.to_dict() for element in self.elements],
        }

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "sequence": self.sequence,
            "app_id": self.app_id,
            "window_id": self.window_id,
            "frame": self.frame.to_public_dict(),
            "state_sha256": self.state_sha256,
            "element_count": len(self.elements),
            "element_roles": sorted({element.role for element in self.elements}),
            "content_redacted": True,
        }


@dataclass(frozen=True)
class ActionIntent:
    intent_id: str
    session_id: str
    app_id: str
    action: ComputerAction
    expected_frame_id: str
    expected_state_sha256: str
    target_id: str | None = None
    payload: str = ""
    key: str = ""
    wait_ms: int = 0
    risk: ActionRisk = "low"
    requires_approval: bool = False
    verify_element_id: str | None = None
    verify_attributes: tuple[tuple[str, AttributeValue], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent_id", _identifier(self.intent_id, "intent_id"))
        object.__setattr__(self, "session_id", _identifier(self.session_id, "session_id"))
        object.__setattr__(self, "app_id", _identifier(self.app_id, "app_id"))
        action = str(self.action or "").strip().casefold()
        if action not in _ACTIONS:
            raise ValueError(f"action must be one of {sorted(_ACTIONS)}")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "expected_frame_id", _identifier(self.expected_frame_id, "expected_frame_id"))
        object.__setattr__(
            self,
            "expected_state_sha256",
            _sha256(self.expected_state_sha256, "expected_state_sha256"),
        )
        target = str(self.target_id or "").strip() or None
        if action in {"click", "type_text"} and target is None:
            raise ValueError(f"{action} requires target_id")
        object.__setattr__(self, "target_id", _identifier(target, "target_id") if target is not None else None)
        if action == "type_text" and not self.payload:
            raise ValueError("type_text requires payload")
        if len(self.payload) > 4000:
            raise ValueError("payload cannot exceed 4000 characters")
        key = str(self.key or "").strip()
        if action == "press_key" and not key:
            raise ValueError("press_key requires key")
        object.__setattr__(self, "key", key)
        if action == "wait" and not 1 <= self.wait_ms <= 30_000:
            raise ValueError("wait requires wait_ms between 1 and 30000")
        risk = str(self.risk or "").strip().casefold()
        if risk not in _RISKS:
            raise ValueError(f"risk must be one of {sorted(_RISKS)}")
        object.__setattr__(self, "risk", risk)
        verify_element_id = str(self.verify_element_id or "").strip() or None
        normalized_attributes = _attributes(dict(self.verify_attributes), "verify_attributes")
        if normalized_attributes and verify_element_id is None:
            raise ValueError("verify_attributes requires verify_element_id")
        object.__setattr__(
            self,
            "verify_element_id",
            _identifier(verify_element_id, "verify_element_id") if verify_element_id is not None else None,
        )
        object.__setattr__(self, "verify_attributes", normalized_attributes)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionIntent":
        if not isinstance(data, dict):
            raise ValueError("action intent must be an object")
        _strict_fields(
            data,
            {
                "intent_id",
                "session_id",
                "app_id",
                "action",
                "expected_frame_id",
                "expected_state_sha256",
                "target_id",
                "payload",
                "key",
                "wait_ms",
                "risk",
                "requires_approval",
                "verify_element_id",
                "verify_attributes",
            },
            "action intent",
        )
        action = str(data.get("action") or "").strip().casefold()
        if action not in _ACTIONS:
            raise ValueError(f"action must be one of {sorted(_ACTIONS)}")
        risk = str(data.get("risk") or "low").strip().casefold()
        if risk not in _RISKS:
            raise ValueError(f"risk must be one of {sorted(_RISKS)}")
        target = str(data.get("target_id") or "").strip() or None
        payload = str(data.get("payload") or "")
        key = str(data.get("key") or "").strip()
        wait_ms = int(data.get("wait_ms") or 0)
        if action in {"click", "type_text"} and target is None:
            raise ValueError(f"{action} requires target_id")
        if action == "type_text" and not payload:
            raise ValueError("type_text requires payload")
        if action == "press_key" and not key:
            raise ValueError("press_key requires key")
        if action == "wait" and not 1 <= wait_ms <= 30_000:
            raise ValueError("wait requires wait_ms between 1 and 30000")
        if len(payload) > 4000:
            raise ValueError("payload cannot exceed 4000 characters")
        verify_element_id = str(data.get("verify_element_id") or "").strip() or None
        verify_attributes = _attributes(data.get("verify_attributes"), "verify_attributes")
        if verify_attributes and verify_element_id is None:
            raise ValueError("verify_attributes requires verify_element_id")
        return cls(
            intent_id=_identifier(data.get("intent_id"), "intent_id"),
            session_id=_identifier(data.get("session_id"), "session_id"),
            app_id=_identifier(data.get("app_id"), "app_id"),
            action=action,  # type: ignore[arg-type]
            expected_frame_id=_identifier(data.get("expected_frame_id"), "expected_frame_id"),
            expected_state_sha256=_sha256(data.get("expected_state_sha256"), "expected_state_sha256"),
            target_id=_identifier(target, "target_id") if target is not None else None,
            payload=payload,
            key=key,
            wait_ms=wait_ms,
            risk=risk,  # type: ignore[arg-type]
            requires_approval=bool(data.get("requires_approval", False)),
            verify_element_id=(
                _identifier(verify_element_id, "verify_element_id") if verify_element_id is not None else None
            ),
            verify_attributes=verify_attributes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "session_id": self.session_id,
            "app_id": self.app_id,
            "action": self.action,
            "expected_frame_id": self.expected_frame_id,
            "expected_state_sha256": self.expected_state_sha256,
            "target_id": self.target_id,
            "payload": self.payload,
            "key": self.key,
            "wait_ms": self.wait_ms,
            "risk": self.risk,
            "requires_approval": self.requires_approval,
            "verify_element_id": self.verify_element_id,
            "verify_attributes": dict(self.verify_attributes),
        }

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "session_id": self.session_id,
            "app_id": self.app_id,
            "action": self.action,
            "expected_frame_id": self.expected_frame_id,
            "expected_state_sha256": self.expected_state_sha256,
            "target_id": self.target_id,
            "payload_chars": len(self.payload),
            "payload_redacted": bool(self.payload),
            "key": self.key,
            "wait_ms": self.wait_ms,
            "risk": self.risk,
            "requires_approval": self.requires_approval,
            "verify_element_id": self.verify_element_id,
            "verify_attribute_names": [name for name, _value in self.verify_attributes],
        }


@dataclass(frozen=True)
class ActionReceipt:
    intent_id: str
    session_id: str
    app_id: str
    status: ReceiptStatus
    reason: str
    before_frame: FrameRef
    after_frame: FrameRef
    state_changed: bool
    verification: VerificationStatus = "not_requested"
    action_index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent_id", _identifier(self.intent_id, "intent_id"))
        object.__setattr__(self, "session_id", _identifier(self.session_id, "session_id"))
        object.__setattr__(self, "app_id", _identifier(self.app_id, "app_id"))
        if self.status not in {"executed", "rejected", "stale", "failed"}:
            raise ValueError("invalid receipt status")
        if self.verification not in {"passed", "failed", "not_requested"}:
            raise ValueError("invalid verification status")
        if self.action_index < 0:
            raise ValueError("action_index cannot be negative")
        object.__setattr__(self, "reason", _required_text(self.reason, "reason", max_length=500))
        if not isinstance(self.before_frame, FrameRef) or not isinstance(self.after_frame, FrameRef):
            raise ValueError("receipt frames must be FrameRef values")

    def to_dict(self, *, public: bool = False) -> dict[str, Any]:
        frame_serializer = FrameRef.to_public_dict if public else FrameRef.to_dict
        return {
            "intent_id": self.intent_id,
            "session_id": self.session_id,
            "app_id": self.app_id,
            "status": self.status,
            "reason": self.reason,
            "before_frame": frame_serializer(self.before_frame),
            "after_frame": frame_serializer(self.after_frame),
            "state_changed": self.state_changed,
            "verification": self.verification,
            "action_index": self.action_index,
        }
