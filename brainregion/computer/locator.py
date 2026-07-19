"""Locator: relative/path-based UI targeting (never absolute coordinates).

The main brain refers to UI by a ``Locator`` (panel anchor + within-panel relative
position + semantic descriptor). A locator resolves to ``element_id``(s) against a
specific ``SceneObservation``; the resolved binding (``ResolvedTarget``) carries the
observation's ``frame_id``/``state_sha256`` so the resulting ``ActionIntent`` is
freshness-bound to that observation by construction.

Pure data only — no ``resolve`` method here (avoids a circular import with the
perception layer). Resolution lives in ``perception.py``.

Freshness note: ``ResolvedTarget`` is a convenience that binds element_id +
expected_frame_id + expected_state_sha256 from one observation, eliminating the
accidental mix-and-match bug (element from obs A, frame from obs B). It is not a
cryptographic guarantee against deliberate fabrication — direct construction is
possible but unsupported; use ``ResolvedTarget.from_observation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import ActionIntent, AttributeValue, UIElement
from .validation import attributes as _attributes, identifier as _identifier


_BANDS = frozenset({"top", "near_top", "middle", "middle_bottom", "bottom"})
_RELATIONS = frozenset({"below", "above", "beside"})
_BLOCKERS = frozenset({"disabled", "hidden", "below_fold"})


def _norm(value: Any) -> str:
    return str(value or "").strip().casefold()


@dataclass(frozen=True)
class ElementDescriptor:
    """Semantic identity of an element. At least one field must be set."""

    role: str | None = None
    label: str | None = None
    attributes: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        role = self.role
        if role is not None:
            object.__setattr__(self, "role", _norm(role))
        label = self.label
        if label is not None:
            label = _norm(label)
            object.__setattr__(self, "label", label)
        normalized = _attributes(dict(self.attributes), "attributes")
        object.__setattr__(self, "attributes", normalized)
        if self.role is None and self.label is None and not self.attributes:
            raise ValueError("ElementDescriptor requires at least one of role/label/attributes")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ElementDescriptor":
        if not isinstance(data, dict):
            raise ValueError("element descriptor must be an object")
        unknown = set(data) - {"role", "label", "attributes"}
        if unknown:
            raise ValueError(f"element descriptor unknown field(s): {sorted(unknown)}")
        role = data.get("role")
        label = data.get("label")
        return cls(
            role=_norm(role) if role is not None else None,
            label=_norm(label) if label is not None else None,
            attributes=_attributes(data.get("attributes"), "attributes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "label": self.label, "attributes": dict(self.attributes)}

    def matches(self, element: UIElement) -> bool:
        """role normalized-equal, label normalized-substring, attributes subset-equal."""
        if self.role is not None and element.role != self.role:
            return False
        if self.label is not None and self.label not in _norm(element.label):
            return False
        if self.attributes:
            actual = element.attribute_map()
            for key, value in self.attributes:
                if actual.get(key) != value:
                    return False
        return True


@dataclass(frozen=True)
class PanelAnchor:
    """Which container. Multiple conditions are AND. At least one must be set.

    ``ordinal`` is a spatial hint declared by the adapter (e.g. "leftmost"),
    never pixel-derived. ``just_opened`` is NOT an ordinal here — it is resolved
    by the perception layer via ``Panel.spawn_sequence`` (max wins; ties ambiguous).
    """

    panel_name: str | None = None
    ordinal: str | None = None
    transient_kind: str | None = None
    spawned_by: ElementDescriptor | None = None

    def __post_init__(self) -> None:
        panel_name = self.panel_name
        if panel_name is not None:
            object.__setattr__(self, "panel_name", _norm(panel_name))
        ordinal = self.ordinal
        if ordinal is not None:
            object.__setattr__(self, "ordinal", _norm(ordinal))
        kind = self.transient_kind
        if kind is not None:
            object.__setattr__(self, "transient_kind", _norm(kind))
        if self.spawned_by is not None and not isinstance(self.spawned_by, ElementDescriptor):
            raise ValueError("spawned_by must be an ElementDescriptor")
        if not any([self.panel_name, self.ordinal, self.transient_kind, self.spawned_by]):
            raise ValueError(
                "PanelAnchor requires at least one of panel_name/ordinal/transient_kind/spawned_by"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PanelAnchor":
        if not isinstance(data, dict):
            raise ValueError("panel anchor must be an object")
        unknown = set(data) - {"panel_name", "ordinal", "transient_kind", "spawned_by"}
        if unknown:
            raise ValueError(f"panel anchor unknown field(s): {sorted(unknown)}")
        panel_name = data.get("panel_name")
        ordinal = data.get("ordinal")
        kind = data.get("transient_kind")
        spawned = data.get("spawned_by")
        return cls(
            panel_name=_norm(panel_name) if panel_name is not None else None,
            ordinal=_norm(ordinal) if ordinal is not None else None,
            transient_kind=_norm(kind) if kind is not None else None,
            spawned_by=ElementDescriptor.from_dict(spawned) if spawned is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel_name": self.panel_name,
            "ordinal": self.ordinal,
            "transient_kind": self.transient_kind,
            "spawned_by": self.spawned_by.to_dict() if self.spawned_by is not None else None,
        }


@dataclass(frozen=True)
class WithinPanel:
    """Position within a panel. ``relation`` and ``relative_to`` come together.

    ``band`` resolves by element order within the panel (top→bottom = first→last)
    unless the element overrides via ``semantic_band``. Mock supports
    ``below``/``above``; ``beside`` returns ``unsupported_relation`` at resolve time
    (needs geometry the mock does not carry).
    """

    band: str | None = None
    relation: str | None = None
    relative_to: ElementDescriptor | None = None

    def __post_init__(self) -> None:
        band = self.band
        if band is not None:
            band = _norm(band)
            if band not in _BANDS:
                raise ValueError(f"band must be one of {sorted(_BANDS)}")
            object.__setattr__(self, "band", band)
        relation = self.relation
        if relation is not None:
            relation = _norm(relation)
            if relation not in _RELATIONS:
                raise ValueError(f"relation must be one of {sorted(_RELATIONS)}")
            object.__setattr__(self, "relation", relation)
        has_rel = relation is not None
        has_ref = self.relative_to is not None
        if has_rel != has_ref:
            raise ValueError("relation and relative_to must both be set or both be None")
        if self.relative_to is not None and not isinstance(self.relative_to, ElementDescriptor):
            raise ValueError("relative_to must be an ElementDescriptor")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WithinPanel":
        if not isinstance(data, dict):
            raise ValueError("within-panel must be an object")
        unknown = set(data) - {"band", "relation", "relative_to"}
        if unknown:
            raise ValueError(f"within-panel unknown field(s): {sorted(unknown)}")
        band = data.get("band")
        relation = data.get("relation")
        ref = data.get("relative_to")
        band_n = _norm(band) if band is not None else None
        if band_n is not None and band_n not in _BANDS:
            raise ValueError(f"band must be one of {sorted(_BANDS)}")
        rel_n = _norm(relation) if relation is not None else None
        if rel_n is not None and rel_n not in _RELATIONS:
            raise ValueError(f"relation must be one of {sorted(_RELATIONS)}")
        if (rel_n is not None) != (ref is not None):
            raise ValueError("relation and relative_to must both be set or both be None")
        return cls(
            band=band_n,
            relation=rel_n,
            relative_to=ElementDescriptor.from_dict(ref) if ref is not None else None,  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "band": self.band,
            "relation": self.relation,
            "relative_to": self.relative_to.to_dict() if self.relative_to is not None else None,
        }


@dataclass(frozen=True)
class Locator:
    anchor: PanelAnchor
    within: WithinPanel | None = None
    descriptor: ElementDescriptor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.anchor, PanelAnchor):
            raise ValueError("anchor must be a PanelAnchor")
        if self.within is not None and not isinstance(self.within, WithinPanel):
            raise ValueError("within must be a WithinPanel")
        if self.descriptor is not None and not isinstance(self.descriptor, ElementDescriptor):
            raise ValueError("descriptor must be an ElementDescriptor")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Locator":
        if not isinstance(data, dict):
            raise ValueError("locator must be an object")
        unknown = set(data) - {"anchor", "within", "descriptor"}
        if unknown:
            raise ValueError(f"locator unknown field(s): {sorted(unknown)}")
        anchor = data.get("anchor")
        within = data.get("within")
        descriptor = data.get("descriptor")
        if anchor is None:
            raise ValueError("locator.anchor is required")
        return cls(
            anchor=PanelAnchor.from_dict(anchor),
            within=WithinPanel.from_dict(within) if within is not None else None,
            descriptor=ElementDescriptor.from_dict(descriptor) if descriptor is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor": self.anchor.to_dict(),
            "within": self.within.to_dict() if self.within is not None else None,
            "descriptor": self.descriptor.to_dict() if self.descriptor is not None else None,
        }


@dataclass(frozen=True)
class ResolutionStep:
    """One stage of a resolution trace (user-visible; never sent to telemetry)."""

    stage: str  # "anchor" | "within" | "descriptor"
    detail: str
    matched: int


@dataclass(frozen=True)
class ResolvedTarget:
    """A target resolved against a specific observation.

    ``frame_id``/``state_sha256`` come from the observation the locator was
    resolved against, so ``to_action_intent`` produces an intent whose freshness
    preconditions are bound to that same observation (no accidental cross-obs mix).
    """

    element_id: str
    panel_id: str | None
    frame_id: str
    state_sha256: str
    available: bool
    blocker: str | None
    trace: tuple[ResolutionStep, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "element_id", _identifier(self.element_id, "element_id"))
        if self.panel_id is not None:
            object.__setattr__(self, "panel_id", _identifier(self.panel_id, "panel_id"))
        object.__setattr__(self, "frame_id", _identifier(self.frame_id, "frame_id"))
        object.__setattr__(self, "state_sha256", str(self.state_sha256 or "").strip().casefold())
        if not self.state_sha256:
            raise ValueError("state_sha256 cannot be empty")
        blocker = self.blocker
        if blocker is not None:
            blocker = _norm(blocker)
            if blocker not in _BLOCKERS:
                raise ValueError(f"blocker must be one of {sorted(_BLOCKERS)} or None")
            object.__setattr__(self, "blocker", blocker)
        if not self.available and blocker is None:
            raise ValueError("unavailable target requires a blocker reason")
        if self.available and blocker is not None:
            raise ValueError("available target cannot have a blocker")

    @classmethod
    def from_observation(
        cls,
        *,
        element: UIElement,
        frame_id: str,
        state_sha256: str,
        available: bool,
        blocker: str | None,
        trace: tuple[ResolutionStep, ...] = (),
    ) -> "ResolvedTarget":
        """Recommended construction: binds element_id + frame/state from one observation."""
        return cls(
            element_id=element.element_id,
            panel_id=element.panel_id,
            frame_id=frame_id,
            state_sha256=state_sha256,
            available=available,
            blocker=blocker,
            trace=trace,
        )

    def to_action_intent(
        self,
        *,
        action: str,
        intent_id: str,
        session_id: str,
        app_id: str,
        payload: str = "",
        key: str = "",
        wait_ms: int = 0,
        risk: str = "low",
        requires_approval: bool = False,
        button: str = "left",
        verify_element_id: str | None = None,
        verify_attributes: tuple[tuple[str, AttributeValue], ...] = (),
    ) -> ActionIntent:
        """Build an ActionIntent bound to this target's observation (same-source freshness)."""
        return ActionIntent(
            intent_id=intent_id,
            session_id=session_id,
            app_id=app_id,
            action=action,
            expected_frame_id=self.frame_id,
            expected_state_sha256=self.state_sha256,
            button=button,
            target_id=self.element_id,
            payload=payload,
            key=key,
            wait_ms=wait_ms,
            risk=risk,
            requires_approval=requires_approval,
            verify_element_id=verify_element_id,
            verify_attributes=verify_attributes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "panel_id": self.panel_id,
            "frame_id": self.frame_id,
            "state_sha256": self.state_sha256,
            "available": self.available,
            "blocker": self.blocker,
        }
