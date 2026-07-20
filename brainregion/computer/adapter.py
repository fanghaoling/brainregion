"""Host adapter boundary for computer interaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .contracts import ActionIntent, SceneObservation


@dataclass(frozen=True)
class AdapterExecution:
    succeeded: bool
    reason: str = ""


@runtime_checkable
class ComputerUseAdapter(Protocol):
    """Mechanism-only adapter; policy and completion belong to the runtime."""

    app_id: str

    def observe(self, *, session_id: str) -> SceneObservation: ...

    def execute(self, intent: ActionIntent) -> AdapterExecution: ...


@runtime_checkable
class FocusableComputerUseAdapter(ComputerUseAdapter, Protocol):
    """Capability Protocol: adapters that support hierarchical panel focus.

    ``observe_focus`` returns a LOCAL observation scoped to ``panel_id`` + its visible
    descendants (same logical scope across adapters: mock logically crops the panel tree,
    VisionAdapter crops pixels + re-parses). The focus root's ``parent_panel_id`` is
    normalized to None (self-contained focused obs); ancestors ride in
    ``focus_ancestor_path`` as descriptive labels. Not all adapters support focus — the
    v1 ``MockComputerUseAdapter`` opts out. Gate with
    ``isinstance(adapter, FocusableComputerUseAdapter)`` and raise ``FocusNotSupported``.
    """

    def observe_focus(self, *, session_id: str, panel_id: str) -> SceneObservation: ...


class FocusNotSupported(Exception):
    """Raised when focus is requested on an adapter lacking the ``observe_focus`` capability."""

    def __init__(self, app_id: str) -> None:
        super().__init__(f"adapter {app_id!r} does not support panel focus")
        self.app_id = app_id


class NoRegionForPanel(Exception):
    """Raised by ``observe_focus`` when a panel's crop region cannot be determined.

    Reasons: ``panel_not_found`` (no bbox and no usable cursor anchor), ``below_fold``
    (panel exists but is scrolled out of view — reveal the parent first), ``bbox_missing``
    (panel known but no bbox recorded), ``cursor_anchor_stale`` (anchor consumed/invalid),
    ``session_mismatch``. ``nearest_visible_ancestor_panel_id`` guides reveal-before-focus
    (缝 5); ``source_frame_id``/``source_state_sha256`` identify the observation the panel
    id came from (ids are only unique within one observation — 缝 8).
    """

    def __init__(
        self,
        *,
        panel_id: str,
        reason: str,
        nearest_visible_ancestor_panel_id: str | None = None,
        source_frame_id: str | None = None,
        source_state_sha256: str | None = None,
    ) -> None:
        self.panel_id = panel_id
        self.reason = reason
        self.nearest_visible_ancestor_panel_id = nearest_visible_ancestor_panel_id
        self.source_frame_id = source_frame_id
        self.source_state_sha256 = source_state_sha256
        super().__init__(f"no region for panel {panel_id!r}: {reason}")
