"""TargetingController reveal/retry + freshness-binding integration tests."""

from __future__ import annotations

import hashlib

import pytest

from brainregion.computer.adapter import AdapterExecution, NoRegionForPanel
from brainregion.computer.contracts import (
    ActionIntent,
    ActionReceipt,
    FrameRef,
    Panel,
    SceneObservation,
    UIElement,
)
from brainregion.computer.locator import (
    ElementDescriptor,
    Locator,
    PanelAnchor,
    WithinPanel,
)
from brainregion.computer.perception import PerceptionRegion
from brainregion.computer.session import ComputerUseSession
from brainregion.computer.targeting import FocusBudget, TargetingController
from brainregion.computer.unity_mock import UnityEditorMockAdapter


_DIGEST = "a" * 64


def _intent(target_id, *, action="click", button="left", payload="", key="") -> ActionIntent:
    return ActionIntent(
        intent_id="i",
        session_id="s",
        app_id="unity.editor",
        action=action,
        expected_frame_id="f",
        expected_state_sha256=_DIGEST,
        target_id=target_id,
        button=button,
        payload=payload,
        key=key,
    )


def _setup() -> tuple[UnityEditorMockAdapter, ComputerUseSession, PerceptionRegion]:
    adapter = UnityEditorMockAdapter()
    adapter.execute(_intent("hierarchy-background", button="right"))
    adapter.execute(_intent("menu-3d-object", action="hover"))
    adapter.execute(_intent("submenu-cube"))
    session = ComputerUseSession(session_id="s", adapter=adapter, allowed_apps={"unity.editor"})
    perception = PerceptionRegion(event_sink=lambda *a, **k: None)
    return adapter, session, perception


def _add_component_locator() -> Locator:
    return Locator(
        anchor=PanelAnchor(panel_name="inspector", ordinal="rightmost"),
        within=WithinPanel(band="bottom"),
        descriptor=ElementDescriptor(role="button", label="add comp"),
    )


def test_target_below_fold_is_revealed_then_resolved():
    _adapter, session, perception = _setup()
    controller = TargetingController(session=session, perception=perception, max_reveals=3)
    result = controller.target(_add_component_locator())
    assert result.status == "resolved"
    assert result.first.element_id == "inspector-add-component"


def test_target_already_available_skips_reveal():
    _adapter, session, perception = _setup()
    controller = TargetingController(session=session, perception=perception)
    locator = Locator(
        anchor=PanelAnchor(panel_name="toolbar"),
        descriptor=ElementDescriptor(role="button", attributes=(("icon_shape", "play"),)),
    )
    result = controller.target(locator)
    assert result.status == "resolved"
    assert result.first.element_id == "toolbar-play"


def test_target_reveal_failure_aborts_immediately_with_reason():
    adapter, session, perception = _setup()

    def failing_reveal(_session, obs):
        return ActionReceipt(
            intent_id="x",
            session_id="s",
            app_id="unity.editor",
            status="failed",
            reason="reveal_unsupported",
            before_frame=obs.frame,
            after_frame=obs.frame,
            state_changed=False,
        )

    controller = TargetingController(
        session=session,
        perception=perception,
        max_reveals=3,
        reveal_strategy=failing_reveal,
    )
    result = controller.target(_add_component_locator())
    assert result.status == "blocked"
    assert "reveal_failed" in result.reason


def test_resolved_target_to_action_intent_passes_session_freshness():
    _adapter, session, perception = _setup()
    controller = TargetingController(session=session, perception=perception, max_reveals=3)
    result = controller.target(_add_component_locator())
    assert result.status == "resolved"
    intent = result.first.to_action_intent(action="click", intent_id="act1", session_id="s", app_id="unity.editor")
    receipt = session.perform(intent)
    assert receipt.status == "executed"


def test_stale_after_out_of_band_change_between_resolve_and_perform():
    adapter, session, perception = _setup()
    controller = TargetingController(session=session, perception=perception, max_reveals=3)
    result = controller.target(_add_component_locator())
    intent = result.first.to_action_intent(action="click", intent_id="act1", session_id="s", app_id="unity.editor")
    adapter.mutate_out_of_band(component_search="changed")  # state changes out of band
    receipt = session.perform(intent)
    assert receipt.status == "stale"


def test_end_to_end_create_cube_and_add_rigidbody_via_locators():
    """Full stack capstone: locator -> controller -> session performs the whole flow.

    Exercises the user's worked example end-to-end: right-click Hierarchy blank,
    hover 3D Object (just-opened context menu), click Cube (submenu spawned by 3D
    Object), reveal + click Add Component (Inspector bottom), type search, click
    Rigidbody result. The main brain only ever holds Locators; coordinates/keys
    never appear in the orchestration.
    """
    adapter = UnityEditorMockAdapter()
    session = ComputerUseSession(session_id="s", adapter=adapter, allowed_apps={"unity.editor"})
    perception = PerceptionRegion(event_sink=lambda *a, **k: None)
    controller = TargetingController(session=session, perception=perception)

    def act(locator, *, action, intent_id, button="left", payload=""):
        result = controller.target(locator)
        assert result.status == "resolved", f"{intent_id}: {result.status} {result.reason}"
        intent = result.first.to_action_intent(
            action=action,
            intent_id=intent_id,
            session_id="s",
            app_id="unity.editor",
            button=button,
            payload=payload,
        )
        receipt = session.perform(intent)
        assert receipt.status == "executed", f"{intent_id}: {receipt.status} {receipt.reason}"

    # 1. right-click Hierarchy blank -> context menu
    act(
        Locator(
            anchor=PanelAnchor(panel_name="hierarchy"),
            descriptor=ElementDescriptor(role="canvas", label="hierarchy blank"),
        ),
        action="click",
        intent_id="open-menu",
        button="right",
    )
    # 2. hover 3D Object (the just-opened context menu)
    act(
        Locator(
            anchor=PanelAnchor(transient_kind="context_menu", ordinal="just_opened"),
            descriptor=ElementDescriptor(role="menu_item", label="3d object"),
        ),
        action="hover",
        intent_id="hover-3dobject",
    )
    # 3. click Cube (submenu spawned by the 3D Object item)
    act(
        Locator(
            anchor=PanelAnchor(
                transient_kind="submenu",
                spawned_by=ElementDescriptor(label="3d object"),
            ),
            descriptor=ElementDescriptor(role="menu_item", label="cube"),
        ),
        action="click",
        intent_id="create-cube",
    )
    assert adapter._state["selected_object_id"] == "cube-1"
    # 4. add Rigidbody (Add Component is at Inspector bottom, below fold -> controller reveals)
    act(
        Locator(
            anchor=PanelAnchor(panel_name="inspector", ordinal="rightmost"),
            within=WithinPanel(band="bottom"),
            descriptor=ElementDescriptor(role="button", label="add comp"),
        ),
        action="click",
        intent_id="open-acp",
    )
    act(
        Locator(
            anchor=PanelAnchor(transient_kind="popup", ordinal="just_opened"),
            descriptor=ElementDescriptor(
                role="text_input",
                attributes=(("icon_shape", "question_mark"),),
            ),
        ),
        action="type_text",
        intent_id="type-rigidbody",
        payload="rigidbody",
    )
    act(
        Locator(
            anchor=PanelAnchor(transient_kind="popup"),
            within=WithinPanel(relation="below", relative_to=ElementDescriptor(role="text_input")),
            descriptor=ElementDescriptor(role="list_item", label="rigidbody"),
        ),
        action="click",
        intent_id="pick-rigidbody",
    )
    assert "Rigidbody" in adapter._state["scene_objects"][0]["components"]


# --- 缝 5: nested focus chain + reveal-before-focus + FocusBudget ---


class _NestedRevealAdapter:
    """Test-only FocusableComputerUseAdapter for 缝 5 controller tests.

    Models inspector → transform → position. Position is below-fold until a reveal
    (press End) flips a flag. State is content-derived from ``reveal_count`` so the
    digest is stable across consecutive observes (session.perform freshness passes),
    but panel ids carry the ``reveal_count`` suffix — so the controller MUST re-resolve
    the descriptor each step and cannot cache the id across the reveal (缝 8).
    """

    app_id = "fake.nested"

    def __init__(self, *, position_block_reason: str = "below_fold", never_reveals: bool = False) -> None:
        self._reveal_count = 0
        self._position_block_reason = position_block_reason
        self._never_reveals = never_reveals
        self.focus_calls: list[str] = []  # panel_ids handed to observe_focus (no-cache probe)

    def _digest(self) -> str:
        return hashlib.sha256(f"state-{self._reveal_count}".encode("utf-8")).hexdigest()

    def _suffix(self) -> str:
        return str(self._reveal_count)

    def _panels(self) -> tuple[Panel, ...]:
        s = self._suffix()
        return (
            Panel(panel_id=f"inspector-{s}", role="inspector", label="Inspector"),
            Panel(
                panel_id=f"transform-{s}",
                role="inspector_component",
                label="Transform",
                parent_panel_id=f"inspector-{s}",
            ),
            Panel(
                panel_id=f"position-{s}",
                role="inspector_component",
                label="Position",
                parent_panel_id=f"transform-{s}",
            ),
        )

    def _elements(self) -> tuple[UIElement, ...]:
        s = self._suffix()
        return (UIElement(element_id=f"position-{s}-x", role="number_field", label="X", panel_id=f"position-{s}"),)

    def observe(self, *, session_id: str) -> SceneObservation:
        digest = self._digest()
        return SceneObservation(
            session_id=session_id,
            sequence=1,
            app_id=self.app_id,
            window_id="w",
            window_title="T",
            frame=FrameRef(
                frame_id=f"frame-{digest[:8]}",
                sha256=digest,
                width=10,
                height=10,
                artifact_uri="mock://x",
            ),
            state_sha256=digest,
            elements=self._elements(),
            panels=self._panels(),
        )

    def observe_focus(self, *, session_id: str, panel_id: str) -> SceneObservation:
        self.focus_calls.append(panel_id)
        position_available = self._reveal_count > 0 and not self._never_reveals
        if panel_id.startswith("position-") and not position_available:
            s = self._suffix()
            digest = self._digest()
            raise NoRegionForPanel(
                panel_id=panel_id,
                reason=self._position_block_reason,
                nearest_visible_ancestor_panel_id=f"transform-{s}",
                source_frame_id=f"frame-{digest[:8]}",
                source_state_sha256=digest,
            )
        full = self.observe(session_id=session_id)
        return full.focused_view(panel_id)

    def execute(self, intent: ActionIntent) -> AdapterExecution:
        if intent.action == "press_key" and intent.key == "end":
            self._reveal_count += 1
            return AdapterExecution(True, "scrolled_to_bottom")
        return AdapterExecution(False, "unsupported")


def _chain_controller(adapter: _NestedRevealAdapter) -> TargetingController:
    session = ComputerUseSession(session_id="s", adapter=adapter, allowed_apps={"fake.nested"})
    perception = PerceptionRegion(event_sink=lambda *a, **k: None)
    return TargetingController(session=session, perception=perception)


def test_focus_chain_drives_to_depth_3_with_reveal_before_focus():
    adapter = _NestedRevealAdapter()
    controller = _chain_controller(adapter)
    result = controller.focus_chain(
        [
            PanelAnchor(panel_name="inspector"),
            PanelAnchor(panel_name="transform"),
            PanelAnchor(panel_name="position"),
        ]
    )
    assert result.outcome == "focused"
    assert result.semantic_path == ("Inspector", "Transform", "Position")
    assert result.depth == 3
    # Position was below-fold → exactly one scoped reveal at the position level
    assert result.reveals_used == 1
    # the final focused obs is the post-reveal Position (suffix 1), self-contained
    assert result.observation is not None
    assert result.observation.focus_root_panel_id == "position-1"
    assert result.observation.panel("position-1").parent_panel_id is None


def test_focus_chain_re_resolves_id_after_reveal_no_cache():
    # 缝 8: ids are per-frame; the controller must re-resolve the descriptor after a
    # reveal rather than reuse the stale pre-reveal id.
    adapter = _NestedRevealAdapter()
    controller = _chain_controller(adapter)
    result = controller.focus_chain(
        [
            PanelAnchor(panel_name="inspector"),
            PanelAnchor(panel_name="transform"),
            PanelAnchor(panel_name="position"),
        ]
    )
    assert result.outcome == "focused"
    assert "position-0" in adapter.focus_calls  # first attempt (blocked, below-fold)
    assert "position-1" in adapter.focus_calls  # re-resolved id after the reveal


def test_focus_chain_exhausts_reveal_budget_when_position_never_reveals():
    adapter = _NestedRevealAdapter(never_reveals=True)
    controller = _chain_controller(adapter)
    result = controller.focus_chain(
        [
            PanelAnchor(panel_name="inspector"),
            PanelAnchor(panel_name="transform"),
            PanelAnchor(panel_name="position"),
        ]
    )
    assert result.outcome == "reveal_budget_exhausted"
    assert result.semantic_path == ("Inspector", "Transform")  # Position never appended
    assert result.reveals_used == 2  # max_reveals_per_level cap


def test_focus_chain_depth_exceeded():
    adapter = _NestedRevealAdapter()
    controller = _chain_controller(adapter)
    result = controller.focus_chain(
        [
            PanelAnchor(panel_name="inspector"),
            PanelAnchor(panel_name="transform"),
            PanelAnchor(panel_name="position"),
        ],
        budget=FocusBudget(max_focus_depth=2),
    )
    assert result.outcome == "focus_depth_exceeded"
    assert result.semantic_path == ("Inspector", "Transform")


def test_focus_chain_detects_cycle_on_repeated_label():
    adapter = _NestedRevealAdapter()
    controller = _chain_controller(adapter)
    result = controller.focus_chain(
        [
            PanelAnchor(panel_name="inspector"),
            PanelAnchor(panel_name="transform"),
            PanelAnchor(panel_name="transform"),  # same label revisited → cycle
        ]
    )
    assert result.outcome == "focus_cycle_detected"
    assert result.semantic_path == ("Inspector", "Transform", "Transform")


def test_focus_chain_panel_not_rediscovered():
    adapter = _NestedRevealAdapter()
    controller = _chain_controller(adapter)
    result = controller.focus_chain([PanelAnchor(panel_name="inspector"), PanelAnchor(panel_name="ghost")])
    assert result.outcome == "panel_not_rediscovered"
    assert result.semantic_path == ("Inspector",)


def test_focus_chain_region_still_unavailable_for_non_fold_reason():
    # a non-below_fold NoRegionForPanel (bbox_missing) is not reveal-recoverable
    adapter = _NestedRevealAdapter(position_block_reason="bbox_missing")
    controller = _chain_controller(adapter)
    result = controller.focus_chain(
        [
            PanelAnchor(panel_name="inspector"),
            PanelAnchor(panel_name="transform"),
            PanelAnchor(panel_name="position"),
        ]
    )
    assert result.outcome == "region_still_unavailable"
    assert "bbox_missing" in result.reason
    assert result.reveals_used == 0  # no reveal attempted for a non-fold block


def test_focus_chain_empty_anchors_raises():
    adapter = _NestedRevealAdapter()
    controller = _chain_controller(adapter)
    with pytest.raises(ValueError):
        controller.focus_chain([])


# --- 阶段 7 集成 capstone (缝 8 确认: 主脑全程只用描述符, locator.py diff==0) ---


def test_capstone_focus_chain_to_position_x_with_reveal_and_resolve():
    """§206 D 集成 capstone — full stack to a depth-3 element, descriptors only.

    Inspector → Transform → Position X, WITH reveal-before-focus (Position below-fold)
    and id re-resolution across the reveal (position-0 → position-1, 缝 8). The main
    brain holds only ``PanelAnchor`` descriptors + a ``Locator`` throughout; the resolved
    element is reached without ever exposing a raw panel_id or a coordinate.
    """
    adapter = _NestedRevealAdapter()
    controller = _chain_controller(adapter)
    perception = PerceptionRegion(event_sink=lambda *a, **k: None)

    # focus chain: descriptors only — no raw panel_id anywhere in the input
    result = controller.focus_chain(
        [
            PanelAnchor(panel_name="inspector"),
            PanelAnchor(panel_name="transform"),
            PanelAnchor(panel_name="position"),
        ]
    )
    assert result.outcome == "focused"
    assert result.semantic_path == ("Inspector", "Transform", "Position")
    # reveal-before-focus fired at the Position level + id was re-resolved (not cached)
    assert result.reveals_used == 1
    assert "position-0" in adapter.focus_calls  # first attempt (blocked, below_fold)
    assert "position-1" in adapter.focus_calls  # re-resolved id after the reveal

    # within the focused Position obs, resolve the X field via a Locator (descriptors only)
    locator = Locator(
        anchor=PanelAnchor(panel_name="position"),
        descriptor=ElementDescriptor(role="number_field", label="X"),
    )
    resolved = perception.resolve(locator, result.observation)
    assert resolved.status == "resolved"
    assert resolved.first.element_id == "position-1-x"


def test_capstone_focus_chain_on_unity_mock_to_position_x():
    """Unity mock (full-fidelity, stable ids, no reveal): focus_chain reaches Position X.

    Exercises the REAL ``UnityEditorMockAdapter`` nesting from 缝 6 (inspector → transform
    → position) through ``focus_chain`` and a Locator resolve — happy path, no reveal.
    """
    _adapter, session, perception = _setup()  # cube created → Transform component
    controller = TargetingController(session=session, perception=perception)
    result = controller.focus_chain(
        [
            PanelAnchor(panel_name="inspector"),
            PanelAnchor(panel_name="transform"),
            PanelAnchor(panel_name="position"),
        ]
    )
    assert result.outcome == "focused"
    assert result.semantic_path == ("Inspector", "Transform", "Position")
    assert result.reveals_used == 0  # Unity mock is full-fidelity — no below-fold at panel level

    locator = Locator(
        anchor=PanelAnchor(panel_name="position"),
        descriptor=ElementDescriptor(role="number_field", label="X"),
    )
    resolved = perception.resolve(locator, result.observation)
    assert resolved.status == "resolved"
    assert resolved.first.element_id == "inspector-component-transform-position-x"
