"""TargetingController reveal/retry + freshness-binding integration tests."""

from __future__ import annotations

from brainregion.computer.contracts import ActionIntent, ActionReceipt
from brainregion.computer.locator import (
    ElementDescriptor,
    Locator,
    PanelAnchor,
    WithinPanel,
)
from brainregion.computer.perception import PerceptionRegion
from brainregion.computer.session import ComputerUseSession
from brainregion.computer.targeting import TargetingController
from brainregion.computer.unity_mock import UnityEditorMockAdapter


_DIGEST = "a" * 64


def _intent(target_id, *, action="click", button="left", payload="", key="") -> ActionIntent:
    return ActionIntent(
        intent_id="i", session_id="s", app_id="unity.editor", action=action,
        expected_frame_id="f", expected_state_sha256=_DIGEST,
        target_id=target_id, button=button, payload=payload, key=key,
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
            intent_id="x", session_id="s", app_id="unity.editor", status="failed",
            reason="reveal_unsupported", before_frame=obs.frame, after_frame=obs.frame,
            state_changed=False,
        )

    controller = TargetingController(
        session=session, perception=perception, max_reveals=3, reveal_strategy=failing_reveal,
    )
    result = controller.target(_add_component_locator())
    assert result.status == "blocked"
    assert "reveal_failed" in result.reason


def test_resolved_target_to_action_intent_passes_session_freshness():
    _adapter, session, perception = _setup()
    controller = TargetingController(session=session, perception=perception, max_reveals=3)
    result = controller.target(_add_component_locator())
    assert result.status == "resolved"
    intent = result.first.to_action_intent(
        action="click", intent_id="act1", session_id="s", app_id="unity.editor"
    )
    receipt = session.perform(intent)
    assert receipt.status == "executed"


def test_stale_after_out_of_band_change_between_resolve_and_perform():
    adapter, session, perception = _setup()
    controller = TargetingController(session=session, perception=perception, max_reveals=3)
    result = controller.target(_add_component_locator())
    intent = result.first.to_action_intent(
        action="click", intent_id="act1", session_id="s", app_id="unity.editor"
    )
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
    session = ComputerUseSession(
        session_id="s", adapter=adapter, allowed_apps={"unity.editor"}
    )
    perception = PerceptionRegion(event_sink=lambda *a, **k: None)
    controller = TargetingController(session=session, perception=perception)

    def act(locator, *, action, intent_id, button="left", payload=""):
        result = controller.target(locator)
        assert result.status == "resolved", f"{intent_id}: {result.status} {result.reason}"
        intent = result.first.to_action_intent(
            action=action, intent_id=intent_id, session_id="s",
            app_id="unity.editor", button=button, payload=payload,
        )
        receipt = session.perform(intent)
        assert receipt.status == "executed", f"{intent_id}: {receipt.status} {receipt.reason}"

    # 1. right-click Hierarchy blank -> context menu
    act(
        Locator(
            anchor=PanelAnchor(panel_name="hierarchy"),
            descriptor=ElementDescriptor(role="canvas", label="hierarchy blank"),
        ),
        action="click", intent_id="open-menu", button="right",
    )
    # 2. hover 3D Object (the just-opened context menu)
    act(
        Locator(
            anchor=PanelAnchor(transient_kind="context_menu", ordinal="just_opened"),
            descriptor=ElementDescriptor(role="menu_item", label="3d object"),
        ),
        action="hover", intent_id="hover-3dobject",
    )
    # 3. click Cube (submenu spawned by the 3D Object item)
    act(
        Locator(
            anchor=PanelAnchor(
                transient_kind="submenu", spawned_by=ElementDescriptor(label="3d object"),
            ),
            descriptor=ElementDescriptor(role="menu_item", label="cube"),
        ),
        action="click", intent_id="create-cube",
    )
    assert adapter._state["selected_object_id"] == "cube-1"
    # 4. add Rigidbody (Add Component is at Inspector bottom, below fold -> controller reveals)
    act(
        Locator(
            anchor=PanelAnchor(panel_name="inspector", ordinal="rightmost"),
            within=WithinPanel(band="bottom"),
            descriptor=ElementDescriptor(role="button", label="add comp"),
        ),
        action="click", intent_id="open-acp",
    )
    act(
        Locator(
            anchor=PanelAnchor(transient_kind="popup", ordinal="just_opened"),
            descriptor=ElementDescriptor(
                role="text_input", attributes=(("icon_shape", "question_mark"),),
            ),
        ),
        action="type_text", intent_id="type-rigidbody", payload="rigidbody",
    )
    act(
        Locator(
            anchor=PanelAnchor(transient_kind="popup"),
            within=WithinPanel(relation="below", relative_to=ElementDescriptor(role="text_input")),
            descriptor=ElementDescriptor(role="list_item", label="rigidbody"),
        ),
        action="click", intent_id="pick-rigidbody",
    )
    assert "Rigidbody" in adapter._state["scene_objects"][0]["components"]
