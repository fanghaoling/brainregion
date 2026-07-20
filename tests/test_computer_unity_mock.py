"""UnityEditorMockAdapter interaction-contract + flow tests."""

from __future__ import annotations

import pytest

from brainregion.computer.adapter import FocusableComputerUseAdapter, FocusNotSupported
from brainregion.computer.contracts import ActionIntent
from brainregion.computer.locator import ElementDescriptor, Locator, PanelAnchor
from brainregion.computer.mock import MockComputerUseAdapter
from brainregion.computer.perception import PerceptionRegion
from brainregion.computer.session import ComputerUseSession
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


def _adapter_with_cube() -> UnityEditorMockAdapter:
    adapter = UnityEditorMockAdapter()
    adapter.execute(_intent("hierarchy-background", button="right"))
    adapter.execute(_intent("menu-3d-object", action="hover"))
    adapter.execute(_intent("submenu-cube"))
    return adapter


def test_context_menu_requires_right_click():
    adapter = UnityEditorMockAdapter()
    assert not adapter.execute(_intent("hierarchy-background", button="left")).succeeded
    assert adapter.execute(_intent("hierarchy-background", button="right")).succeeded
    assert adapter._state["context_menu_open"] is True


def test_submenu_requires_hover_not_click():
    adapter = UnityEditorMockAdapter()
    adapter.execute(_intent("hierarchy-background", button="right"))
    assert not adapter.execute(_intent("menu-3d-object", action="click")).succeeded
    assert adapter.execute(_intent("menu-3d-object", action="hover")).succeeded
    assert adapter._state["submenu_open"] is True


def test_below_fold_click_rejected_by_adapter():
    adapter = _adapter_with_cube()  # scroll=top -> add-component below fold
    result = adapter.execute(_intent("inspector-add-component"))
    assert not result.succeeded
    assert "below_fold" in result.reason


def test_invisible_result_click_rejected():
    adapter = _adapter_with_cube()
    adapter.execute(_intent(None, action="press_key", key="end"))
    adapter.execute(_intent("inspector-add-component"))
    # search empty -> result invisible
    result = adapter.execute(_intent("acp-result-rigidbody"))
    assert not result.succeeded
    assert "not_interactable" in result.reason or "not_visible" in result.reason


def test_create_cube_flow():
    adapter = _adapter_with_cube()
    assert len(adapter._state["scene_objects"]) == 1
    assert adapter._state["selected_object_id"] == "cube-1"
    observation = adapter.observe(session_id="s")
    assert any(e.element_id == "hierarchy-item-cube-1" for e in observation.elements)
    assert any(e.element_id == "inspector-add-component" for e in observation.elements)


def test_add_rigidbody_flow():
    adapter = _adapter_with_cube()
    adapter.execute(_intent(None, action="press_key", key="end"))  # reveal Add Component
    adapter.execute(_intent("inspector-add-component"))
    adapter.execute(_intent("acp-search", action="type_text", payload="rigidbody"))
    adapter.execute(_intent("acp-result-rigidbody"))
    assert "Rigidbody" in adapter._state["scene_objects"][0]["components"]


def test_mutate_out_of_band_changes_state_hash():
    adapter = _adapter_with_cube()
    before = adapter.observe(session_id="s")
    adapter.mutate_out_of_band(play_mode=True)
    after = adapter.observe(session_id="s")
    assert before.state_sha256 != after.state_sha256


def test_repeated_observe_same_state_same_frame_different_sequence():
    # frame_id is content-derived (digest): same logical state -> same frame_id;
    # sequence still advances. This is what lets session freshness use frame_id.
    adapter = UnityEditorMockAdapter()
    first = adapter.observe(session_id="s")
    second = adapter.observe(session_id="s")
    assert first.state_sha256 == second.state_sha256
    assert first.frame.frame_id == second.frame.frame_id
    assert first.sequence != second.sequence


def test_mutate_out_of_band_rejects_unknown_key():
    adapter = UnityEditorMockAdapter()
    with pytest.raises(ValueError):
        adapter.mutate_out_of_band(bogus_key=True)


def test_observe_focus_returns_local_observation():
    adapter = _adapter_with_cube()
    full = adapter.observe(session_id="s")
    focused = adapter.observe_focus(session_id="s", panel_id="inspector")
    # focus metadata: inspector is a top-level panel → no ancestors
    assert focused.focus_root_panel_id == "inspector"
    assert focused.focus_ancestor_path == ()
    # focus root parent normalized to None (self-contained focused obs)
    assert focused.panel("inspector").parent_panel_id is None
    # scope narrowing: only the inspector SUBTREE survives (缝 6 nesting). The full
    # scene's hierarchy/scene/game/toolbar panels + their elements are dropped, while
    # the inspector's descendant panels (Transform, Position) and their elements survive.
    subtree = {
        "inspector",
        "inspector-component-transform",
        "inspector-component-transform-position",
    }
    assert {p.panel_id for p in focused.panels} == subtree
    assert {"hierarchy", "scene", "game", "toolbar"}.isdisjoint({p.panel_id for p in focused.panels})
    assert all(e.panel_id in subtree for e in focused.elements)
    assert all(e.element_id != "hierarchy-background" for e in focused.elements)
    assert all(e.element_id != "toolbar-play" for e in focused.elements)
    # depth-3 descendant element (Position X) is reachable through the focused view
    assert any(e.element_id == "inspector-component-transform-position-x" for e in focused.elements)
    # distinct sequence; same frame_id (same underlying state)
    assert focused.sequence != full.sequence
    assert focused.frame.frame_id == full.frame.frame_id


def test_observe_focus_rejects_unknown_panel():
    adapter = UnityEditorMockAdapter()
    with pytest.raises(ValueError):
        adapter.observe_focus(session_id="s", panel_id="ghost")


def test_mock_is_focusable_v1_mock_opts_out():
    # the Unity mock implements observe_focus -> satisfies the capability Protocol
    assert isinstance(UnityEditorMockAdapter(), FocusableComputerUseAdapter)
    # the v1 MockComputerUseAdapter does not -> opts out (isinstance False, no NotImplementedError)
    assert not isinstance(MockComputerUseAdapter(), FocusableComputerUseAdapter)
    err = FocusNotSupported("x.app")
    assert err.app_id == "x.app"
    assert "focus" in str(err)


def test_session_focus_by_descriptor_crops_panel():
    adapter = _adapter_with_cube()
    session = ComputerUseSession(session_id="s", adapter=adapter, allowed_apps={"unity.editor"})
    session.observe()  # establish the latest observation
    focused = session.focus(PanelAnchor(panel_name="inspector"))
    # descriptor resolved to the internal handle, adapter cropped to the inspector subtree
    assert focused.focus_root_panel_id == "inspector"
    subtree = {
        "inspector",
        "inspector-component-transform",
        "inspector-component-transform-position",
    }
    assert all(e.panel_id in subtree for e in focused.elements)  # scope narrowed to subtree
    assert session._latest is focused  # session adopted the focused obs


def test_session_focus_non_focusable_adapter_raises_focus_not_supported():
    adapter = MockComputerUseAdapter()
    session = ComputerUseSession(session_id="s", adapter=adapter, allowed_apps={adapter.app_id})
    with pytest.raises(FocusNotSupported):
        session.focus(PanelAnchor(panel_name="x"))


def test_session_focus_unresolved_anchor_raises():
    adapter = UnityEditorMockAdapter()
    session = ComputerUseSession(session_id="s", adapter=adapter, allowed_apps={"unity.editor"})
    session.observe()
    with pytest.raises(ValueError, match="not_found"):
        session.focus(PanelAnchor(panel_name="ghost-panel"))


# --- 缝 6: nested window model + focus-chain (主风险卸载点) ---


def test_inspector_nests_components_to_depth_3():
    adapter = _adapter_with_cube()
    obs = adapter.observe(session_id="s")
    # depth 2: the Transform component is a child panel of the inspector
    transform = obs.panel("inspector-component-transform")
    assert transform is not None
    assert transform.parent_panel_id == "inspector"
    assert "inspector-component-transform" in {p.panel_id for p in obs.children_of("inspector")}
    # depth 3: Position is a child of Transform
    position = obs.panel("inspector-component-transform-position")
    assert position is not None
    assert position.parent_panel_id == "inspector-component-transform"
    # descendants_of(inspector) reaches Position (depth 3) in deterministic pre-order
    assert [p.panel_id for p in obs.descendants_of("inspector")] == [
        "inspector-component-transform",
        "inspector-component-transform-position",
    ]
    # ancestors_of(Position) walks the parent chain; labels are descriptive (缝 8, not ids)
    assert tuple(p.label for p in obs.ancestors_of("inspector-component-transform-position")) == (
        "Transform",
        "Inspector",
    )


def test_focus_chain_narrows_scope_to_each_subtree():
    adapter = _adapter_with_cube()
    # survey: the full scene carries hierarchy/toolbar alongside the inspector tree
    full = adapter.observe(session_id="s")
    assert {"hierarchy", "toolbar", "inspector"} <= {p.panel_id for p in full.panels}

    # focus inspector (depth 1) → only the inspector subtree survives
    inspector_focused = adapter.observe_focus(session_id="s", panel_id="inspector")
    assert inspector_focused.focus_root_panel_id == "inspector"
    assert inspector_focused.focus_ancestor_path == ()
    assert {p.panel_id for p in inspector_focused.panels} == {
        "inspector",
        "inspector-component-transform",
        "inspector-component-transform-position",
    }
    # narrowing is real (not透传): sibling top-level panels + their elements are gone
    assert {"hierarchy", "toolbar"}.isdisjoint(p.panel_id for p in inspector_focused.panels)
    assert all(e.element_id != "hierarchy-background" for e in inspector_focused.elements)
    assert all(e.element_id != "toolbar-play" for e in inspector_focused.elements)

    # focus Transform (depth 2) → only Transform + Position survive
    transform_focused = adapter.observe_focus(session_id="s", panel_id="inspector-component-transform")
    assert transform_focused.focus_root_panel_id == "inspector-component-transform"
    # ancestor context is descriptive labels (缝 3/8)
    assert transform_focused.focus_ancestor_path == ("Inspector",)
    assert {p.panel_id for p in transform_focused.panels} == {
        "inspector-component-transform",
        "inspector-component-transform-position",
    }
    # focus root parent normalized None; Position's parent link stays intact within subtree
    assert transform_focused.panel("inspector-component-transform").parent_panel_id is None
    assert (
        transform_focused.panel("inspector-component-transform-position").parent_panel_id
        == "inspector-component-transform"
    )
    # add-component lives directly under inspector → gone when focusing Transform (narrowing)
    assert all(e.element_id != "inspector-add-component" for e in transform_focused.elements)

    # focus Position (depth 3) → only Position + its X/Y/Z fields
    position_focused = adapter.observe_focus(session_id="s", panel_id="inspector-component-transform-position")
    assert position_focused.focus_root_panel_id == "inspector-component-transform-position"
    assert position_focused.focus_ancestor_path == ("Transform", "Inspector")
    assert {p.panel_id for p in position_focused.panels} == {"inspector-component-transform-position"}
    assert {e.element_id for e in position_focused.elements} == {
        "inspector-component-transform-position-x",
        "inspector-component-transform-position-y",
        "inspector-component-transform-position-z",
    }


def test_resolve_within_focused_position_via_descriptor():
    # gate: survey → focus(by descriptor) → resolve within focused obs (no LLM).
    # The main brain holds descriptors only (缝 2/8); session.focus turns the descriptor
    # into the internal handle, and the focused obs is itself resolve-able.
    adapter = _adapter_with_cube()
    session = ComputerUseSession(session_id="s", adapter=adapter, allowed_apps={"unity.editor"})
    focused = session.focus(PanelAnchor(panel_name="position"))
    assert focused.focus_root_panel_id == "inspector-component-transform-position"
    locator = Locator(
        anchor=PanelAnchor(panel_name="position"),
        descriptor=ElementDescriptor(role="number_field"),
    )
    result = PerceptionRegion(event_sink=lambda *a, **k: None).resolve(locator, focused)
    assert result.status == "ambiguous"
    assert {c.element_id for c in result.candidates} == {
        "inspector-component-transform-position-x",
        "inspector-component-transform-position-y",
        "inspector-component-transform-position-z",
    }
