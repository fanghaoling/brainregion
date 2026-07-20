"""UnityEditorMockAdapter interaction-contract + flow tests."""

from __future__ import annotations

import pytest

from brainregion.computer.adapter import FocusableComputerUseAdapter, FocusNotSupported
from brainregion.computer.contracts import ActionIntent
from brainregion.computer.mock import MockComputerUseAdapter
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
    # focus metadata
    assert focused.focus_root_panel_id == "inspector"
    assert focused.focus_ancestor_path == ()  # inspector is top-level in the flat mock
    # focus root parent normalized to None (self-contained focused obs)
    assert focused.panel("inspector").parent_panel_id is None
    # scope narrowing: only inspector's elements survive (no hierarchy/scene/toolbar)
    assert all(e.panel_id == "inspector" for e in focused.elements)
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
