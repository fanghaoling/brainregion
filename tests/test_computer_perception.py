"""Perception layer + panel evolution tests (algorithmic, no LLM, no adapter)."""

from __future__ import annotations

import pytest

from brainregion.computer.contracts import FrameRef, Panel, SceneObservation, UIElement
from brainregion.computer.locator import (
    ElementDescriptor,
    Locator,
    PanelAnchor,
    WithinPanel,
)
from brainregion.computer.mock import MockComputerUseAdapter
from brainregion.computer.perception import PerceptionRegion


_DIGEST = "a" * 64


def _frame() -> FrameRef:
    return FrameRef(frame_id="f1", sha256=_DIGEST, width=10, height=10, artifact_uri="mock://x")


def _scene(elements=(), panels=()) -> SceneObservation:
    return SceneObservation(
        session_id="s",
        sequence=1,
        app_id="a",
        window_id="w",
        window_title="T",
        frame=_frame(),
        state_sha256=_DIGEST,
        elements=elements,
        panels=panels,
    )


def _recorder():
    events: list[dict] = []

    def sink(event_type: str, **fields):
        events.append({"type": event_type, **fields})

    return events, sink


def _quiet() -> PerceptionRegion:
    return PerceptionRegion(event_sink=lambda *a, **k: None)


# --- panel evolution ---


def test_panel_transient_derived_from_transient_kind():
    assert Panel(panel_id="cm", role="menu", label="CM", transient_kind="context_menu").transient is True
    assert Panel(panel_id="insp", role="inspector", label="Inspector").transient is False


def test_settings_mock_backward_compat_flat():
    adapter = MockComputerUseAdapter()
    observation = adapter.observe(session_id="s")
    assert observation.panels == ()
    assert all(element.panel_id is None for element in observation.elements)


def test_scene_rejects_dangling_element_panel_id():
    element = UIElement(element_id="e", role="btn", label="X", panel_id="ghost")
    with pytest.raises(ValueError):
        _scene(elements=(element,))


def test_scene_rejects_dangling_spawned_by():
    panel = Panel(
        panel_id="cm",
        role="menu",
        label="CM",
        transient_kind="context_menu",
        spawned_by_element_id="missing",
    )
    with pytest.raises(ValueError):
        _scene(panels=(panel,))


def test_scene_rejects_duplicate_panel_id():
    panels = (
        Panel(panel_id="p", role="r", label="L"),
        Panel(panel_id="p", role="r", label="L2"),
    )
    with pytest.raises(ValueError):
        _scene(panels=panels)


# --- resolver paths ---


def _inspector_scene() -> SceneObservation:
    inspector = Panel(panel_id="inspector", role="inspector", label="Inspector", ordinal="rightmost")
    elements = (
        UIElement(element_id="t", role="component", label="Transform", panel_id="inspector"),
        UIElement(element_id="m", role="component", label="Mesh", panel_id="inspector"),
        UIElement(element_id="c", role="component", label="Collider", panel_id="inspector"),
        UIElement(
            element_id="ac",
            role="button",
            label="Add Component",
            panel_id="inspector",
            semantic_band="bottom",
            attributes=(("below_fold", True),),
        ),
    )
    return _scene(elements=elements, panels=(inspector,))


def test_resolve_band_bottom_is_last_element():
    locator = Locator(anchor=PanelAnchor(panel_name="inspector"), within=WithinPanel(band="bottom"))
    result = _quiet().resolve(locator, _inspector_scene())
    assert result.first is not None
    assert result.first.element_id == "ac"


def test_resolve_blocked_below_fold():
    locator = Locator(
        anchor=PanelAnchor(panel_name="inspector"),
        within=WithinPanel(band="bottom"),
        descriptor=ElementDescriptor(role="button", label="add comp"),
    )
    result = _quiet().resolve(locator, _inspector_scene())
    assert result.status == "blocked"
    assert result.first.blocker == "below_fold"


def test_resolve_not_found_enumerates_candidates():
    locator = Locator(
        anchor=PanelAnchor(panel_name="inspector"),
        descriptor=ElementDescriptor(role="button", label="nonexistent"),
    )
    result = _quiet().resolve(locator, _inspector_scene())
    assert result.status == "not_found"
    assert len(result.not_found_candidates) == 4


def test_resolve_ambiguous_returns_all():
    locator = Locator(
        anchor=PanelAnchor(panel_name="inspector"),
        descriptor=ElementDescriptor(role="component"),
    )
    result = _quiet().resolve(locator, _inspector_scene())
    assert result.status == "ambiguous"
    assert len(result.candidates) == 3


def test_resolve_anchor_panel_name_and_ordinal_and_semantic_descriptor():
    locator = Locator(
        anchor=PanelAnchor(panel_name="inspector", ordinal="rightmost"),
        descriptor=ElementDescriptor(role="button"),
    )
    result = _quiet().resolve(locator, _inspector_scene())
    assert result.first.element_id == "ac"


def test_resolve_beside_is_unsupported_relation():
    locator = Locator(
        anchor=PanelAnchor(panel_name="inspector"),
        within=WithinPanel(relation="beside", relative_to=ElementDescriptor(role="component")),
    )
    result = _quiet().resolve(locator, _inspector_scene())
    assert result.status == "unsupported"
    assert result.reason == "unsupported_relation"


def test_resolve_descriptor_matches_icon_shape_attribute():
    toolbar = Panel(panel_id="toolbar", role="toolbar", label="Toolbar", ordinal="top")
    play = UIElement(
        element_id="play",
        role="button",
        label="Play",
        panel_id="toolbar",
        attributes=(("icon_shape", "play"),),
    )
    observation = _scene(elements=(play,), panels=(toolbar,))
    locator = Locator(
        anchor=PanelAnchor(panel_name="toolbar"),
        descriptor=ElementDescriptor(role="button", attributes=(("icon_shape", "play"),)),
    )
    result = _quiet().resolve(locator, observation)
    assert result.status == "resolved"
    assert result.first.element_id == "play"


def test_resolve_transient_just_opened_via_spawn_sequence():
    hierarchy = Panel(panel_id="h", role="hierarchy", label="Hierarchy", ordinal="leftmost")
    background = UIElement(element_id="hb", role="canvas", label="Hierarchy blank", panel_id="h")
    menu = Panel(
        panel_id="cm",
        role="menu",
        label="Context Menu",
        transient_kind="context_menu",
        spawned_by_element_id="hb",
        spawn_sequence=5,
    )
    item = UIElement(element_id="m3d", role="menu_item", label="3D Object", panel_id="cm")
    observation = _scene(elements=(background, item), panels=(hierarchy, menu))
    locator = Locator(anchor=PanelAnchor(transient_kind="context_menu", ordinal="just_opened"))
    result = _quiet().resolve(locator, observation)
    assert result.first is not None
    assert result.first.element_id == "m3d"


def test_resolve_just_opened_with_no_transients_is_not_found():
    # Guard: max() on empty transients must not raise (review point 2).
    hierarchy = Panel(panel_id="h", role="hierarchy", label="Hierarchy", ordinal="leftmost")
    background = UIElement(element_id="hb", role="canvas", label="blank", panel_id="h")
    observation = _scene(elements=(background,), panels=(hierarchy,))
    locator = Locator(anchor=PanelAnchor(ordinal="just_opened"))
    result = _quiet().resolve(locator, observation)
    assert result.status == "not_found"


def test_resolve_empty_panel_band_is_not_found_not_index_error():
    # Guard: band=bottom on an element-less panel must not IndexError (review point 1).
    empty = Panel(panel_id="empty", role="region", label="Empty", ordinal="rightmost")
    observation = _scene(elements=(), panels=(empty,))
    locator = Locator(anchor=PanelAnchor(panel_name="empty"), within=WithinPanel(band="bottom"))
    result = _quiet().resolve(locator, observation)
    assert result.status == "not_found"


def test_find_not_found_truncates_candidates():
    elements = tuple(UIElement(element_id=f"e{i}", role="x", label=f"L{i}", panel_id="p") for i in range(30))
    panel = Panel(panel_id="p", role="r", label="P")
    observation = _scene(elements=elements, panels=(panel,))
    result = _quiet().find(observation, ElementDescriptor(role="nomatch"), max_candidates=5)
    assert result["status"] == "not_found"
    assert len(result["result"]["candidates"]) == 5
    assert result["result"]["truncated"] is True


def test_perception_events_do_not_leak_element_content():
    events, sink = _recorder()
    perception = PerceptionRegion(event_sink=sink)
    perception.survey(_inspector_scene())
    perception.resolve(
        Locator(anchor=PanelAnchor(panel_name="inspector"), descriptor=ElementDescriptor(role="button")),
        _inspector_scene(),
    )
    blob = repr(events)
    assert "Add Component" not in blob  # labels are content, must not reach telemetry
    assert "below_fold" not in blob  # blocker values not leaked as raw
    for event in events:
        assert "trace" not in event  # resolution trace is user-visible only
        assert "label" not in event


# --- 缝 8: survey/focus feedback carries descriptors, not raw panel_id ---


def test_survey_does_not_leak_panel_id():
    obs = _scene(panels=(Panel(panel_id="inspector", role="inspector", label="Inspector", ordinal="rightmost"),))
    entries = _quiet().survey(obs)["result"]["panels"]
    assert entries
    for entry in entries:
        assert "panel_id" not in entry  # main brain sees descriptors, not the internal handle
        assert {"role", "label", "ordinal", "transient_kind", "element_count"} <= set(entry)


def test_focus_does_not_leak_panel_id():
    obs = _scene(
        panels=(Panel(panel_id="inspector", role="inspector", label="Inspector"),),
        elements=(UIElement(element_id="i", role="button", label="Add Component", panel_id="inspector"),),
    )
    panel = _quiet().focus(obs, "inspector")["result"]["panel"]
    assert "panel_id" not in panel
    assert panel["label"] == "Inspector"


# --- resolve_panel: anchor → panel_id (for session.focus) ---


def test_resolve_panel_single_match():
    obs = _scene(panels=(Panel(panel_id="inspector", role="inspector", label="Inspector", ordinal="rightmost"),))
    assert _quiet().resolve_panel(PanelAnchor(panel_name="inspector"), obs) == ("resolved", "inspector")


def test_resolve_panel_ambiguous_and_not_found():
    two_region = _scene(
        panels=(
            Panel(panel_id="a", role="region", label="A"),
            Panel(panel_id="b", role="region", label="B"),
        )
    )
    # both share role="region" → panel_name="region" matches both
    status, _pid = _quiet().resolve_panel(PanelAnchor(panel_name="region"), two_region)
    assert status == "ambiguous"
    status, pid = _quiet().resolve_panel(PanelAnchor(panel_name="ghost"), two_region)
    assert (status, pid) == ("not_found", None)
