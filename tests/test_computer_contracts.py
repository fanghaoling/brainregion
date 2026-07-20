"""Contract-level tests for Panel tree (parent_panel_id) + focus metadata.

Stage 1 of §206 D (nested-window model): additive data layer, zero behavior change.
Covers: parent_panel_id dangling/cycle validation, tree helpers (children_of /
ancestors_of / descendants_of), focus_root_panel_id / focus_ancestor_path metadata,
Panel serialization roundtrip + backward compat. No LLM, no adapter.
"""

from __future__ import annotations

import pytest

from brainregion.computer.contracts import FrameRef, Panel, SceneObservation, UIElement


_DIGEST = "a" * 64


def _frame() -> FrameRef:
    return FrameRef(frame_id="f1", sha256=_DIGEST, width=10, height=10, artifact_uri="mock://x")


def _scene(panels=(), elements=(), *, focus_root_panel_id=None, focus_ancestor_path=()) -> SceneObservation:
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
        focus_root_panel_id=focus_root_panel_id,
        focus_ancestor_path=focus_ancestor_path,
    )


def _panel(pid, *, parent=None, role="region", label=None):
    return Panel(panel_id=pid, role=role, label=label or pid, parent_panel_id=parent)


# --- parent_panel_id: additive default + backward compat ---


def test_parent_panel_id_defaults_none():
    panel = Panel(panel_id="h", role="hierarchy", label="Hierarchy")
    assert panel.parent_panel_id is None


def test_panel_from_dict_old_payload_has_no_parent():
    """A payload serialized before parent_panel_id existed still loads (backward compat)."""
    panel = Panel.from_dict({"panel_id": "h", "role": "hierarchy", "label": "Hierarchy"})
    assert panel.parent_panel_id is None


def test_panel_roundtrip_with_parent_panel_id():
    panel = Panel(panel_id="transform", role="section", label="Transform", parent_panel_id="inspector")
    rt = Panel.from_dict(panel.to_dict())
    assert rt == panel
    assert rt.parent_panel_id == "inspector"
    # to_dict carries the field
    assert panel.to_dict()["parent_panel_id"] == "inspector"


# --- parent_panel_id dangling ---


def test_dangling_parent_panel_id_rejected():
    panels = (_panel("transform", parent="inspector"),)  # inspector not in scene
    with pytest.raises(ValueError, match="parent_panel_id"):
        _scene(panels=panels)


def test_valid_parent_chain_constructs():
    panels = (
        _panel("root"),
        _panel("inspector", parent="root"),
        _panel("transform", parent="inspector"),
        _panel("position", parent="transform"),
    )
    obs = _scene(panels=panels)
    assert {p.panel_id for p in obs.panels} == {"root", "inspector", "transform", "position"}


# --- cycle detection (self-ref / 2-node / 3-node) ---


def test_self_referential_parent_rejected():
    panels = (_panel("a", parent="a"),)
    with pytest.raises(ValueError, match="cycle"):
        _scene(panels=panels)


def test_two_node_cycle_rejected():
    panels = (_panel("a", parent="b"), _panel("b", parent="a"))
    with pytest.raises(ValueError, match="cycle"):
        _scene(panels=panels)


def test_three_node_cycle_rejected():
    panels = (
        _panel("a", parent="b"),
        _panel("b", parent="c"),
        _panel("c", parent="a"),
    )
    with pytest.raises(ValueError, match="cycle"):
        _scene(panels=panels)


# --- tree helpers ---


def _depth3():
    return (
        _panel("root"),
        _panel("inspector", parent="root"),
        _panel("transform", parent="inspector"),
        _panel("position", parent="transform"),
    )


def test_children_of_preserves_order():
    panels = (
        _panel("root"),
        _panel("inspector", parent="root"),
        _panel("hierarchy", parent="root"),
    )
    obs = _scene(panels=panels)
    assert [p.panel_id for p in obs.children_of("root")] == ["inspector", "hierarchy"]
    assert obs.children_of("transform") == ()


def test_ancestors_of_walks_to_root_immediate_first():
    obs = _scene(panels=_depth3())
    assert [p.panel_id for p in obs.ancestors_of("position")] == ["transform", "inspector", "root"]
    assert obs.ancestors_of("root") == ()


def test_descendants_of_preorder_dfs():
    obs = _scene(panels=_depth3())
    assert [p.panel_id for p in obs.descendants_of("root")] == [
        "inspector",
        "transform",
        "position",
    ]
    assert obs.descendants_of("position") == ()


def test_helper_max_depth_is_defensive_cap():
    """max_depth caps traversal even if a pathological cycle slipped past validation."""
    # Build a scene whose panels reference a cycle via direct construction is impossible
    # (validation rejects it), so instead verify max_depth bounds a deep legitimate chain.
    deep = [_panel("n0")] + [_panel(f"n{i}", parent=f"n{i - 1}") for i in range(1, 20)]
    obs = _scene(panels=tuple(deep))
    assert len(obs.ancestors_of("n19", max_depth=5)) == 5
    assert len(obs.descendants_of("n0", max_depth=4)) == 4


# --- focus metadata ---


def test_focus_root_panel_id_must_reference_scene_panel():
    panels = (_panel("inspector"),)
    # valid: focus root is in scene
    obs = _scene(panels=panels, focus_root_panel_id="inspector")
    assert obs.focus_root_panel_id == "inspector"
    # invalid: focus root not in scene
    with pytest.raises(ValueError, match="focus_root_panel_id"):
        _scene(panels=panels, focus_root_panel_id="ghost")


def test_focus_ancestor_path_rejects_non_tuple_and_normalizes():
    panels = (_panel("inspector"),)
    with pytest.raises(ValueError, match="focus_ancestor_path"):
        _scene(panels=panels, focus_ancestor_path=["root", "inspector"])  # list, not tuple
    obs = _scene(
        panels=panels,
        focus_root_panel_id="inspector",
        focus_ancestor_path=(" root ", "inspector"),
    )
    assert obs.focus_ancestor_path == ("root", "inspector")


def test_focused_observation_root_parent_normalized_none():
    """缝 3: a focused obs is self-contained — focus root's parent normalized to None,
    descendants keep their in-scene parents, ancestors ride in focus_ancestor_path."""
    panels = (
        _panel("transform", parent=None),  # focus root, normalized
        _panel("position", parent="transform"),
    )
    obs = _scene(
        panels=panels,
        focus_root_panel_id="transform",
        focus_ancestor_path=("root", "inspector"),
    )
    assert obs.focus_root_panel_id == "transform"
    assert obs.focus_ancestor_path == ("root", "inspector")
    # descendants_of works within the focused subtree
    assert [p.panel_id for p in obs.descendants_of("transform")] == ["position"]


def test_focused_observation_rejects_unnormalized_root_parent():
    """If the constructor fails to normalize (root parent points outside the focused obs),
    the dangling check rejects it — self-containment is enforced, not optional."""
    panels = (
        _panel("transform", parent="inspector"),  # inspector not in this focused obs
        _panel("position", parent="transform"),
    )
    with pytest.raises(ValueError, match="parent_panel_id"):
        _scene(panels=panels, focus_root_panel_id="transform")


# --- serialization exposes focus fields ---


def test_scene_to_dict_carries_focus_fields():
    obs = _scene(
        panels=(_panel("inspector"),),
        focus_root_panel_id="inspector",
        focus_ancestor_path=("root",),
    )
    d = obs.to_dict()
    assert d["focus_root_panel_id"] == "inspector"
    assert d["focus_ancestor_path"] == ["root"]


def test_scene_to_public_dict_exposes_focus_shape_not_ancestor_labels():
    obs = _scene(
        panels=(_panel("inspector"),),
        focus_root_panel_id="inspector",
        focus_ancestor_path=("root", "inspector"),
    )
    public = obs.to_public_dict()
    assert public["focus_root_panel_id"] == "inspector"
    assert public["focus_ancestor_depth"] == 2
    # ancestor labels are content — must not leak in the public redacted view
    assert "focus_ancestor_path" not in public


def test_default_scene_has_no_focus_metadata():
    """Backward compat: a plain survey observation (no focus) has null focus fields."""
    obs = _scene(panels=(_panel("hierarchy"),))
    assert obs.focus_root_panel_id is None
    assert obs.focus_ancestor_path == ()
    assert obs.to_dict()["focus_root_panel_id"] is None


# --- focused_view: local observation (缝 3) ---


def test_focused_view_selects_subtree_and_normalizes_root():
    panels = (
        _panel("root"),
        _panel("inspector", parent="root"),
        _panel("transform", parent="inspector"),
        _panel("position", parent="transform"),
        _panel("hierarchy", parent="root"),  # sibling subtree, must be excluded
    )
    elements = (
        UIElement(element_id="insp-hdr", role="header", label="H", panel_id="inspector"),
        UIElement(element_id="pos-x", role="input", label="X", panel_id="position"),
        UIElement(element_id="hier-bg", role="canvas", label="bg", panel_id="hierarchy"),
    )
    obs = _scene(panels=panels, elements=elements)
    focused = obs.focused_view("inspector")

    # subtree only: inspector + descendants (transform, position); root + hierarchy excluded
    assert {p.panel_id for p in focused.panels} == {"inspector", "transform", "position"}
    # focus root parent normalized to None (self-contained focused obs)
    assert focused.panel("inspector").parent_panel_id is None
    # descendants keep their in-scene parents
    assert focused.panel("transform").parent_panel_id == "inspector"
    assert focused.panel("position").parent_panel_id == "transform"
    # focus metadata
    assert focused.focus_root_panel_id == "inspector"
    assert focused.focus_ancestor_path == ("root",)  # ancestor labels, not ids
    # elements filtered to the subtree (hierarchy's element excluded)
    assert {e.element_id for e in focused.elements} == {"insp-hdr", "pos-x"}
    # frame preserved (same underlying frame); caller stamps a fresh sequence
    assert focused.frame is obs.frame


def test_focused_view_on_deep_panel_records_multi_ancestor_path():
    panels = (
        _panel("root"),
        _panel("inspector", parent="root"),
        _panel("transform", parent="inspector"),
    )
    obs = _scene(panels=panels)
    focused = obs.focused_view("transform")
    assert focused.focus_root_panel_id == "transform"
    assert focused.panel("transform").parent_panel_id is None
    # ancestors walked to root: immediate parent first
    assert [p.panel_id for p in obs.ancestors_of("transform")] == ["inspector", "root"]
    assert focused.focus_ancestor_path == ("inspector", "root")


def test_focused_view_rejects_unknown_panel():
    obs = _scene(panels=(_panel("inspector"),))
    with pytest.raises(ValueError, match="focused_view"):
        obs.focused_view("ghost")
