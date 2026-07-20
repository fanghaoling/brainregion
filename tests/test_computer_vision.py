"""Tests for VisionAdapter: observe() scene build + observe_focus() bbox mapping.

The VLM call (``_parse`` -> ``requests.post``) is monkeypatched to a fixed parse; no
network. PIL is a pyautogui dependency — skip the module if absent.
"""

from __future__ import annotations

import io

import pytest

from brainregion.computer import VISION_PRESETS, VisionAdapter
from brainregion.computer.adapter import NoRegionForPanel
from brainregion.computer.vision_adapter import CursorAnchor

pytest.importorskip("PIL.Image")
from PIL import Image  # noqa: E402


def _fake_image_bytes(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def _adapter_with_parse(img_bytes: bytes, parsed: dict) -> VisionAdapter:
    adapter = VisionAdapter(
        vision=VISION_PRESETS["siliconflow-qwen3vl-8b"],
        screenshot=lambda: img_bytes,
        api_key="fake",
    )
    adapter._parse = lambda ib: parsed  # type: ignore[method-assign]
    return adapter


def _adapter_with_parse_sequence(img_bytes: bytes, parses: list[dict]) -> VisionAdapter:
    """``_parse`` returns parses[0], parses[1], ... on successive calls (clamped to last).

    Lets observe() and observe_focus() see different parses — eg the full screen misses a
    transient (empty) while the cursor-anchor crop reveals it.
    """
    adapter = VisionAdapter(
        vision=VISION_PRESETS["siliconflow-qwen3vl-8b"],
        screenshot=lambda: img_bytes,
        api_key="fake",
    )
    state = {"i": 0}

    def fake_parse(ib):
        i = min(state["i"], len(parses) - 1)
        state["i"] += 1
        return parses[i]

    adapter._parse = fake_parse  # type: ignore[method-assign]
    return adapter


# a crop parse that synthesizes a `context_menu` panel via the menu_item post-process
_FOCUS_MENU = {
    "panels": [{"name": "Hierarchy", "role": "hierarchy", "bbox": [0, 0, 500, 1000]}],
    "elements": [
        {
            "panel": "hierarchy",
            "role": "menu_item",
            "label": "3D Object",
            "bbox": [10, 100, 200, 130],
            "interactable": True,
        }
    ],
}


def test_observe_maps_screenshot_to_scene():
    parsed = {
        "panels": [{"name": "Hierarchy", "role": "hierarchy", "bbox": [0, 0, 200, 1000]}],
        "elements": [
            {
                "panel": "hierarchy",
                "role": "tree_item",
                "label": "Main Camera",
                "bbox": [10, 100, 190, 120],
                "interactable": True,
            }
        ],
    }
    adapter = _adapter_with_parse(_fake_image_bytes(1000, 600), parsed)
    obs = adapter.observe(session_id="t")
    assert any(p.role == "hierarchy" for p in obs.panels)
    elem = [e for e in obs.elements if "camera" in (e.label or "").lower()][0]
    assert elem.role == "tree_item"
    # bbox stored verbatim (full-screen normalized, no crop in plain observe)
    assert adapter._bbox_map[elem.element_id] == [10, 100, 190, 120]


def test_observe_focus_maps_crop_bbox_to_fullscreen():
    """A menu_item found at crop-normalized [350,540,1000,560] inside region (0,60,520,550)
    on a 1000x600 screen must map back to full-screen normalized [182,541,520,557], and the
    mapped center must land inside the crop region (so execute clicks the right pixel).
    """
    parsed = {
        "panels": [
            {
                "name": "context_menu",
                "role": "menu",
                "transient_kind": "context_menu",
                "bbox": [350, 540, 1000, 560],
            }
        ],
        "elements": [
            {
                "panel": "context_menu",
                "role": "menu_item",
                "label": "3D Object",
                "bbox": [350, 540, 1000, 560],
                "interactable": True,
            }
        ],
    }
    adapter = _adapter_with_parse(_fake_image_bytes(1000, 600), parsed)
    obs = adapter._observe_focus_region(session_id="t", region=(0, 60, 520, 550))
    # synthetic context_menu panel is present (menu_item -> context_menu post-process)
    assert any(p.panel_id == "context_menu" for p in obs.panels)
    elem = [e for e in obs.elements if "3d object" in (e.label or "").lower()][0]
    assert adapter._bbox_map[elem.element_id] == [182, 541, 520, 557]
    # mapped center lands inside the crop region (full-screen pixels)
    cx = (182 + 520) / 2 / 1000 * 1000
    cy = (541 + 557) / 2 / 1000 * 600
    assert 0 <= cx <= 520 and 60 <= cy <= 550


# --- 缝 8: dup-skip fix (same-role siblings survive with unique panel_ids) ---


def test_dup_skip_same_role_panels_get_unique_ids():
    """Without the fix, two role=component panels collided (pid=role) and the second was
    dropped — a nesting blocker. Now both survive with suffixed unique ids."""
    parsed = {
        "panels": [
            {"name": "Transform", "role": "component", "bbox": [0, 0, 100, 50]},
            {"name": "Mesh", "role": "component", "bbox": [0, 50, 100, 100]},
            {"name": "Collider", "role": "component", "bbox": [0, 100, 100, 150]},
        ],
        "elements": [],
    }
    adapter = _adapter_with_parse(_fake_image_bytes(200, 200), parsed)
    obs = adapter.observe(session_id="t")
    pids = [p.panel_id for p in obs.panels if p.role == "component"]
    assert pids == ["component", "component-2", "component-3"]


# --- 缝 3/4: observe_focus(panel_id) region resolution ---


def test_observe_focus_panel_id_crops_panel_region():
    """observe_focus(panel_id) resolves the panel bbox → crops that region; result is focused."""
    parsed = {
        "panels": [{"name": "Inspector", "role": "inspector", "bbox": [700, 0, 1000, 1000]}],
        "elements": [
            {
                "panel": "inspector",
                "role": "input",
                "label": "X",
                "bbox": [800, 100, 900, 120],
                "interactable": True,
            }
        ],
    }
    adapter = _adapter_with_parse(_fake_image_bytes(1000, 600), parsed)
    adapter.observe(session_id="t")  # populate _panel_bbox_map with the inspector bbox
    focused = adapter.observe_focus(session_id="t", panel_id="inspector")
    assert focused.focus_root_panel_id == "inspector"
    assert any(e.label == "X" for e in focused.elements)


def test_observe_focus_cursor_anchor_bootstrap_for_transient():
    """A transient panel with no bbox falls back to the cursor-anchor region (one-shot).

    Full screen (observe) misses the context_menu → empty parse → not in _panel_bbox_map.
    The cursor-anchor crop reveals it (menu_item → synthetic context_menu panel)."""
    adapter = _adapter_with_parse_sequence(
        _fake_image_bytes(1000, 600),
        [{"panels": [], "elements": []}, _FOCUS_MENU],
    )
    adapter.observe(session_id="t")  # call 0: empty — context_menu not in bbox_map
    adapter._cursor_anchor = CursorAnchor(
        session_id="t",
        click_xy=(300, 400),
        expected_transient_kind="context_menu",
        source_state_sha256=adapter._last_state_sha256,
    )
    focused = adapter.observe_focus(session_id="t", panel_id="context_menu")  # call 1: crop
    assert focused.focus_root_panel_id == "context_menu"
    # anchor consumed after one use (缝 4 rule ①)
    assert adapter._cursor_anchor is not None
    assert adapter._cursor_anchor.consumed is True


def test_observe_focus_no_bbox_no_anchor_raises_panel_not_found():
    parsed = {"panels": [], "elements": []}
    adapter = _adapter_with_parse(_fake_image_bytes(200, 200), parsed)
    adapter.observe(session_id="t")
    with pytest.raises(NoRegionForPanel, match="panel_not_found"):
        adapter.observe_focus(session_id="t", panel_id="ghost")


def test_observe_focus_consumed_anchor_then_cursor_anchor_stale():
    """After the anchor is consumed, a second focus on the same transient → cursor_anchor_stale."""
    adapter = _adapter_with_parse_sequence(
        _fake_image_bytes(1000, 600),
        [{"panels": [], "elements": []}, _FOCUS_MENU],
    )
    adapter.observe(session_id="t")  # empty
    adapter._cursor_anchor = CursorAnchor(
        session_id="t",
        click_xy=(300, 400),
        expected_transient_kind="context_menu",
        source_state_sha256=adapter._last_state_sha256,
    )
    adapter.observe_focus(session_id="t", panel_id="context_menu")  # consumes the anchor
    with pytest.raises(NoRegionForPanel, match="cursor_anchor_stale"):
        adapter.observe_focus(session_id="t", panel_id="context_menu")


def test_observe_focus_anchor_session_mismatch_raises_session_mismatch():
    """An anchor from another session is not used → session_mismatch."""
    parsed = {"panels": [], "elements": []}
    adapter = _adapter_with_parse(_fake_image_bytes(1000, 600), parsed)
    adapter.observe(session_id="t")
    adapter._cursor_anchor = CursorAnchor(
        session_id="other",
        click_xy=(300, 400),
        expected_transient_kind="context_menu",
        source_state_sha256=None,
    )
    with pytest.raises(NoRegionForPanel, match="session_mismatch"):
        adapter.observe_focus(session_id="t", panel_id="context_menu")
