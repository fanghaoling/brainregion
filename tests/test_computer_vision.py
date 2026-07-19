"""Tests for VisionAdapter: observe() scene build + observe_focus() bbox mapping.

The VLM call (``_parse`` -> ``requests.post``) is monkeypatched to a fixed parse; no
network. PIL is a pyautogui dependency — skip the module if absent.
"""

from __future__ import annotations

import io

import pytest

from brainregion.computer import VISION_PRESETS, VisionAdapter

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
    obs = adapter.observe_focus(session_id="t", region=(0, 60, 520, 550))
    # synthetic context_menu panel is present (menu_item -> context_menu post-process)
    assert any(p.panel_id == "context_menu" for p in obs.panels)
    elem = [e for e in obs.elements if "3d object" in (e.label or "").lower()][0]
    assert adapter._bbox_map[elem.element_id] == [182, 541, 520, 557]
    # mapped center lands inside the crop region (full-screen pixels)
    cx = (182 + 520) / 2 / 1000 * 1000
    cy = (541 + 557) / 2 / 1000 * 600
    assert 0 <= cx <= 520 and 60 <= cy <= 550
