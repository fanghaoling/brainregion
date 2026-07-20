"""Tests for VisionAdapter: observe() scene build + observe_focus() bbox mapping.

The VLM call (``_parse`` -> ``requests.post``) is monkeypatched to a fixed parse; no
network. PIL is a pyautogui dependency — skip the module if absent.
"""

from __future__ import annotations

import io

import pytest

from brainregion.computer import VISION_PRESETS, VisionAdapter
from brainregion.computer.adapter import NoRegionForPanel
from brainregion.computer.contracts import Panel, UIElement
from brainregion.computer.vision_adapter import (
    CursorAnchor,
    VisionModelConfig,
    _build_scene,
    _semantic_state_digest,
    infer_missing_panel_parents,
    vision_config_from_env,
)

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


# --- 缝 7: bbox-containment parent inference ---


def _panel(pid, role, label=None, transient_kind=None, parent=None):
    return Panel(
        panel_id=pid,
        role=role,
        label=label or pid,
        transient_kind=transient_kind,
        parent_panel_id=parent,
    )


def test_infer_nests_panel_inside_container():
    panels = (
        _panel("outer", "group", "Outer"),
        _panel("inner", "group", "Inner"),
    )
    bbox = {"outer": [0, 0, 1000, 1000], "inner": [100, 100, 200, 200]}
    result = infer_missing_panel_parents(panels, bbox)
    by_id = {p.panel_id: p for p in result}
    assert by_id["inner"].parent_panel_id == "outer"
    assert by_id["outer"].parent_panel_id is None


def test_infer_smallest_containing_container_wins():
    # outer ⊃ middle ⊃ inner; inner's parent is middle (tightest), middle's is outer
    panels = (
        _panel("outer", "group", "Outer"),
        _panel("middle", "section", "Middle"),
        _panel("inner", "group", "Inner"),
    )
    bbox = {
        "outer": [0, 0, 1000, 1000],
        "middle": [100, 100, 800, 800],
        "inner": [200, 200, 300, 300],
    }
    result = infer_missing_panel_parents(panels, bbox)
    by_id = {p.panel_id: p for p in result}
    assert by_id["inner"].parent_panel_id == "middle"
    assert by_id["middle"].parent_panel_id == "outer"
    assert by_id["outer"].parent_panel_id is None


def test_infer_tiebreak_smaller_area_beats_larger():
    # both big and tight contain inner; tight (smaller area) wins
    panels = (
        _panel("big", "group", "Big"),
        _panel("tight", "group", "Tight"),
        _panel("inner", "group", "Inner"),
    )
    bbox = {
        "big": [0, 0, 1000, 1000],
        "tight": [150, 150, 250, 250],
        "inner": [180, 180, 200, 200],
    }
    result = infer_missing_panel_parents(panels, bbox)
    by_id = {p.panel_id: p for p in result}
    assert by_id["inner"].parent_panel_id == "tight"


def test_infer_excludes_transient_panels_as_independent_roots():
    # a transient inside a container stays parent=None (overlay, not structural child);
    # a transient is also not a container candidate.
    panels = (
        _panel("container", "group", "C"),
        _panel("popup", "popup", "P", transient_kind="popup"),
    )
    bbox = {"container": [0, 0, 1000, 1000], "popup": [400, 400, 600, 600]}
    result = infer_missing_panel_parents(panels, bbox)
    by_id = {p.panel_id: p for p in result}
    assert by_id["popup"].parent_panel_id is None
    assert by_id["container"].parent_panel_id is None


def test_infer_excludes_non_nestable_roles():
    # role "other" (non-nestable) inside a group → not nested, and not a container
    panels = (
        _panel("grp", "group", "G"),
        _panel("deco", "other", "Deco"),
    )
    bbox = {"grp": [0, 0, 1000, 1000], "deco": [100, 100, 200, 200]}
    result = infer_missing_panel_parents(panels, bbox)
    by_id = {p.panel_id: p for p in result}
    assert by_id["deco"].parent_panel_id is None


def test_infer_identical_bboxes_stay_siblings():
    panels = (
        _panel("a", "group", "A"),
        _panel("b", "group", "B"),
    )
    same = [100, 100, 500, 500]
    bbox = {"a": list(same), "b": list(same)}
    result = infer_missing_panel_parents(panels, bbox)
    by_id = {p.panel_id: p for p in result}
    assert by_id["a"].parent_panel_id is None
    assert by_id["b"].parent_panel_id is None


def test_infer_zero_area_bbox_skipped():
    # a zero-height bbox is neither nested nor a container
    panels = (
        _panel("big", "group", "Big"),
        _panel("flat", "group", "Flat"),
    )
    bbox = {"big": [0, 0, 1000, 1000], "flat": [100, 100, 200, 100]}
    result = infer_missing_panel_parents(panels, bbox)
    by_id = {p.panel_id: p for p in result}
    assert by_id["flat"].parent_panel_id is None


def test_infer_preserves_explicit_parent_over_containment():
    # explicit parent (adapter/VLM) wins; containment does not override it
    panels = (
        _panel("outer", "group", "Outer"),
        _panel("declared-parent", "group", "DP"),
        _panel("explicit", "section", "E", parent="declared-parent"),
    )
    bbox = {
        "outer": [0, 0, 1000, 1000],
        "declared-parent": [50, 50, 250, 250],
        "explicit": [100, 100, 200, 200],
    }
    result = infer_missing_panel_parents(panels, bbox)
    by_id = {p.panel_id: p for p in result}
    assert by_id["explicit"].parent_panel_id == "declared-parent"  # not overridden to outer
    assert by_id["declared-parent"].parent_panel_id == "outer"  # inferred (no explicit parent)


def test_observe_infers_containment_parents_for_nestable_panels():
    # integration: a group nested inside a window (both nestable, contained bbox) is wired
    # through observe() → _build_scene → infer_missing_panel_parents.
    parsed = {
        "panels": [
            {"name": "Root Window", "role": "window", "bbox": [0, 0, 1000, 1000]},
            {"name": "Inner Group", "role": "group", "bbox": [100, 100, 300, 300]},
        ],
        "elements": [],
    }
    adapter = _adapter_with_parse(_fake_image_bytes(1000, 600), parsed)
    obs = adapter.observe(session_id="t")
    by_role = {p.role: p for p in obs.panels}
    assert by_role["group"].parent_panel_id == by_role["window"].panel_id
    assert by_role["window"].parent_panel_id is None


# ---------------------------------------------------------------------------
# G plan S3 — dual digest (raw artifact identity + semantic structural state)
# ---------------------------------------------------------------------------


def _scene_parsed(*, with_element: bool = True) -> dict:
    """Minimal VLM parse for _build_scene: one inspector panel + one Position X input."""
    elements = (
        [{"panel": "Inspector", "role": "input", "label": "position x", "bbox": [10, 10, 40, 30]}]
        if with_element
        else []
    )
    return {
        "panels": [{"name": "Inspector", "role": "inspector", "bbox": [0, 0, 100, 100]}],
        "elements": elements,
    }


def test_semantic_state_digest_stable_across_element_order():
    """Canonical sort: same elements in different order → same semantic digest."""
    e1 = [
        UIElement(element_id="inspector-1", role="input", label="y", panel_id="inspector"),
        UIElement(element_id="inspector-0", role="input", label="x", panel_id="inspector"),
    ]
    e2 = [
        UIElement(element_id="inspector-0", role="input", label="x", panel_id="inspector"),
        UIElement(element_id="inspector-1", role="input", label="y", panel_id="inspector"),
    ]
    panels = [_panel("inspector", "inspector", "Inspector")]
    assert _semantic_state_digest(e1, panels) == _semantic_state_digest(e2, panels)


def test_semantic_state_digest_changes_on_real_ui_change():
    """A real UI change (element added / label changed) → semantic digest changes."""
    panels = [_panel("inspector", "inspector", "Inspector")]
    base = [UIElement(element_id="inspector-0", role="input", label="position x", panel_id="inspector")]
    added = base + [UIElement(element_id="inspector-1", role="input", label="position y", panel_id="inspector")]
    relabeled = [UIElement(element_id="inspector-0", role="input", label="position z", panel_id="inspector")]
    d_base = _semantic_state_digest(base, panels)
    assert _semantic_state_digest(added, panels) != d_base
    assert _semantic_state_digest(relabeled, panels) != d_base


def test_build_scene_raw_mode_state_matches_raw_digest():
    """digest_mode='raw' (default): state_sha256 == the raw image-bytes digest; frame.sha256
    also raw (artifact identity)."""
    obs, _bbox, _pbox = _build_scene(
        _scene_parsed(),
        session_id="t",
        sequence=1,
        app_id="unity.editor",
        window_id="w",
        digest="a" * 64,
        digest_mode="raw",
    )
    assert obs.state_sha256 == "a" * 64
    assert obs.frame.sha256 == "a" * 64


def test_build_scene_semantic_mode_filters_byte_noise_nuisance_invariance():
    """GPT #6 nuisance invariance: same logical scene, different image bytes (cursor blink,
    PNG re-encode, sub-pixel jitter) → raw state flips but semantic state is stable. Under
    BoundFreshness the semantic state_sha256 won't flip → no false stale."""
    parsed = _scene_parsed()
    obs_raw1, _, _ = _build_scene(
        parsed, session_id="t", sequence=1, app_id="unity.editor", window_id="w", digest="a" * 64, digest_mode="raw"
    )
    obs_raw2, _, _ = _build_scene(
        parsed, session_id="t", sequence=1, app_id="unity.editor", window_id="w", digest="b" * 64, digest_mode="raw"
    )
    obs_sem1, _, _ = _build_scene(
        parsed,
        session_id="t",
        sequence=1,
        app_id="unity.editor",
        window_id="w",
        digest="a" * 64,
        digest_mode="semantic",
    )
    obs_sem2, _, _ = _build_scene(
        parsed,
        session_id="t",
        sequence=1,
        app_id="unity.editor",
        window_id="w",
        digest="b" * 64,
        digest_mode="semantic",
    )
    assert obs_raw1.state_sha256 != obs_raw2.state_sha256  # raw: byte noise → false state_changed
    assert obs_sem1.state_sha256 == obs_sem2.state_sha256  # semantic: stable under byte noise
    assert obs_sem1.frame.sha256 == "a" * 64  # frame.sha256 stays raw (artifact identity / debug)
    assert obs_sem2.frame.sha256 == "b" * 64


def test_build_scene_semantic_mode_detects_structural_change_sensitivity():
    """GPT #6 semantic sensitivity: a real UI change (element removed) flips the semantic
    digest (a pure quantized-image digest might swallow a small change; structural does not)."""
    obs_with, _, _ = _build_scene(
        _scene_parsed(with_element=True),
        session_id="t",
        sequence=1,
        app_id="unity.editor",
        window_id="w",
        digest="a" * 64,
        digest_mode="semantic",
    )
    obs_without, _, _ = _build_scene(
        _scene_parsed(with_element=False),
        session_id="t",
        sequence=1,
        app_id="unity.editor",
        window_id="w",
        digest="a" * 64,
        digest_mode="semantic",
    )
    assert obs_with.state_sha256 != obs_without.state_sha256


def test_observe_semantic_digest_mode_propagates_to_state_sha256():
    """End-to-end via observe(): digest_mode='semantic' makes state_sha256 the structural
    digest (≠ raw bytes); frame.sha256 stays raw; _last_state_sha256 tracks state."""
    import hashlib
    from dataclasses import replace as _replace

    parsed = _scene_parsed()
    cfg = VISION_PRESETS["siliconflow-qwen3vl-8b"]
    assert cfg.digest_mode == "raw"  # default preserved
    cfg_sem = _replace(cfg, digest_mode="semantic")
    adapter = VisionAdapter(vision=cfg_sem, screenshot=lambda: _fake_image_bytes(200, 200), api_key="fake")
    adapter._parse = lambda ib: parsed  # type: ignore[method-assign]
    obs = adapter.observe(session_id="t")
    raw = hashlib.sha256(_fake_image_bytes(200, 200)).hexdigest()
    assert obs.state_sha256 != raw  # semantic, not raw bytes
    assert obs.frame.sha256 == raw  # frame stays raw artifact identity
    assert adapter._last_state_sha256 == obs.state_sha256


# ---------------------------------------------------------------------------
# G plan 视觉配置: thinking 控制(推理 VLM 兼容)+ env 自定义(像 consult_panel 灵活配置)
# ---------------------------------------------------------------------------


class _FakeVlmResp:
    status_code = 200
    text = '{"choices":[{"message":{"content":"{}"}}]}'

    def json(self):
        return {"choices": [{"message": {"content": "{}"}}]}


def _capture_payload(adapter, monkeypatch):
    captured = {}

    def fake_post(url, **kw):
        captured.update(kw.get("json", {}))
        return _FakeVlmResp()

    import brainregion.computer.vision_adapter as va

    monkeypatch.setattr(va.requests, "post", fake_post)
    adapter._parse(_fake_image_bytes(10, 10))
    return captured


def test_vision_thinking_disabled_injects_payload(monkeypatch):
    """cfg.thinking=False → _parse payload 含 thinking:{type:disabled}(关推理 VLM 的思考)。"""
    cfg = VisionModelConfig("t", "glm-4.6v", "http://x/v1", "K", thinking=False)
    adapter = VisionAdapter(vision=cfg, screenshot=lambda: _fake_image_bytes(10, 10), api_key="k")
    payload = _capture_payload(adapter, monkeypatch)
    assert payload["thinking"] == {"type": "disabled"}


def test_vision_thinking_none_omits_from_payload(monkeypatch):
    """cfg.thinking=None(默认)→ payload 不含 thinking(用 provider 默认,如硅基流动 Qwen3-VL)。"""
    cfg = VisionModelConfig("t", "qwen3vl", "http://x/v1", "K")  # thinking=None
    adapter = VisionAdapter(vision=cfg, screenshot=lambda: _fake_image_bytes(10, 10), api_key="k")
    payload = _capture_payload(adapter, monkeypatch)
    assert "thinking" not in payload


def test_vision_config_from_env_overrides_preset(monkeypatch):
    """env VISION_MODEL/BASE_URL/THINKING/DIGEST_MODE → vision_config_from_env 构造自定义 cfg
    (像 consult_panel 那样灵活配置,不硬编码 VISION_PRESETS)。"""
    monkeypatch.setenv("VISION_MODEL", "my-glm")
    monkeypatch.setenv("VISION_BASE_URL", "http://x/v1/")
    monkeypatch.setenv("VISION_THINKING", "false")
    monkeypatch.setenv("VISION_DIGEST_MODE", "semantic")
    cfg = vision_config_from_env()
    assert cfg is not None
    assert cfg.model == "my-glm"
    assert cfg.base_url == "http://x/v1"  # trailing slash stripped
    assert cfg.thinking is False
    assert cfg.digest_mode == "semantic"


def test_vision_config_from_env_none_when_no_model(monkeypatch):
    """未设 VISION_MODEL → None(caller 回落 VISION_PRESETS)。"""
    monkeypatch.delenv("VISION_MODEL", raising=False)
    assert vision_config_from_env() is None
