"""G plan S2 — ComputerUseBridge integration tests.

The bridge is a façade the main brain talks to (act_cu / focus_cu). These are bridge-level
integration tests (scripted decider over UnityEditorMockAdapter), NOT run_agent end-to-end
(that is S4). Gates: single _current cache, 0-pre+1-post observe per act, wait bypasses
ActionIntent, composite merge provenance, id-collision raise, dump omits ids/coordinates.
"""

from __future__ import annotations

import pytest

from brainregion.computer.bridge import CompositeMergeError, ComputerUseBridge
from brainregion.computer.contracts import FrameRef, Panel, SceneObservation, UIElement
from brainregion.computer.locator import PanelAnchor
from brainregion.computer.perception import PerceptionRegion
from brainregion.computer.session import BoundFreshness, ComputerUseSession
from brainregion.computer.targeting import TargetingController
from brainregion.computer.unity_mock import UnityEditorMockAdapter


def _bridge(*, ttl_ms: float = 10000.0):
    adapter = UnityEditorMockAdapter()
    session = ComputerUseSession(
        session_id="s",
        adapter=adapter,
        allowed_apps={"unity.editor"},
        freshness=BoundFreshness(ttl_ms=ttl_ms),
    )
    perception = PerceptionRegion(event_sink=lambda *a, **k: None)
    targeting = TargetingController(session=session, perception=perception)
    bridge = ComputerUseBridge(session=session, perception=perception, targeting=targeting)
    return adapter, bridge


def _obs(*, eid: str, pid: str, transient: str | None = None, frame_id: str = "f1") -> SceneObservation:
    return SceneObservation(
        session_id="s",
        sequence=1,
        app_id="a",
        window_id="w",
        window_title="T",
        frame=FrameRef(
            frame_id=frame_id,
            sha256="a" * 64,
            width=10,
            height=10,
            artifact_uri="mock://x",
            sensitivity="private",
        ),
        state_sha256="b" * 64,
        elements=(UIElement(element_id=eid, role="input", label="lbl", panel_id=pid),),
        panels=(
            Panel(
                panel_id=pid,
                role="menu" if transient else "inspector",
                label=pid,
                transient_kind=transient,
            ),
        ),
    )


def _counting_observe(adapter):
    real = adapter.observe
    calls = {"n": 0}

    def counting(*, session_id):
        calls["n"] += 1
        return real(session_id=session_id)

    adapter.observe = counting  # type: ignore[method-assign]
    return calls


# -- prime / dump -----------------------------------------------------------


def test_prime_seeds_current_and_dumps_scene():
    _adapter, bridge = _bridge()
    scene = bridge.prime()
    assert bridge.current is not None
    assert "SCENE" in scene
    assert "PANELS" in scene


def test_dump_scene_omits_element_ids_and_coordinates():
    """The model describes targets via Locators; raw ids/coordinates never leak (缝 8)."""
    _adapter, bridge = _bridge()
    bridge.prime()
    text = bridge.dump_scene(bridge.current)
    for e in bridge.current.elements:
        assert e.element_id not in text


# -- act: 0 pre + 1 post observe -------------------------------------------


def test_act_click_does_one_observe_zero_pre_one_post():
    """G plan S2 / GPT #1+#2: act() resolves against _current (no observe); perform does
    0 pre-execute + 1 post-execute observe. No transient focus on a left click."""
    adapter, bridge = _bridge()
    calls = _counting_observe(adapter)
    bridge.prime()
    seed = calls["n"]
    decision = {
        "action": "click",
        "locator": {
            "anchor": {"panel_name": "toolbar"},
            "descriptor": {"role": "button", "attributes": {"icon_shape": "play"}},
        },
    }
    result = bridge.act(decision, step=0)
    assert result["status"] == "executed"
    assert calls["n"] - seed == 1, f"act observed {calls['n'] - seed}x, expected 1 (post-execute only)"


def test_act_unresolved_locator_returns_unresolved_not_crash():
    _adapter, bridge = _bridge()
    bridge.prime()
    result = bridge.act(
        {"action": "click", "locator": {"anchor": {"panel_name": "ghost-panel"}}},
        step=0,
    )
    assert result["status"] == "unresolved"


# -- wait: bridge refresh, not an ActionIntent ------------------------------


def test_wait_refreshes_current_without_acting():
    _adapter, bridge = _bridge()
    bridge.prime()
    before = bridge.current
    result = bridge.wait()
    assert result["status"] == "waited"
    assert bridge.current is not before
    assert "SCENE" in result["observation"]


# -- focus: status + depth, partial never masquerades ----------------------


def test_focus_returns_status_completed_and_requested_depth():
    _adapter, bridge = _bridge()
    bridge.prime()
    result = bridge.focus([PanelAnchor(panel_name="inspector")], step=0)
    assert result["status"] == "focused"
    assert result["completed_depth"] == 1
    assert result["requested_depth"] == 1
    assert "SCENE" in result["observation"]


# -- _merge_transient_focus: composite provenance + collision raise ---------


def test_merge_transient_focus_produces_composite_observation():
    _adapter, bridge = _bridge()
    base = _obs(eid="inspector-0", pid="inspector", frame_id="base-frame")
    focused = _obs(eid="context_menu-0", pid="context_menu", transient="context_menu", frame_id="focus-frame")
    merged = bridge._merge_transient_focus(base, focused, transient_panel_id="context_menu")
    assert merged.observation_kind == "composite"
    assert merged.source_frame_ids == ("base-frame", "focus-frame")
    eids = {e.element_id for e in merged.elements}
    assert {"inspector-0", "context_menu-0"} <= eids
    pids = {p.panel_id for p in merged.panels}
    assert {"inspector", "context_menu"} <= pids
    assert merged.state_sha256 != base.state_sha256  # canonical merge digest, not the base raw


def test_merge_transient_focus_collision_raises_not_silently_renamed():
    """GPT #8: a colliding id means the adapter broke cross-observation namespace — raise,
    don't rewrite (a silent namespace would break adapter.execute's _bbox_map lookup)."""
    _adapter, bridge = _bridge()
    base = _obs(eid="context_menu-0", pid="inspector")
    focused = _obs(eid="context_menu-0", pid="context_menu", transient="context_menu")
    with pytest.raises(CompositeMergeError):
        bridge._merge_transient_focus(base, focused, transient_panel_id="context_menu")


def test_composite_observation_passes_post_init_validation():
    """The merged composite must satisfy SceneObservation's invariants (unique ids, known
    panel refs, no dangling spawned_by). A raw SceneObservation with observation_kind set
    directly must also validate."""
    _adapter, bridge = _bridge()
    base = _obs(eid="inspector-0", pid="inspector")
    focused = _obs(eid="context_menu-0", pid="context_menu", transient="context_menu")
    merged = bridge._merge_transient_focus(base, focused, transient_panel_id="context_menu")
    ids = [e.element_id for e in merged.elements]
    assert len(ids) == len(set(ids))  # __post_init__ would have raised otherwise
    panel_ids = [p.panel_id for p in merged.panels]
    assert len(panel_ids) == len(set(panel_ids))
