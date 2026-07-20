"""G plan S5 — freshness-before-live CI gate (VisionAdapter + byte-noise + Strict/Bound).

The hard gate that S1 (FreshnessPolicy) + S3 (dual digest) + S4 (bridge wiring) must pass
TOGETHER under the real freshness trap: byte-level screenshot noise (cursor blink, PNG
re-encode, sub-pixel jitter) flips the raw digest every observe. A real VLM re-parse of the
same logical scene is stable, so:

- StrictFreshness + raw digest: re-observe sees a changed raw digest → stale (the pilot trap).
- BoundFreshness + semantic digest: binds _latest (no re-observe) + semantic state is stable
  under byte noise → executes, state_changed=False, exactly 1 VLM call per act.

No real VLM (``_parse`` mocked to a fixed scene) and no real screen (``execute`` mocked) —
this is a deterministic CI gate. The live-VLM + live-Unity run is S6 (manual).
"""

from __future__ import annotations

import io

import pytest

pytest.importorskip("PIL.Image")
from PIL import Image  # noqa: E402

from brainregion.computer import VISION_PRESETS, VisionAdapter  # noqa: E402
from brainregion.computer.adapter import AdapterExecution  # noqa: E402
from brainregion.computer.bridge import ComputerUseBridge  # noqa: E402
from brainregion.computer.perception import PerceptionRegion  # noqa: E402
from brainregion.computer.session import (  # noqa: E402
    BoundFreshness,
    ComputerUseSession,
    StrictFreshness,
)
from brainregion.computer.targeting import TargetingController  # noqa: E402

# Fixed logical scene (toolbar + Play button). _parse always returns this regardless of the
# perturbed screenshot bytes → semantic digest stable, raw digest flips with byte noise.
_FIXED_PARSE = {
    "panels": [{"name": "Toolbar", "role": "toolbar", "bbox": [0, 0, 1000, 60]}],
    "elements": [
        {
            "panel": "toolbar",
            "role": "button",
            "label": "Play",
            "bbox": [10, 10, 50, 50],
            "attributes": {"icon_shape": "play"},
            "interactable": True,
        }
    ],
}

_DECISION = {
    "action": "click",
    "locator": {
        "anchor": {"panel_name": "toolbar"},
        "descriptor": {"role": "button", "label": "play"},
    },
}


def _fake_bytes(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def _perturbing_screenshot(base_bytes: bytes):
    """Each call flips one pixel + re-encodes PNG, so raw sha256 changes every observe while
    the logical scene (mock _parse) stays identical — the cursor-blink/PNG-reencode trap."""

    state = {"n": 0}

    def shot():
        img = Image.open(io.BytesIO(base_bytes)).convert("RGB")
        state["n"] += 1
        px = img.load()
        y = state["n"] % img.height
        r, g, b = px[0, y]
        px[0, y] = (r ^ 1, g, b)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    return shot


def _vision_bridge(digest_mode: str, freshness):
    from dataclasses import replace

    cfg = replace(VISION_PRESETS["siliconflow-qwen3vl-8b"], digest_mode=digest_mode)
    adapter = VisionAdapter(
        vision=cfg,
        screenshot=_perturbing_screenshot(_fake_bytes(200, 200)),
        api_key="fake",
    )
    adapter._parse = lambda ib: _FIXED_PARSE  # type: ignore[method-assign] — no real VLM
    adapter.execute = lambda intent: AdapterExecution(True, "mocked")  # type: ignore[method-assign] — no pyautogui/screen
    session = ComputerUseSession(
        session_id="cu",
        adapter=adapter,
        allowed_apps={"unity.editor"},
        freshness=freshness,
    )
    perception = PerceptionRegion(event_sink=lambda *a, **k: None)
    targeting = TargetingController(session=session, perception=perception)
    return ComputerUseBridge(session=session, perception=perception, targeting=targeting), adapter


def test_strict_raw_reports_stale_on_byte_noise():
    """The pilot trap, reproduced: StrictFreshness re-observes, byte noise flips the raw
    digest, the bound observation is reported stale. This is the对照 — BoundFreshness must
    NOT do this."""
    bridge, _adapter = _vision_bridge("raw", StrictFreshness())
    bridge.prime()
    result = bridge.act(_DECISION, step=0)
    assert result["status"] == "stale", f"StrictFreshness+raw should go stale under byte noise; got {result}"


def test_bound_semantic_tolerates_byte_noise_executes_with_stable_state():
    """BoundFreshness binds _latest (no re-observe) + semantic digest is stable under byte
    noise → executes, state_changed=False (semantic unchanged though raw bytes flipped)."""
    bridge, _adapter = _vision_bridge("semantic", BoundFreshness(ttl_ms=10000.0))
    bridge.prime()
    result = bridge.act(_DECISION, step=0)
    assert result["status"] == "executed", f"BoundFreshness+semantic should execute; got {result}"
    assert result["state_changed"] is False, "semantic digest must be stable under byte noise"


def test_bound_semantic_exactly_one_vlm_call_per_act():
    """0 pre-execute + 1 post-execute VLM call per act (the pilot double-VLM bug is gone).
    Counts _parse calls (VLM) across one act after prime()."""
    bridge, adapter = _vision_bridge("semantic", BoundFreshness(ttl_ms=10000.0))
    calls = {"n": 0}
    real_parse = adapter._parse

    def counting(ib):
        calls["n"] += 1
        return real_parse(ib)

    adapter._parse = counting  # type: ignore[method-assign]
    bridge.prime()
    seed = calls["n"]
    bridge.act(_DECISION, step=0)
    delta = calls["n"] - seed
    assert delta == 1, f"act used {delta} VLM calls, expected 1 (post-execute observe only)"
