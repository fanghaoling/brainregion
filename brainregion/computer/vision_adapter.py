"""VisionAdapter: a ComputerUseAdapter backed by a VLM screen parser.

``observe()`` captures a screenshot, sends it to a VLM (OpenAI-compatible chat API)
that returns structured panels + elements (role/label/bbox), and builds a
``SceneObservation``. Coordinates stay INSIDE the adapter (an ``element_id -> bbox``
map); the contract layer never sees pixels — exactly the design principle
"absolute coordinates are an adapter implementation detail, never enter the contract".

``execute()`` maps an ``ActionIntent``'s ``target_id`` back to its bbox and drives
pyautogui (click / type / hover / press_key). This is the real-vision counterpart to
``UnityEditorMockAdapter``: same contract, observation source swapped from a state
machine to a VLM parse of live pixels.

Below-fold note: a VLM only sees the current viewport. An element scrolled out of view
(eg Add Component when the Inspector is long) is simply ABSENT from the parse — it
shows up as ``not_found`` at resolve time, not as a ``below_fold`` marker. The
``TargetingController`` reveal loop (scroll -> re-observe -> re-resolve) handles this.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import requests

from .adapter import AdapterExecution, NoRegionForPanel
from .contracts import ActionIntent, FrameRef, Panel, SceneObservation, UIElement


# ---------------------------------------------------------------- model config


@dataclass(frozen=True)
class VisionModelConfig:
    """A VLM endpoint + sampling params. OpenAI-compatible chat/completions."""

    name: str  # preset key
    model: str
    base_url: str  # up to /v1 (no trailing slash)
    api_key_env: str
    max_tokens: int = 4096
    temperature: float = 0.1
    digest_mode: str = "raw"  # G plan S3: "raw" (sha256 of image bytes, today's behavior) | "semantic" (sha256 of canonical scene structure — filters cursor/PNG/sub-pixel noise)


PRESETS: dict[str, VisionModelConfig] = {
    "siliconflow-qwen3vl-32b": VisionModelConfig(
        "siliconflow-qwen3vl-32b", "Qwen/Qwen3-VL-32B-Instruct", "https://api.siliconflow.cn/v1", "SILICONFLOW_API_KEY"
    ),
    "siliconflow-qwen3vl-8b": VisionModelConfig(
        "siliconflow-qwen3vl-8b", "Qwen/Qwen3-VL-8B-Instruct", "https://api.siliconflow.cn/v1", "SILICONFLOW_API_KEY"
    ),
    "siliconflow-glm-4.5v": VisionModelConfig(
        "siliconflow-glm-4.5v", "zai-org/GLM-4.5V", "https://api.siliconflow.cn/v1", "SILICONFLOW_API_KEY"
    ),
    # Zhipu official (ZAI_API_KEY) — GLM-4.6V / GLM-5V-Turbo are newer than SiliconFlow
    # carries; model ids to confirm at test time (Zhipu vision naming convention = +v):
    "zhipu-glm-4.6v": VisionModelConfig(
        "zhipu-glm-4.6v", "glm-4.6v", "https://open.bigmodel.cn/api/paas/v4", "ZAI_API_KEY"
    ),
    "zhipu-glm-5v-turbo": VisionModelConfig(
        "zhipu-glm-5v-turbo", "glm-5v-turbo", "https://open.bigmodel.cn/api/paas/v4", "ZAI_API_KEY"
    ),
}


# ---------------------------------------------------------------- parse prompt

PARSE_PROMPT = """\
Parse this Unity Editor screenshot into structured UI elements. Output ONLY a JSON \
object (no prose, no markdown fences) with this exact shape:
{
  "panels": [
    {"name": "<panel label>", "role": "hierarchy|inspector|scene|game|toolbar|project|console|menu_bar|menu|popup|other", "transient_kind": "persistent|context_menu|submenu|popup", "bbox": [x1,y1,x2,y2]}
  ],
  "elements": [
    {"panel": "<panel name or role>", "role": "button|menu_item|input|tab|list_item|tree_item|dropdown|slider|header|component|other", "label": "<visible text/label>", "bbox": [x1,y1,x2,y2], "interactable": true}
  ]
}
Rules:
- Coordinates are NORMALIZED to [0,1000], origin top-left, bbox = [x1,y1,x2,y2].
- First identify the main panel regions (Hierarchy left, Inspector right, Scene \
center, Game, Toolbar top, Project/Console bottom, menu bar). Mark transient_kind: \
"persistent" for docked panels, "context_menu" for a right-click popup menu, "submenu" \
for a menu's expanded sub-items, "popup" for a floating dialog (e.g. Add Component search).
- Then list INTERACTABLE UI elements inside each panel: buttons, menu items, tabs, input \
fields, list/tree items, dropdowns, sliders, headers. Include their visible label/text.
- Do NOT treat 3D objects inside the Scene/Game viewport as UI elements — the viewport is \
just one panel. (But DO list Hierarchy tree items, which represent those objects.)
"""


# ----------------------------------------------------------------- helpers


def _norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def _semantic_state_digest(elements: list[UIElement], panels: list[Panel]) -> str:
    """Structural digest over the parsed scene (G plan S3).

    Filters visual noise — cursor blink, PNG re-encode, sub-pixel jitter, downsample
    rounding — that flips the raw byte digest but leaves the parsed structure unchanged.
    Two logically-identical scenes yield the same semantic digest; a real UI change
    (checkbox toggle, value edit, element add/remove) changes it.

    Residual noise: VLM non-determinism (a re-parse may drop a field) can still flip this
    — that is §187 multi-observation fusion's problem, not S3's. S3 only separates raw
    (artifact identity / debug) from semantic (state identity / change detection).
    """
    payload = {
        "panels": [
            {
                "id": p.panel_id,
                "role": p.role,
                "label": p.label,
                "parent": p.parent_panel_id,
                "transient": p.transient_kind,
            }
            for p in sorted(panels, key=lambda x: x.panel_id)
        ],
        "elements": [
            {
                "id": e.element_id,
                "role": e.role,
                "label": e.label,
                "panel": e.panel_id,
                "enabled": e.enabled,
                "visible": e.visible,
                "attrs": sorted((e.attribute_map() or {}).items()),
            }
            for e in sorted(elements, key=lambda x: x.element_id)
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _build_scene(
    parsed: dict,
    *,
    session_id: str,
    sequence: int,
    app_id: str,
    window_id: str,
    digest: str,
    digest_mode: str = "raw",
) -> tuple[SceneObservation, dict[str, list[int]], dict[str, list[int]]]:
    """Map VLM panels/elements -> (SceneObservation, element_id->bbox, panel_id->bbox).

    bbox stays out of the contract (returned separately for execute() and panel viz).
    ``digest`` is always the raw sha256 of image bytes (FrameRef.sha256 = artifact
    identity). ``state_sha256`` follows ``digest_mode``: "raw" (today's behavior) or
    "semantic" (structural digest that filters visual noise — G plan S3).
    """
    pid_by_key: dict[str, str] = {}
    panels: list[Panel] = []
    panel_bbox_map: dict[str, list[int]] = {}
    used_pids: set[str] = set()

    def _unique_pid(base: str) -> str:
        # 缝 8: panel_id must be unique within the scene. Same-role siblings used to
        # collide (pid = role) and get silently dropped by the dup-skip — a nesting
        # blocker. Disambiguate by suffix so every parsed panel survives.
        if base not in used_pids:
            used_pids.add(base)
            return base
        i = 2
        while f"{base}-{i}" in used_pids:
            i += 1
        used_pids.add(f"{base}-{i}")
        return f"{base}-{i}"

    for p in parsed.get("panels", []):
        role = _norm(p.get("role")) or "other"
        name = str(p.get("name") or p.get("role") or "panel")
        base = role if role != "other" else _norm(name)
        pid = _unique_pid(base)
        tk_raw = _norm(p.get("transient_kind"))
        tk = None if tk_raw in ("", "persistent", "none", "null") else tk_raw
        panels.append(Panel(panel_id=pid, role=role, label=name, transient_kind=tk))
        pid_by_key[_norm(name)] = pid
        pid_by_key[role] = pid
        pbbox = p.get("bbox") or []
        if len(pbbox) == 4:
            panel_bbox_map[pid] = [int(v) for v in pbbox]

    by_panel: dict[str, list[dict]] = {}
    unassigned = 0
    for e in parsed.get("elements", []):
        pstr = _norm(e.get("panel"))
        pid = pid_by_key.get(pstr)
        if pid is None and pstr:
            for k, v in pid_by_key.items():
                if pstr in k or k in pstr:
                    pid = v
                    break
        if pid is None:
            unassigned += 1
            continue
        by_panel.setdefault(pid, []).append(e)

    elements: list[UIElement] = []
    bbox_map: dict[str, list[int]] = {}
    for pid, elems in by_panel.items():
        elems.sort(key=lambda e: (e.get("bbox") or [0, 0, 0, 0])[1])  # top -> bottom
        for i, e in enumerate(elems):
            role = _norm(e.get("role")) or "element"
            label = str(e.get("label") or "")
            eid = f"{pid}-{i}"
            # context-menu items: VLM inconsistently files them under hierarchy/other;
            # normalize into a synthetic context_menu transient panel (not menu_bar's).
            elem_pid = "context_menu" if role == "menu_item" and pid != "menu_bar" else pid
            elements.append(
                UIElement(
                    element_id=eid,
                    role=role,
                    label=label,
                    panel_id=elem_pid,
                    enabled=bool(e.get("interactable", True)),
                    visible=True,
                )
            )
            bbox = e.get("bbox") or []
            if len(bbox) == 4:
                bbox_map[eid] = [int(v) for v in bbox]

    # synthetic context_menu transient panel for any menu_items we normalized above.
    if any(e.panel_id == "context_menu" for e in elements) and not any(p.panel_id == "context_menu" for p in panels):
        panels.append(Panel(panel_id="context_menu", role="menu", label="Context Menu", transient_kind="context_menu"))

    # 缝 7: infer parent_panel_id for persistent nestable panels via bbox containment.
    # Transients (incl. the synthetic context_menu above) stay independent roots; an
    # explicit parent (already set) is never overridden.
    panels = infer_missing_panel_parents(tuple(panels), panel_bbox_map)

    # G plan S3: frame.sha256 is always the raw image-bytes digest (artifact identity).
    # state_sha256 follows digest_mode — semantic filters the visual noise that would
    # otherwise flip the bound observation's state under BoundFreshness / receipt.
    state_digest = _semantic_state_digest(elements, panels) if digest_mode == "semantic" else digest

    obs = SceneObservation(
        session_id=session_id,
        sequence=sequence,
        app_id=app_id,
        window_id=window_id,
        window_title="Unity Editor",
        frame=FrameRef(
            frame_id=f"vision-{digest[:12]}",
            sha256=digest,
            width=2560,
            height=1528,
            artifact_uri="mock://vision",
            sensitivity="private",
        ),
        state_sha256=state_digest,
        elements=tuple(elements),
        panels=tuple(panels),
    )
    return obs, bbox_map, panel_bbox_map


# 缝 7: persistent-panel roles eligible for bbox-containment nesting. Decorative /
# overlay / unknown roles are excluded so a background or transient layer cannot create
# meaningless depth. Transient panels are independent roots regardless of geometry.
NESTABLE_PANEL_ROLES = frozenset({"window", "panel", "group", "section", "inspector_component"})


def _bbox_area(bbox: list[int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _bbox_contains(outer: list[int], inner: list[int]) -> bool:
    """True if ``inner`` lies entirely within ``outer`` (boundary-inclusive)."""
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _bbox_same(a: list[int], b: list[int]) -> bool:
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and a[3] == b[3]


def infer_missing_panel_parents(panels: tuple[Panel, ...], panel_bbox_map: dict[str, list[int]]) -> tuple[Panel, ...]:
    """Infer ``parent_panel_id`` for persistent nestable panels via bbox containment (缝 7).

    Priority: explicit parent (already set) > bbox containment > None — containment never
    overrides an existing parent. Only persistent panels (transients are independent roots)
    with a nestable container role AND a non-zero bbox participate, keeping decorative /
    unknown layers from creating meaningless depth. When several containers qualify, the
    smallest (tightest) wins; ties break by ``panel_id`` lexicographically. Identical
    bboxes stay siblings (neither parents the other); zero-area bboxes are skipped.
    """
    # candidate containers: persistent + nestable + non-zero bbox, sorted smallest-first
    # (tightest container wins) with a panel_id tiebreak for determinism.
    containers: list[tuple[Panel, list[int]]] = []
    for panel in panels:
        if panel.transient or panel.role not in NESTABLE_PANEL_ROLES:
            continue
        bb = panel_bbox_map.get(panel.panel_id)
        if bb is None or len(bb) != 4 or _bbox_area(bb) <= 0:
            continue
        containers.append((panel, bb))
    containers.sort(key=lambda t: (_bbox_area(t[1]), t[0].panel_id))

    parent_of: dict[str, str] = {}
    for panel in panels:
        if panel.parent_panel_id is not None or panel.transient or panel.role not in NESTABLE_PANEL_ROLES:
            continue  # explicit parent wins; transients / non-nestable excluded
        bb = panel_bbox_map.get(panel.panel_id)
        if bb is None or len(bb) != 4 or _bbox_area(bb) <= 0:
            continue
        for cand, cbb in containers:
            if cand.panel_id == panel.panel_id or _bbox_same(cbb, bb):
                continue  # never self-parent; identical bboxes stay siblings
            if _bbox_contains(cbb, bb):
                parent_of[panel.panel_id] = cand.panel_id
                break  # first (smallest) containing container wins

    if not parent_of:
        return panels
    return tuple(
        replace(panel, parent_panel_id=parent_of[panel.panel_id]) if panel.panel_id in parent_of else panel
        for panel in panels
    )


def _map_crop_bbox(crop_bbox: list[int], region: tuple[int, int, int, int], full_w: int, full_h: int) -> list[int]:
    """Map a [0,1000]-normalized bbox inside a crop back to [0,1000] full-screen normalized.

    region = (x1, y1, x2, y2) full-screen pixel box. A bbox at normalized ``v`` inside the
    crop sits at full-screen pixel ``x1 + v/1000*(x2-x1)``; re-normalize by full screen size.
    """
    x1, y1, x2, y2 = region
    cw, ch = x2 - x1, y2 - y1

    def mx(v: int) -> int:
        return int((x1 + v / 1000 * cw) / full_w * 1000)

    def my(v: int) -> int:
        return int((y1 + v / 1000 * ch) / full_h * 1000)

    return [mx(crop_bbox[0]), my(crop_bbox[1]), mx(crop_bbox[2]), my(crop_bbox[3])]


@dataclass(frozen=True)
class CursorAnchor:
    """A screen-region hint derived from the click that opened a transient panel.

    Transient panels (eg a right-click context menu) are absent from the full-screen
    parse (no bbox in ``_panel_bbox_map``), so ``observe_focus`` falls back to cropping
    around the click that opened them. Lifecycle (缝 4): invalidated by any subsequent
    execute (rule ③), consumed after one focus use, session-scoped. ``click_xy`` is a
    full-screen pixel point; ``source_state_sha256`` is the observation the click acted on.
    """

    session_id: str
    click_xy: tuple[int, int]
    expected_transient_kind: str | None
    source_state_sha256: str | None
    consumed: bool = False


def _bbox_to_region(bbox: list[int], full_w: int, full_h: int) -> tuple[int, int, int, int]:
    """Normalized [0,1000] panel bbox → full-screen pixel crop region (clamped, non-degenerate)."""
    x1 = max(0, int(bbox[0] / 1000 * full_w))
    y1 = max(0, int(bbox[1] / 1000 * full_h))
    x2 = min(full_w, int(bbox[2] / 1000 * full_w))
    y2 = min(full_h, int(bbox[3] / 1000 * full_h))
    if x2 <= x1:
        x2 = min(full_w, x1 + 1)
    if y2 <= y1:
        y2 = min(full_h, y1 + 1)
    return (x1, y1, x2, y2)


def _anchor_region(
    click_xy: tuple[int, int], full_w: int, full_h: int, *, radius: int = 320
) -> tuple[int, int, int, int]:
    """A full-screen pixel box around a cursor click — the fallback region for transients."""
    cx, cy = click_xy
    return (max(0, cx - radius), max(0, cy - radius), min(full_w, cx + radius), min(full_h, cy + radius))


# ----------------------------------------------------------------- adapter


class VisionAdapter:
    """ComputerUseAdapter over a VLM screen parser + pyautogui actuator.

    ``screenshot`` is injectable (callable returning image bytes) so observe() can be
    driven by a static image for testing without pyautogui / a live screen.
    """

    def __init__(
        self,
        *,
        vision: VisionModelConfig,
        app_id: str = "unity.editor",
        window_id: str = "unity-main",
        screenshot: Callable[[], bytes] | None = None,
        api_key: str | None = None,
    ) -> None:
        self.vision = vision
        self.app_id = app_id
        self.window_id = window_id
        self._screenshot = screenshot
        self._api_key = api_key or os.environ.get(vision.api_key_env, "")
        if not self._api_key:
            raise ValueError(f"{vision.api_key_env} not set (vision={vision.name})")
        self._sequence = 0
        self._bbox_map: dict[str, list[int]] = {}
        self._panel_bbox_map: dict[str, list[int]] = {}
        self._cursor_anchor: CursorAnchor | None = None
        self._last_state_sha256: str | None = None

    # -- ComputerUseAdapter protocol ----------------------------------------

    def observe(self, *, session_id: str) -> SceneObservation:
        self._sequence += 1
        img_bytes = self._capture()
        # downsample the full screen before VLM: a 2560x1528 PNG is ~3-5MB base64 and
        # stalls 8B on the full-screen parse. bbox is normalized [0,1000] so resolution
        # does not affect coordinates. observe_focus crops first (already small), so it
        # skips this.
        from PIL import Image

        img = Image.open(io.BytesIO(img_bytes))
        maxw = 1600
        if img.width > maxw:
            img = img.resize((maxw, round(img.height * maxw / img.width)))
        vbuf = io.BytesIO()
        img.save(vbuf, format="PNG")
        parsed = self._parse(vbuf.getvalue())
        digest = hashlib.sha256(img_bytes).hexdigest()
        obs, bbox_map, panel_bbox_map = _build_scene(
            parsed,
            session_id=session_id,
            sequence=self._sequence,
            app_id=self.app_id,
            window_id=self.window_id,
            digest=digest,
            digest_mode=self.vision.digest_mode,
        )
        self._bbox_map = bbox_map
        self._panel_bbox_map = panel_bbox_map
        self._last_state_sha256 = obs.state_sha256
        return obs

    def _focus_core(
        self,
        *,
        session_id: str,
        region: tuple[int, int, int, int],
        img: Any,
        img_bytes: bytes,
    ) -> SceneObservation:
        """Crop ``img`` to ``region`` (full-screen pixel box), VLM-parse the crop, map bboxes
        back to full-screen normalized [0,1000]. Shared by region-based and panel_id-based
        focus. Caller stamps ``focus_root_panel_id`` once the panel is confirmed in the crop.
        """
        full_w, full_h = img.size
        crop = img.crop(region)
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        _dbg = os.environ.get("VISION_FOCUS_DEBUG")
        if _dbg:
            try:
                crop.save(_dbg)
            except Exception:
                pass
        parsed = self._parse(buf.getvalue())
        digest = hashlib.sha256(img_bytes).hexdigest()
        obs, bbox_map, panel_bbox_map = _build_scene(
            parsed,
            session_id=session_id,
            sequence=self._sequence,
            app_id=self.app_id,
            window_id=self.window_id,
            digest=digest,
            digest_mode=self.vision.digest_mode,
        )
        self._bbox_map = {eid: _map_crop_bbox(bb, region, full_w, full_h) for eid, bb in bbox_map.items()}
        self._panel_bbox_map = {pid: _map_crop_bbox(bb, region, full_w, full_h) for pid, bb in panel_bbox_map.items()}
        self._last_state_sha256 = obs.state_sha256
        return obs

    def _observe_focus_region(self, *, session_id: str, region: tuple[int, int, int, int]) -> SceneObservation:
        """Low-level: focus an explicit full-screen pixel region (no panel_id resolution).

        Retained for tests of the crop→coord mapping and as a primitive. Full-screen 8B
        misses small/transient regions (eg a right-click context menu); narrowing to a crop
        recovered 0 -> 23 menu items in the probe.
        """
        from PIL import Image

        self._sequence += 1
        img_bytes = self._capture()
        img = Image.open(io.BytesIO(img_bytes))
        return self._focus_core(session_id=session_id, region=region, img=img, img_bytes=img_bytes)

    def observe_focus(self, *, session_id: str, panel_id: str) -> SceneObservation:
        """LOCAL observation scoped to ``panel_id`` (FocusableComputerUseAdapter capability).

        Region resolution (缝 3/4): (1) ``_panel_bbox_map`` hit → crop that panel's region;
        (2) transient panel with a live ``CursorAnchor`` → crop around the click that opened
        it (one-shot, then consumed); (3) otherwise ``NoRegionForPanel``. The result carries
        ``focus_root_panel_id``. Coordinates stay adapter-internal; the contract never sees them.
        """
        from PIL import Image

        self._sequence += 1
        img_bytes = self._capture()
        img = Image.open(io.BytesIO(img_bytes))
        full_w, full_h = img.size
        bbox = self._panel_bbox_map.get(panel_id)
        if bbox is not None:
            region = _bbox_to_region(bbox, full_w, full_h)
            focused = self._focus_core(session_id=session_id, region=region, img=img, img_bytes=img_bytes)
            return self._stamp_focus_root(focused, panel_id)
        anchor = self._cursor_anchor
        if anchor is not None and not anchor.consumed and anchor.session_id == session_id:
            region = _anchor_region(anchor.click_xy, full_w, full_h)
            focused = self._focus_core(session_id=session_id, region=region, img=img, img_bytes=img_bytes)
            self._cursor_anchor = replace(anchor, consumed=True)
            return self._stamp_focus_root(focused, panel_id)
        if anchor is None:
            reason = "panel_not_found"
        elif anchor.consumed:
            reason = "cursor_anchor_stale"
        else:
            reason = "session_mismatch"
        raise NoRegionForPanel(
            panel_id=panel_id,
            reason=reason,
            source_state_sha256=self._last_state_sha256,
        )

    def _stamp_focus_root(self, focused: SceneObservation, panel_id: str) -> SceneObservation:
        """Mark ``focused`` with ``focus_root_panel_id`` only if the crop actually revealed
        the panel; otherwise the focus failed (the region did not contain it)."""
        if not any(p.panel_id == panel_id for p in focused.panels):
            raise NoRegionForPanel(
                panel_id=panel_id,
                reason="panel_not_found",
                source_state_sha256=self._last_state_sha256,
            )
        return replace(focused, focus_root_panel_id=panel_id)

    def execute(self, intent: ActionIntent) -> AdapterExecution:
        self._cursor_anchor = None  # 缝 4 rule ③: any execute invalidates a pending anchor
        act = intent.action
        if act == "wait":
            return AdapterExecution(True, "wait_completed")

        tid = intent.target_id
        bbox = self._bbox_map.get(tid) if tid else None
        if act in {"click", "hover", "type_text"}:
            if not tid or not bbox:
                return AdapterExecution(False, "target_not_found:no_bbox_in_viewport")

        try:
            if act == "press_key":
                self._press(intent.key or "end")
                return AdapterExecution(True, f"key_pressed:{intent.key}")
            if act == "click":
                x, y = self._denorm_center(bbox)
                self._record_anchor(intent, x, y)
                self._click(x, y, button=intent.button or "left")
                return AdapterExecution(True, f"clicked:{tid}")
            if act == "hover":
                x, y = self._denorm_center(bbox)
                self._move(x, y)
                return AdapterExecution(True, f"hovered:{tid}")
            if act == "type_text":
                x, y = self._denorm_center(bbox)
                self._click(x, y, button="left")
                self._type(intent.payload)
                return AdapterExecution(True, f"typed:{tid}")
        except Exception as exc:  # pyautogui failures must not crash the session
            return AdapterExecution(False, f"actuator_error:{exc}")
        return AdapterExecution(False, "unsupported_vision_action")

    def _record_anchor(self, intent: ActionIntent, x: int, y: int) -> None:
        """Record a cursor anchor for the transient a click may open (right-click → context menu)."""
        button = intent.button or "left"
        self._cursor_anchor = CursorAnchor(
            session_id=intent.session_id,
            click_xy=(x, y),
            expected_transient_kind="context_menu" if button == "right" else None,
            source_state_sha256=self._last_state_sha256,
        )

    # -- internals ----------------------------------------------------------

    def _capture(self) -> bytes:
        if self._screenshot is not None:
            return self._screenshot()
        import pyautogui

        img = pyautogui.screenshot()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _parse(self, img_bytes: bytes) -> dict:
        mime = "image/png" if img_bytes[:4] == b"\x89PNG" else "image/jpeg"
        data_url = f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"
        payload = {
            "model": self.vision.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": PARSE_PROMPT},
                    ],
                }
            ],
            "max_tokens": self.vision.max_tokens,
            "temperature": self.vision.temperature,
        }
        url = self.vision.base_url + "/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        resp = None
        last_err: Exception | None = None
        for _attempt in range(2):  # one retry on timeout / transient non-200
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=300)
                if resp.status_code == 200:
                    break
                last_err = RuntimeError(f"VLM HTTP {resp.status_code}: {resp.text[:300]}")
            except requests.RequestException as exc:
                last_err = exc
        if resp is None or resp.status_code != 200:
            raise last_err or RuntimeError("VLM request failed")
        text = resp.json()["choices"][0]["message"]["content"]
        if isinstance(text, list):
            text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise RuntimeError(f"VLM returned no JSON: {text[:200]}")
        return _parse_vlm_json(m.group(0))

    def _denorm_center(self, bbox: list[int]) -> tuple[int, int]:
        """Normalized [0,1000] bbox -> screen pixel center (assumes screenshot == full screen)."""
        import pyautogui

        sw, sh = pyautogui.size()
        cx = (bbox[0] + bbox[2]) / 2 / 1000 * sw
        cy = (bbox[1] + bbox[3]) / 2 / 1000 * sh
        return int(cx), int(cy)

    def _click(self, x: int, y: int, *, button: str) -> None:
        import pyautogui

        pyautogui.click(x, y, button=button)

    def _move(self, x: int, y: int) -> None:
        import pyautogui

        pyautogui.moveTo(x, y)

    def _type(self, text: str) -> None:
        import pyautogui

        pyautogui.write(text)

    def _press(self, key: str) -> None:
        import pyautogui

        pyautogui.press(key)


def _parse_vlm_json(raw: str) -> dict:
    """Lenient VLM-JSON parse. VLMs occasionally emit trailing commas or an unescaped
    char inside a label; we degrade gracefully rather than crash the whole observation.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    repaired = re.sub(r",\s*([}\]])", r"\1", raw)  # drop trailing commas
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    extracted = _lenient_extract_arrays(repaired)  # brace-match, skip broken objects
    if extracted.get("panels") or extracted.get("elements"):
        return extracted
    raise RuntimeError("VLM JSON unparseable (no valid panels/elements)")


def _lenient_extract_arrays(raw: str) -> dict:
    """Brace-match top-level objects inside the panels/elements arrays; skip any object
    that fails to json.loads. Survives a bad label inside one element.
    """
    out: dict[str, list] = {"panels": [], "elements": []}
    for key in ("panels", "elements"):
        m = re.search(rf'"{key}"\s*:\s*\[', raw)
        if not m:
            continue
        i, depth, obj_start = m.end(), 0, None
        in_str = False
        objs: list[str] = []
        while i < len(raw):
            c = raw[i]
            if c == '"':
                in_str = not in_str
            elif not in_str:
                if c == "{":
                    if depth == 0:
                        obj_start = i
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0 and obj_start is not None:
                        objs.append(raw[obj_start : i + 1])
                        obj_start = None
                elif c == "]" and depth == 0:
                    break
            i += 1
        for o in objs:
            try:
                out[key].append(json.loads(o))
            except json.JSONDecodeError:
                continue
    return out
