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
from dataclasses import dataclass
from typing import Any

import requests

from .adapter import AdapterExecution
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


def _build_scene(
    parsed: dict, *, session_id: str, sequence: int, app_id: str, window_id: str, digest: str
) -> tuple[SceneObservation, dict[str, list[int]], dict[str, list[int]]]:
    """Map VLM panels/elements -> (SceneObservation, element_id->bbox, panel_id->bbox).

    bbox stays out of the contract (returned separately for execute() and panel viz).
    """
    pid_by_key: dict[str, str] = {}
    panels: list[Panel] = []
    panel_bbox_map: dict[str, list[int]] = {}
    for p in parsed.get("panels", []):
        role = _norm(p.get("role")) or "other"
        name = str(p.get("name") or p.get("role") or "panel")
        pid = role if role != "other" else _norm(name)
        if any(pp.panel_id == pid for pp in panels):
            continue
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
        state_sha256=digest,
        elements=tuple(elements),
        panels=tuple(panels),
    )
    return obs, bbox_map, panel_bbox_map


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
        )
        self._bbox_map = bbox_map
        self._panel_bbox_map = panel_bbox_map
        return obs

    def observe_focus(self, *, session_id: str, region: tuple[int, int, int, int]) -> SceneObservation:
        """Crop the screen to ``region`` (full-screen pixel box x1,y1,x2,y2), run the VLM on
        the CROP only, and map element bboxes back to full-screen normalized [0,1000] coords.

        Full-screen 8B misses small/transient regions (eg a right-click context menu overlaid
        on Hierarchy) — the probe showed 0 -> 23 menu items when narrowed. Bboxes are mapped
        back to full-screen so ``execute()`` clicks the correct pixel without changes.
        """
        self._sequence += 1
        img_bytes = self._capture()
        from PIL import Image

        img = Image.open(io.BytesIO(img_bytes))
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
        )
        self._bbox_map = {eid: _map_crop_bbox(bb, region, full_w, full_h) for eid, bb in bbox_map.items()}
        self._panel_bbox_map = {pid: _map_crop_bbox(bb, region, full_w, full_h) for pid, bb in panel_bbox_map.items()}
        return obs

    def execute(self, intent: ActionIntent) -> AdapterExecution:
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
