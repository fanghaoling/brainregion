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
    {"name": "<panel label as shown>", "role": "hierarchy|inspector|scene|game|toolbar|project|console|menu_bar|other", "bbox": [x1,y1,x2,y2]}
  ],
  "elements": [
    {"panel": "<panel name or role>", "role": "button|menu_item|input|tab|list_item|tree_item|dropdown|slider|header|component|other", "label": "<visible text/label>", "bbox": [x1,y1,x2,y2], "interactable": true}
  ]
}
Rules:
- Coordinates are NORMALIZED to [0,1000], origin top-left, bbox = [x1,y1,x2,y2].
- First identify the main persistent panel regions (Hierarchy usually left, Inspector \
right, Scene viewport center, Game, Toolbar top, Project/Console bottom, menu bar top).
- Then list INTERACTABLE UI elements inside each panel: buttons, menu items, tabs, input \
fields, list/tree items, dropdowns, sliders, headers. Include their visible label/text.
- Do NOT treat 3D objects inside the Scene/Game viewport as UI elements — the viewport is \
just one panel. (But DO list Hierarchy tree items, which represent those objects.)
"""


# ----------------------------------------------------------------- helpers


def _norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def _build_scene(
    parsed: dict, *, session_id: str, sequence: int, app_id: str, window_id: str
) -> tuple[SceneObservation, dict[str, list[int]]]:
    """Map VLM panels/elements -> (SceneObservation, element_id->bbox).

    bbox stays out of the contract (returned separately for the adapter's execute()).
    """
    pid_by_key: dict[str, str] = {}
    panels: list[Panel] = []
    for p in parsed.get("panels", []):
        role = _norm(p.get("role")) or "other"
        name = str(p.get("name") or p.get("role") or "panel")
        pid = role if role != "other" else _norm(name)
        if any(pp.panel_id == pid for pp in panels):
            continue
        panels.append(Panel(panel_id=pid, role=role, label=name))
        pid_by_key[_norm(name)] = pid
        pid_by_key[role] = pid

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
            elements.append(
                UIElement(
                    element_id=eid,
                    role=role,
                    label=label,
                    panel_id=pid,
                    enabled=bool(e.get("interactable", True)),
                    visible=True,
                )
            )
            bbox = e.get("bbox") or []
            if len(bbox) == 4:
                bbox_map[eid] = [int(v) for v in bbox]

    digest = hashlib.sha256(json.dumps(parsed, sort_keys=True).encode()).hexdigest()
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
    return obs, bbox_map


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

    # -- ComputerUseAdapter protocol ----------------------------------------

    def observe(self, *, session_id: str) -> SceneObservation:
        self._sequence += 1
        img_bytes = self._capture()
        parsed = self._parse(img_bytes)
        obs, bbox_map = _build_scene(
            parsed,
            session_id=session_id,
            sequence=self._sequence,
            app_id=self.app_id,
            window_id=self.window_id,
        )
        self._bbox_map = bbox_map
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
