"""Unity-shaped deterministic mock adapter for computer-use perception/locator tests.

Models a Unity-editor-like layout (Hierarchy / Scene / Game / Toolbar / Inspector
plus transient context-menu / submenu / popup) as an in-process state machine. It
exercises the REAL Unity interaction contract (right-click opens the context menu,
hover opens a submenu) rather than faking it with left-clicks, and guards
interactability: clicking a below-fold / invisible / disabled element is rejected
by the adapter itself (defense in depth, not just perception's job).

Scroll is the adapter's internal concern: ``press_key "end"`` flips
``inspector_scroll`` and the Add Component button's ``below_fold`` flag follows it.
The main brain only ever says "Inspector bottom"; the TargetingController handles
the reveal.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

from .adapter import AdapterExecution
from .contracts import ActionIntent, FrameRef, Panel, SceneObservation, UIElement


class UnityEditorMockAdapter:
    app_id = "unity.editor"

    def __init__(self) -> None:
        self._sequence = 0
        self._spawn_seq = 0
        self._object_counter = 0
        self._state: dict[str, Any] = {
            "scene_objects": [],
            "selected_object_id": None,
            "context_menu_open": False,
            "submenu_open": False,
            "add_component_popup_open": False,
            "component_search": "",
            "inspector_scroll": "top",
            "play_mode": False,
        }
        self.execution_count = 0

    # ------------------------------------------------------------------ observe

    def observe(self, *, session_id: str) -> SceneObservation:
        self._sequence += 1
        digest = self._state_digest()
        panels = self._panels()
        elements = self._elements(panels)
        return SceneObservation(
            session_id=session_id,
            sequence=self._sequence,
            app_id=self.app_id,
            window_id="unity-main",
            window_title="Unity Editor",
            frame=FrameRef(
                frame_id=f"frame-{digest[:16]}",
                sha256=digest,
                width=1920,
                height=1080,
                artifact_uri=f"mock://unity/{digest}",
                sensitivity="private",
            ),
            state_sha256=digest,
            elements=elements,
            panels=panels,
        )

    def observe_focus(self, *, session_id: str, panel_id: str) -> SceneObservation:
        """LOCAL observation scoped to ``panel_id`` + its descendants (logical tree crop).

        The mock's full observation carries the complete panel tree, so focus selects the
        subtree via ``SceneObservation.focused_view`` and stamps a fresh sequence. Same
        logical scope as VisionAdapter's pixel crop: focus root + visible descendants,
        root ``parent_panel_id`` normalized to None (self-contained), ancestors in
        ``focus_ancestor_path``. No pixel cropping — the mock is full-fidelity.
        """
        full = self.observe(session_id=session_id)
        focused = full.focused_view(panel_id)
        self._sequence += 1
        return replace(focused, sequence=self._sequence)

    # ----------------------------------------------------------------- execute

    def execute(self, intent: ActionIntent) -> AdapterExecution:
        self.execution_count += 1
        # Interactability guard: a targeted action must land on an interactable element.
        if intent.target_id is not None and intent.action in {"click", "type_text", "hover"}:
            target = self._find_current_element(intent.target_id)
            if target is None:
                return AdapterExecution(False, "target_not_found")
            if not target.visible:
                return AdapterExecution(False, "target_not_interactable:hidden")
            if not target.enabled:
                return AdapterExecution(False, "target_not_interactable:disabled")
            if _elem_below_fold(target):
                return AdapterExecution(False, "target_not_interactable:below_fold")

        act = intent.action
        tid = intent.target_id

        if act == "click" and tid == "hierarchy-background":
            if intent.button != "right":
                return AdapterExecution(False, "wrong_button:context_menu_requires_right_click")
            self._state["context_menu_open"] = True
            self._open_transient()
            return AdapterExecution(True, "context_menu_opened")

        if act == "hover" and tid == "menu-3d-object":
            if not self._state["context_menu_open"]:
                return AdapterExecution(False, "menu_not_open")
            self._state["submenu_open"] = True
            self._open_transient()
            return AdapterExecution(True, "submenu_opened")

        if act == "click" and tid == "menu-3d-object":
            return AdapterExecution(False, "wrong_action:submenu_requires_hover")

        if act == "click" and tid == "submenu-cube":
            if not self._state["submenu_open"]:
                return AdapterExecution(False, "submenu_not_open")
            self._create_object(name="Cube", obj_type="cube")
            self._close_menus()
            return AdapterExecution(True, "cube_created")

        if act == "press_key" and intent.key.casefold() in {"end", "page_down", "pgdn"}:
            self._state["inspector_scroll"] = "bottom"
            return AdapterExecution(True, "scrolled_to_bottom")

        if act == "click" and tid == "inspector-add-component":
            if self._state["selected_object_id"] is None:
                return AdapterExecution(False, "no_selection")
            self._state["add_component_popup_open"] = True
            self._open_transient()
            return AdapterExecution(True, "add_component_popup_opened")

        if act == "type_text" and tid == "acp-search":
            if not self._state["add_component_popup_open"]:
                return AdapterExecution(False, "popup_not_open")
            self._state["component_search"] = intent.payload
            return AdapterExecution(True, "search_entered")

        if act == "click" and tid == "acp-result-rigidbody":
            if not self._state["add_component_popup_open"]:
                return AdapterExecution(False, "popup_not_open")
            if "rigidbody" not in self._state["component_search"].casefold():
                return AdapterExecution(False, "result_not_visible")
            self._add_component_to_selected("Rigidbody")
            self._state["add_component_popup_open"] = False
            self._state["component_search"] = ""
            return AdapterExecution(True, "rigidbody_added")

        if act == "click" and tid == "toolbar-play":
            self._state["play_mode"] = not self._state["play_mode"]
            return AdapterExecution(True, "play_toggled")

        if act == "wait":
            return AdapterExecution(True, "wait_completed")

        if act == "hover":
            return AdapterExecution(True, "hover_acknowledged")

        return AdapterExecution(False, "unsupported_unity_action")

    # ------------------------------------------------------------- test helpers

    def mutate_out_of_band(self, **changes: Any) -> None:
        """Simulate external state changes between planning and execution (stale tests)."""
        unknown = set(changes) - set(self._state)
        if unknown:
            raise ValueError(f"unknown mock state key(s): {sorted(unknown)}")
        self._state.update(changes)

    # --------------------------------------------------------------- internals

    def _state_digest(self) -> str:
        encoded = json.dumps(self._state, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _open_transient(self) -> None:
        self._spawn_seq += 1

    def _close_menus(self) -> None:
        self._state["context_menu_open"] = False
        self._state["submenu_open"] = False

    def _create_object(self, *, name: str, obj_type: str) -> None:
        self._object_counter += 1
        obj_id = f"{obj_type}-{self._object_counter}"
        # every GameObject carries a Transform (real Unity invariant); it nests a Position
        # sub-panel so the inspector tree reaches depth 3 (缝 6).
        self._state["scene_objects"].append({"id": obj_id, "name": name, "type": obj_type, "components": ["Transform"]})
        self._state["selected_object_id"] = obj_id

    def _add_component_to_selected(self, component: str) -> None:
        selected = self._state["selected_object_id"]
        for obj in self._state["scene_objects"]:
            if obj["id"] == selected:
                obj["components"].append(component)
                return

    def _selected_object(self) -> dict[str, Any] | None:
        selected = self._state["selected_object_id"]
        if selected is None:
            return None
        return next((o for o in self._state["scene_objects"] if o["id"] == selected), None)

    def _panels(self) -> tuple[Panel, ...]:
        panels: list[Panel] = [
            Panel(panel_id="hierarchy", role="hierarchy", label="Hierarchy", ordinal="leftmost"),
            Panel(panel_id="scene", role="scene", label="Scene", ordinal="middle_left"),
            Panel(panel_id="game", role="game", label="Game", ordinal="top_center"),
            Panel(panel_id="toolbar", role="toolbar", label="Toolbar", ordinal="top"),
            Panel(panel_id="inspector", role="inspector", label="Inspector", ordinal="rightmost"),
        ]
        # selected object's components are nested sub-panels of the inspector (缝 6);
        # Transform further nests a Position sub-panel so the tree reaches depth 3.
        # Transients below remain overlays (parent_panel_id=None; 缝 7) — components model
        # persistent structure, not transient geometry.
        selected = self._selected_object()
        if selected is not None:
            for component in selected["components"]:
                cid = f"inspector-component-{str(component).casefold()}"
                panels.append(
                    Panel(
                        panel_id=cid,
                        role="inspector_component",
                        label=component,
                        parent_panel_id="inspector",
                    )
                )
                if str(component).casefold() == "transform":
                    panels.append(
                        Panel(
                            panel_id=f"{cid}-position",
                            role="inspector_component",
                            label="Position",
                            parent_panel_id=cid,
                        )
                    )
        if self._state["context_menu_open"]:
            panels.append(
                Panel(
                    panel_id="context_menu",
                    role="menu",
                    label="Context Menu",
                    transient_kind="context_menu",
                    ordinal=None,
                    spawned_by_element_id="hierarchy-background",
                    spawn_sequence=self._spawn_seq,
                )
            )
        if self._state["submenu_open"]:
            panels.append(
                Panel(
                    panel_id="create_submenu",
                    role="menu",
                    label="Create Submenu",
                    transient_kind="submenu",
                    spawned_by_element_id="menu-3d-object",
                    spawn_sequence=self._spawn_seq,
                )
            )
        if self._state["add_component_popup_open"]:
            panels.append(
                Panel(
                    panel_id="add_component_popup",
                    role="popup",
                    label="Add Component Popup",
                    transient_kind="popup",
                    spawned_by_element_id="inspector-add-component",
                    spawn_sequence=self._spawn_seq,
                )
            )
        return tuple(panels)

    def _elements(self, panels: tuple[Panel, ...]) -> tuple[UIElement, ...]:
        below_fold = self._state["inspector_scroll"] == "top"
        elements: list[UIElement] = [
            UIElement(
                element_id="hierarchy-background",
                role="canvas",
                label="Hierarchy blank",
                panel_id="hierarchy",
            ),
            UIElement(
                element_id="scene-background",
                role="canvas",
                label="Scene",
                panel_id="scene",
            ),
            UIElement(
                element_id="game-view",
                role="canvas",
                label="Game",
                panel_id="game",
            ),
            UIElement(
                element_id="toolbar-play",
                role="button",
                label="Play",
                panel_id="toolbar",
                attributes=(("icon_shape", "play"),),
            ),
        ]
        for obj in self._state["scene_objects"]:
            elements.append(
                UIElement(
                    element_id=f"hierarchy-item-{obj['id']}",
                    role="tree_item",
                    label=obj["name"],
                    panel_id="hierarchy",
                    attributes=(("selected", obj["id"] == self._state["selected_object_id"]),),
                )
            )
        selected = self._selected_object()
        if selected is not None:
            elements.append(
                UIElement(
                    element_id=f"inspector-header-{selected['id']}",
                    role="header",
                    label=selected["name"],
                    panel_id="inspector",
                )
            )
            # component foldout headers hang under their component sub-panels (缝 6); the
            # header stays an interactable element so the component remains a resolvable
            # target, while the structural nesting lives in the panel tree. Transform's
            # Position sub-panel carries X/Y/Z number fields (depth 3).
            for component in selected["components"]:
                cid = f"inspector-component-{str(component).casefold()}"
                elements.append(
                    UIElement(
                        element_id=f"{cid}-header",
                        role="component",
                        label=component,
                        panel_id=cid,
                    )
                )
                if str(component).casefold() == "transform":
                    position_id = f"{cid}-position"
                    for axis in ("X", "Y", "Z"):
                        elements.append(
                            UIElement(
                                element_id=f"{position_id}-{axis.casefold()}",
                                role="number_field",
                                label=axis,
                                panel_id=position_id,
                                attributes=(("value", 0.0),),
                            )
                        )
            elements.append(
                UIElement(
                    element_id="inspector-add-component",
                    role="button",
                    label="Add Component",
                    panel_id="inspector",
                    semantic_band="bottom",
                    attributes=(("below_fold", below_fold),),
                )
            )
        if self._state["context_menu_open"]:
            elements.append(
                UIElement(
                    element_id="menu-3d-object",
                    role="menu_item",
                    label="3D Object",
                    panel_id="context_menu",
                    attributes=(("has_submenu", True),),
                )
            )
            elements.append(
                UIElement(
                    element_id="menu-create-empty",
                    role="menu_item",
                    label="Create Empty",
                    panel_id="context_menu",
                )
            )
        if self._state["submenu_open"]:
            elements.append(
                UIElement(
                    element_id="submenu-cube",
                    role="menu_item",
                    label="Cube",
                    panel_id="create_submenu",
                )
            )
            elements.append(
                UIElement(
                    element_id="submenu-sphere",
                    role="menu_item",
                    label="Sphere",
                    panel_id="create_submenu",
                )
            )
        if self._state["add_component_popup_open"]:
            search = self._state["component_search"]
            elements.append(
                UIElement(
                    element_id="acp-search",
                    role="text_input",
                    label="Search",
                    panel_id="add_component_popup",
                    attributes=(
                        ("icon_shape", "question_mark"),
                        ("shape", "long_bar"),
                        ("value", search),
                    ),
                )
            )
            result_visible = "rigidbody" in search.casefold()
            elements.append(
                UIElement(
                    element_id="acp-result-rigidbody",
                    role="list_item",
                    label="Rigidbody",
                    panel_id="add_component_popup",
                    visible=result_visible,
                )
            )
        return tuple(elements)

    def _find_current_element(self, element_id: str) -> UIElement | None:
        panels = self._panels()
        for element in self._elements(panels):
            if element.element_id == element_id:
                return element
        return None


def _elem_below_fold(element: UIElement) -> bool:
    return bool(element.attribute_map().get("below_fold", False))
