"""ComputerUseBridge: thin tool-bridge connecting run_agent to the computer-use stack.

Façade over ComputerUseSession + PerceptionRegion + TargetingController + adapter. Owns the
model-facing concerns (scene-text dump, decision-JSON parsing, transient-focus merge). The
main brain learns 1-2 tools (act_cu / focus_cu); it never sees coordinates or raw ids.

Core invariant (G plan S2 / GPT #2): the bridge holds exactly ONE current observation
``_current``. ``act()`` resolves against ``_current`` (no observe); ``session.perform``'s
post-execute observe (BoundFreshness: 0 pre + 1 post) is the only VLM call per act — the
pilot's double-VLM bug is structurally impossible. Transient focus (right-click context_menu
/ hover submenu) is an EXTRA focus-crop observe, counted separately, and only fires when the
full-screen observe missed the transient (real VisionAdapter; mock adapters see everything).
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

from .contracts import SceneObservation
from .locator import Locator


class CompositeMergeError(RuntimeError):
    """Raised when _merge_transient_focus detects an id collision between the base and focus
    observations — the adapter violated the cross-observation id-namespace assumption. Never
    silently rewritten (that would break adapter.execute's _bbox_map lookup). (GPT #8)"""


class ComputerUseBridge:
    """The single model-facing surface over the computer-use stack.

    Construct with a BoundFreshness session (``freshness=BoundFreshness(...)`` or
    ``ttl_ms=...``) so perform() binds the intent to _latest instead of re-observing. The
    bridge owns ``_current`` — the one observation the model sees and act() resolves against.
    """

    def __init__(self, *, session, perception, targeting, event_sink=None) -> None:
        self._session = session
        self._perception = perception
        self._targeting = targeting
        self._adapter = session.adapter
        self._event_sink = event_sink or (lambda *a, **k: None)
        self._current: SceneObservation | None = None
        self._last_action: tuple[str, str, str | None, tuple[int, int] | None] | None = None

    @property
    def current(self) -> SceneObservation | None:
        return self._current

    # -- model-facing ops (prime / act / focus / wait) ----------------------

    def prime(self) -> str:
        """Seed _current with a plain full-screen observe (no prior action → no transient
        focus). Returns scene text for run_agent's initial_observation."""
        self._current = self._session.observe()
        self._last_action = None
        return self.dump_scene(self._current)

    def act(self, decision: dict, *, step: int) -> dict:
        """Resolve the decision's locator against _current (no observe), perform via the
        BoundFreshness session (0 pre + 1 post observe), then optionally transient-focus the
        region this action opened. ``wait`` / ``done`` are handled inline."""
        action = str(decision.get("action", "")).strip().casefold()
        if action == "wait":
            return self.wait()
        if action == "done":
            return {"status": "done", "observation": self.dump_scene(self._current)}
        if self._current is None:
            self._current = self._session.observe()
        try:
            locator = Locator.from_dict(decision.get("locator") or {})
        except Exception as exc:
            return {
                "status": "unresolved",
                "resolve_status": "locator_parse_error",
                "reason": str(exc),
                "observation": self.dump_scene(self._current),
            }
        result = self._perception.resolve(locator, self._current)
        if result.status != "resolved" or result.first is None:
            return {
                "status": "unresolved",
                "resolve_status": result.status,
                "reason": result.reason,
                "observation": self.dump_scene(self._current),
            }
        target = result.first
        button = str(decision.get("button") or "left").strip().casefold()
        intent = target.to_action_intent(
            action=action,
            intent_id=f"cu-{step}",
            session_id=self._session.session_id,
            app_id=self._adapter.app_id,
            button=button,
            payload=str(decision.get("text") or ""),
            key=str(decision.get("key") or ""),
        )
        receipt = self._session.perform(intent)  # BoundFreshness: 0 pre + 1 post observe
        after = self._session._latest
        if receipt.status == "executed":
            click_xy = self._target_click_xy(target.element_id)
            self._last_action = (action, button, target.panel_id, click_xy)
            self._current = self._maybe_transient_focus(after)
        else:
            self._current = after
        return {
            "status": receipt.status,
            "reason": receipt.reason,
            "state_changed": receipt.state_changed,
            "observation": self.dump_scene(self._current),
        }

    def wait(self) -> dict:
        """Refresh _current without acting. ``wait`` is a bridge-level refresh — never an
        ActionIntent, never touches adapter.execute / freshness gate / receipt (GPT #3)."""
        self._current = self._session.observe()
        return {"status": "waited", "observation": self.dump_scene(self._current)}

    def focus(self, anchors, *, step: int, budget=None) -> dict:
        """Explicit deep focus_chain (Inspector → Transform → ...). Returns status + depth so
        a partial focus never masquerades as success (GPT #13)."""
        ordered = tuple(anchors)
        result = self._targeting.focus_chain(ordered, budget=budget)
        self._current = result.observation
        return {
            "status": result.outcome,
            "completed_depth": result.depth,
            "requested_depth": len(ordered),
            "reason": result.reason,
            "reveals_used": result.reveals_used,
            "observations_used": result.observations_used,
            "observation": self.dump_scene(result.observation) if result.observation is not None else "",
        }

    # -- transient focus + merge (real VisionAdapter path; mock sees everything) ----

    def _target_click_xy(self, element_id: str) -> tuple[int, int] | None:
        tbox = self._adapter._bbox_map.get(element_id) if hasattr(self._adapter, "_bbox_map") else None
        denorm = getattr(self._adapter, "_denorm_center", None)
        if tbox and callable(denorm):
            try:
                xy = denorm(tbox)
                return (int(xy[0]), int(xy[1])) if xy else None
            except Exception:
                return None
        return None

    def _maybe_transient_focus(self, after: SceneObservation) -> SceneObservation:
        """After an action that opens a transient the full-screen observe may miss (8B),
        re-focus the region and merge. Mock adapters return the full state (transient
        present) → returns ``after`` unchanged (no focus, no merge)."""
        if self._last_action is None or not hasattr(self._adapter, "observe_focus"):
            return after
        action, button, panel_id, click_xy = self._last_action
        if click_xy is None:
            return after
        mx, my = click_xy
        if (
            action == "click"
            and button == "right"
            and not any(p.transient_kind == "context_menu" for p in after.panels)
        ):
            region = (max(0, mx - 30), max(0, my - 30), mx + 360, my + 760)
            return self._focus_and_merge(after, region, transient_panel_id="context_menu")
        if action == "hover" and panel_id in ("popup", "context_menu", "menu", "submenu"):
            region = (mx + 10, max(0, my - 80), mx + 430, my + 580)
            try:
                return self._adapter.observe_focus(session_id=self._session.session_id, region=region)
            except Exception:
                return after
        return after

    def _focus_and_merge(
        self, base: SceneObservation, region: tuple[int, int, int, int], *, transient_panel_id: str
    ) -> SceneObservation:
        bbox_snap = dict(self._adapter._bbox_map) if hasattr(self._adapter, "_bbox_map") else {}
        try:
            focused = self._adapter.observe_focus(session_id=self._session.session_id, region=region)
        except Exception:
            return base
        bbox_f = dict(self._adapter._bbox_map) if hasattr(self._adapter, "_bbox_map") else {}
        merged = self._merge_transient_focus(base, focused, transient_panel_id=transient_panel_id)
        cm_eids = {e.element_id for e in focused.elements if e.panel_id == transient_panel_id}
        if hasattr(self._adapter, "_bbox_map"):
            self._adapter._bbox_map = {**bbox_snap, **{k: v for k, v in bbox_f.items() if k in cm_eids}}
        return merged

    def _merge_transient_focus(
        self, base: SceneObservation, focused: SceneObservation, *, transient_panel_id: str
    ) -> SceneObservation:
        """Merge the transient subtree from ``focused`` into ``base`` (pilot observe_effective
        semantics). The transient is an independent panel base lacks → replace/insert it.
        DOES NOT rewrite ids (GPT #8); collision → CompositeMergeError."""
        base_eids = {e.element_id for e in base.elements}
        base_pids = {p.panel_id for p in base.panels}
        t_elems = tuple(e for e in focused.elements if e.panel_id == transient_panel_id)
        t_panels = tuple(p for p in focused.panels if p.panel_id == transient_panel_id)
        for e in t_elems:
            if e.element_id in base_eids:
                raise CompositeMergeError(f"element_id {e.element_id!r} collides between base and focus observations")
        for p in t_panels:
            if p.panel_id in base_pids:
                raise CompositeMergeError(f"panel_id {p.panel_id!r} collides between base and focus observations")
        merged_elements = tuple(e for e in base.elements if e.panel_id != transient_panel_id) + t_elems
        merged_panels = tuple(p for p in base.panels if p.panel_id != transient_panel_id) + t_panels
        digest = self._composite_digest(base, focused)
        return replace(
            base,
            elements=merged_elements,
            panels=merged_panels,
            state_sha256=digest,
            observation_kind="composite",
            source_frame_ids=(base.frame.frame_id, focused.frame.frame_id),
        )

    def _composite_digest(self, base: SceneObservation, focused: SceneObservation) -> str:
        """Canonical digest over the merged structure (base semantic + focused semantic +
        merge-policy version). NOT comparable to a later raw observation's state (composite
        is not a single screenshot)."""
        from .vision_adapter import _semantic_state_digest

        return hashlib.sha256(
            "|".join(
                [
                    _semantic_state_digest(list(base.elements), list(base.panels)),
                    _semantic_state_digest(list(focused.elements), list(focused.panels)),
                    "merge-v1",
                ]
            ).encode("utf-8")
        ).hexdigest()

    # -- scene text (model-facing; no element_ids, no coordinates) ----------

    @staticmethod
    def _avail_blocker(element) -> tuple[bool, str | None]:
        if not element.visible:
            return False, "hidden"
        if not element.enabled:
            return False, "disabled"
        if bool((element.attribute_map() or {}).get("below_fold", False)):
            return False, "below_fold"
        return True, None

    def dump_scene(self, obs: SceneObservation | None) -> str:
        """Compact structured scene text. Panels + elements (role/label/available/blocker).
        Omits element_ids and coordinates — the model describes targets via Locators."""
        if obs is None:
            return "(no observation yet)"
        lines: list[str] = []
        transients = [p for p in obs.panels if p.transient_kind]
        persistent = [p for p in obs.panels if not p.transient_kind]
        kind = " kind=composite" if obs.observation_kind == "composite" else ""
        lines.append(
            f"SCENE (frame={obs.frame.frame_id[:24]} panels={len(obs.panels)} "
            f"transients={len(transients)} elements={len(obs.elements)}){kind}"
        )
        lines.append("PANELS:")
        for p in persistent:
            lines.append(f"  - role={p.role} label={p.label!r} elements={len(obs.elements_in(p.panel_id))}")
        for p in transients:
            parent = obs.element(p.spawned_by_element_id) if p.spawned_by_element_id else None
            parent_lbl = f" (spawned by {parent.label!r})" if parent else ""
            lines.append(
                f"  * TRANSIENT role={p.role} transient_kind={p.transient_kind}{parent_lbl}"
                f"  [target via transient_kind, NOT panel_name]"
                f" elements={len(obs.elements_in(p.panel_id))}"
            )
        lines.append("ELEMENTS by panel:")
        for p in obs.panels:
            elems = list(obs.elements_in(p.panel_id))
            if not elems:
                continue
            tag = p.transient_kind or p.role
            lines.append(f"  [{tag}]")
            for e in elems:
                avail, blocker = self._avail_blocker(e)
                amap = e.attribute_map() or {}
                extra = [
                    f"{k}={amap[k]}" for k in ("icon_shape", "shape", "has_submenu", "selected", "value") if k in amap
                ]
                if getattr(e, "semantic_band", None):
                    extra.append(f"semantic_band={e.semantic_band}")
                tail = f" BLOCKED({blocker})" if not avail else ""
                ex = f"  [{', '.join(extra)}]" if extra else ""
                lines.append(f"    - {e.role} {e.label!r}{ex}{tail}")
        return "\n".join(lines)
