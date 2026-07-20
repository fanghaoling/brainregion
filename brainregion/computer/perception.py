"""Algorithmic perception layer (mock stage, no LLM).

Mirrors the no-LLM region pattern: ``__init__`` + sync consult methods that take a
``SceneObservation`` and return processed dicts / a ``ResolutionResult``. The main
brain consults on demand (survey / focus / find / describe / resolve) and receives
processed structure + positions, never raw pixels or absolute coordinates.

``resolve`` is the single source of truth for locator resolution; it returns a
``ResolutionResult`` whose ``ResolvedTarget`` candidates carry the observation's
``frame_id``/``state_sha256`` so downstream intents are freshness-bound by construction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from brainregion.runtime import emit_event

from .contracts import SceneObservation, UIElement
from .locator import (
    ElementDescriptor,
    Locator,
    PanelAnchor,
    ResolvedTarget,
    ResolutionStep,
    WithinPanel,
)


EventSink = Callable[..., object]

ResolveStatus = Literal["resolved", "blocked", "ambiguous", "not_found", "unsupported"]


@dataclass(frozen=True)
class ResolutionResult:
    """Outcome of resolving a Locator against one observation.

    ``status`` is a single enum (forbids invalid bool combinations like
    ambiguous∧not_found). ``candidates`` are ``ResolvedTarget`` bound to the
    observation ``resolve`` was called with.
    """

    status: ResolveStatus
    candidates: tuple[ResolvedTarget, ...] = ()
    trace: tuple[ResolutionStep, ...] = ()
    not_found_candidates: tuple[dict[str, Any], ...] = ()
    reason: str = ""

    @property
    def first(self) -> ResolvedTarget | None:
        return self.candidates[0] if self.candidates else None


def _band_of(index: int, length: int) -> str | None:
    """Map an element's position in a panel's ordered list to a 5-zone band.

    top ⇒ index 0; bottom ⇒ last index (the Add Component invariant). Interior
    indices split into near_top / middle / middle_bottom by ratio.
    """
    if length <= 0 or index < 0 or index >= length:
        return None
    if index == 0:
        return "top"
    if index == length - 1:
        return "bottom"
    interior_len = length - 2
    if interior_len <= 0:
        return "middle"
    ratio = (index - 1) / interior_len
    if ratio < 1 / 3:
        return "near_top"
    if ratio < 2 / 3:
        return "middle"
    return "middle_bottom"


def _element_band(element: UIElement, index: int, length: int) -> str | None:
    if element.semantic_band is not None:
        return element.semantic_band
    return _band_of(index, length)


def _below_fold(element: UIElement) -> bool:
    return bool(element.attribute_map().get("below_fold", False))


def _blocker_of(element: UIElement) -> tuple[bool, str | None]:
    """Return (available, blocker). below_fold/disabled/hidden each map to a blocker."""
    if not element.visible:
        return False, "hidden"
    if not element.enabled:
        return False, "disabled"
    if _below_fold(element):
        return False, "below_fold"
    return True, None


def _bound(element: UIElement, obs: SceneObservation, trace: tuple[ResolutionStep, ...]) -> ResolvedTarget:
    available, blocker = _blocker_of(element)
    return ResolvedTarget.from_observation(
        element=element,
        frame_id=obs.frame.frame_id,
        state_sha256=obs.state_sha256,
        available=available,
        blocker=blocker,
        trace=trace,
    )


def _element_summary(element: UIElement) -> dict[str, Any]:
    return {
        "element_id": element.element_id,
        "panel_id": element.panel_id,
        "role": element.role,
        "label": element.label,
    }


def _match_panels(anchor: PanelAnchor, obs: SceneObservation) -> tuple[list[ResolutionStep], list]:
    """Resolve the anchor to candidate panels. AND across all set conditions."""
    steps: list[ResolutionStep] = []
    panels = list(obs.panels)

    if anchor.panel_name is not None:
        want = anchor.panel_name
        panels = [p for p in panels if want in (str(p.label).casefold().strip(), p.role)]
        steps.append(ResolutionStep("anchor", f"panel_name={want!r}", len(panels)))

    if anchor.transient_kind is not None:
        kind = anchor.transient_kind
        panels = [p for p in panels if p.transient_kind == kind]
        steps.append(ResolutionStep("anchor", f"transient_kind={kind!r}", len(panels)))

    if anchor.spawned_by is not None:
        parent_ids = {e.element_id for e in obs.elements if anchor.spawned_by.matches(e)}
        panels = [p for p in panels if p.spawned_by_element_id in parent_ids]
        steps.append(ResolutionStep("anchor", f"spawned_by (parents={len(parent_ids)})", len(panels)))

    if anchor.ordinal is not None:
        ordinal = anchor.ordinal
        if ordinal == "just_opened":
            transients = [p for p in panels if p.transient]
            if not transients:
                steps.append(ResolutionStep("anchor", "just_opened: no transients", 0))
                return steps, []
            top_seq = max(p.spawn_sequence for p in transients)
            winners = [p for p in transients if p.spawn_sequence == top_seq]
            panels = winners
            steps.append(
                ResolutionStep(
                    "anchor",
                    f"just_opened seq={top_seq} (ties={len(winners)})",
                    len(panels),
                )
            )
        else:
            panels = [p for p in panels if str(p.ordinal or "").casefold().strip() == ordinal]
            steps.append(ResolutionStep("anchor", f"ordinal={ordinal!r}", len(panels)))

    return steps, panels


def _within_filter(
    within: WithinPanel,
    elements: list[UIElement],
    obs: SceneObservation,
    steps: list[ResolutionStep],
    stage_detail: str,
) -> tuple[list[UIElement], str | None]:
    """Apply within-panel band/relation to an ordered element list.

    Returns (survivors, unsupported_reason). beside ⇒ unsupported_relation.
    """
    if within.relation == "beside":
        return [], "unsupported_relation"

    survivors = list(elements)
    length = len(survivors)

    if within.relation is not None and within.relative_to is not None:
        ref_indices = [i for i, e in enumerate(survivors) if within.relative_to.matches(e)]
        if len(ref_indices) > 1:
            # ambiguous reference: keep none and let caller report ambiguous upstream
            return [], None
        if not ref_indices:
            survivors = []
        else:
            ref = ref_indices[0]
            if within.relation == "below":
                survivors = [e for i, e in enumerate(survivors) if i > ref]
            else:  # above
                survivors = [e for i, e in enumerate(survivors) if i < ref]
        steps.append(ResolutionStep("within", f"{stage_detail} relation={within.relation}", len(survivors)))
        length = len(survivors)

    if within.band is not None:
        band = within.band
        if length == 0:
            survivors = []
        else:
            kept = [e for i, e in enumerate(survivors) if _element_band(e, i, length) == band]
            survivors = kept
        steps.append(ResolutionStep("within", f"{stage_detail} band={band!r}", len(survivors)))

    return survivors, None


def resolve_locator(locator: Locator, obs: SceneObservation, *, max_candidates: int = 20) -> ResolutionResult:
    """Single source of truth for locator → ResolvedTarget resolution."""
    all_steps: list[ResolutionStep] = []

    # --- anchor stage ---
    anchor_steps, panels = _match_panels(locator.anchor, obs)
    all_steps.extend(anchor_steps)
    if not panels:
        return ResolutionResult(
            status="not_found",
            trace=tuple(all_steps),
            reason="anchor matched no panel",
        )

    # --- gather elements per candidate panel (ordered) ---
    unsupported: str | None = None
    descriptor_survivors: list[UIElement] = []
    for panel in panels:
        ordered = list(obs.elements_in(panel.panel_id))
        if locator.within is not None:
            kept, reason = _within_filter(locator.within, ordered, obs, all_steps, f"panel={panel.panel_id}")
            if reason is not None:
                unsupported = reason
                break
            ordered = kept
        else:
            all_steps.append(ResolutionStep("within", f"panel={panel.panel_id} (no within filter)", len(ordered)))
        descriptor_survivors.extend(ordered)

    if unsupported:
        return ResolutionResult(
            status="unsupported",
            trace=tuple(all_steps),
            reason=unsupported,
        )

    # --- descriptor stage ---
    if locator.descriptor is not None:
        descriptor_survivors = [e for e in descriptor_survivors if locator.descriptor.matches(e)]
        all_steps.append(ResolutionStep("descriptor", "descriptor filter", len(descriptor_survivors)))

    if not descriptor_survivors:
        scope_elements = []
        for panel in panels:
            scope_elements.extend(obs.elements_in(panel.panel_id))
        if not scope_elements:
            scope_elements = list(obs.elements)
        enumerated = tuple(_element_summary(e) for e in scope_elements[:max_candidates])
        return ResolutionResult(
            status="not_found",
            trace=tuple(all_steps),
            not_found_candidates=enumerated,
            reason="no element matched the full locator",
        )

    trace = tuple(all_steps)
    candidates = tuple(_bound(e, obs, trace) for e in descriptor_survivors)

    if len(candidates) > 1:
        return ResolutionResult(status="ambiguous", candidates=candidates, trace=trace)

    only = candidates[0]
    status: ResolveStatus = "resolved" if only.available else "blocked"
    return ResolutionResult(status=status, candidates=candidates, trace=trace)


class PerceptionRegion:
    """No-LLM perception layer: structured queries over a SceneObservation.

    Mirrors the TopologicalRegion shape (``__init__`` + sync consult). Caller wraps
    in try/except returning ``("", err)`` if desired. Events carry only shape
    (counts/roles/ids/status), never element labels, attribute values, or traces.
    """

    def __init__(self, *, event_sink: EventSink = emit_event) -> None:
        self._event_sink = event_sink

    def survey(self, obs: SceneObservation) -> dict[str, Any]:
        panels = [
            {
                "role": p.role,
                "label": p.label,
                "ordinal": p.ordinal,
                "transient_kind": p.transient_kind,
                "spawn_sequence": p.spawn_sequence,
                "element_count": len(obs.elements_in(p.panel_id)),
            }
            for p in obs.panels
        ]
        result = {
            "panel_count": len(obs.panels),
            "transient_count": sum(1 for p in obs.panels if p.transient),
            "panels": panels,
        }
        self._emit(
            "computer.perception.survey",
            frame_id=obs.frame.frame_id,
            panel_count=len(obs.panels),
            panel_roles=sorted({p.role for p in obs.panels}),
            transient_count=sum(1 for p in obs.panels if p.transient),
        )
        return self._envelope("survey", obs, "ok", result)

    def focus(self, obs: SceneObservation, panel_id: str) -> dict[str, Any]:
        panel = obs.panel(panel_id)
        if panel is None:
            return self._envelope("focus", obs, "not_found", {"panel_id": panel_id})
        elements = obs.elements_in(panel_id)
        rendered = []
        for element in elements:
            available, blocker = _blocker_of(element)
            rendered.append(
                {
                    "element_id": element.element_id,
                    "role": element.role,
                    "label": element.label,
                    "enabled": element.enabled,
                    "visible": element.visible,
                    "available": available,
                    "blocker": blocker,
                }
            )
        result = {
            "panel": {"role": panel.role, "label": panel.label},
            "element_count": len(elements),
            "elements": rendered,
        }
        self._emit(
            "computer.perception.focus",
            frame_id=obs.frame.frame_id,
            panel_id=panel_id,
            element_count=len(elements),
            blocker_counts=_count_blockers(elements),
        )
        return self._envelope("focus", obs, "ok", result)

    def resolve_panel(self, anchor: PanelAnchor, obs: SceneObservation) -> tuple[str, str | None]:
        """Resolve a PanelAnchor to a single panel_id (anchor → panel, no element).

        Returns ``(status, panel_id)``: ``resolved`` + id for exactly one match,
        ``ambiguous`` + None for >1, ``not_found`` + None for 0. Used by ``session.focus``
        (缝 2/8) to turn a descriptor into the internal handle the adapter crops — the main
        brain never holds a raw panel_id.
        """
        _steps, panels = _match_panels(anchor, obs)
        if not panels:
            return "not_found", None
        if len(panels) > 1:
            return "ambiguous", None
        return "resolved", panels[0].panel_id

    def find(
        self,
        obs: SceneObservation,
        descriptor: ElementDescriptor,
        *,
        panel_id: str | None = None,
        max_candidates: int = 20,
    ) -> dict[str, Any]:
        scope = list(obs.elements_in(panel_id)) if panel_id is not None else list(obs.elements)
        matches = [e for e in scope if descriptor.matches(e)]
        if not matches:
            enumerated = tuple(_element_summary(e) for e in scope[:max_candidates])
            result = {
                "descriptor": descriptor.to_dict(),
                "matches": [],
                "candidates": list(enumerated),
                "truncated": len(scope) > max_candidates,
            }
            self._emit(
                "computer.perception.find",
                frame_id=obs.frame.frame_id,
                status="not_found",
                candidate_count=len(scope),
                truncated=len(scope) > max_candidates,
            )
            return self._envelope("find", obs, "not_found", result, candidate_count=len(scope))
        status = "ambiguous" if len(matches) > 1 else "resolved"
        result = {
            "descriptor": descriptor.to_dict(),
            "matches": [_element_summary(e) for e in matches],
        }
        self._emit(
            "computer.perception.find",
            frame_id=obs.frame.frame_id,
            status=status,
            candidate_count=len(matches),
        )
        return self._envelope("find", obs, status, result, candidate_count=len(matches))

    def describe(self, obs: SceneObservation, element_id: str) -> dict[str, Any]:
        element = obs.element(element_id)
        if element is None:
            return self._envelope("describe", obs, "not_found", {"element_id": element_id})
        available, blocker = _blocker_of(element)
        result = {
            "element_id": element.element_id,
            "panel_id": element.panel_id,
            "role": element.role,
            "label": element.label,
            "enabled": element.enabled,
            "visible": element.visible,
            "available": available,
            "blocker": blocker,
            "attributes": dict(element.attributes),
        }
        self._emit(
            "computer.perception.describe",
            frame_id=obs.frame.frame_id,
            element_id=element_id,
            role=element.role,
            available=available,
        )
        return self._envelope("describe", obs, "ok", result)

    def resolve(self, locator: Locator, obs: SceneObservation, *, max_candidates: int = 20) -> ResolutionResult:
        result = resolve_locator(locator, obs, max_candidates=max_candidates)
        self._emit(
            "computer.perception.resolve",
            frame_id=obs.frame.frame_id,
            status=result.status,
            candidate_count=len(result.candidates),
            truncated=len(result.not_found_candidates) >= max_candidates,
        )
        return result

    def compare(self, obs_before: SceneObservation, obs_after: SceneObservation) -> dict[str, Any]:
        """Temporal verification primitive — deferred at mock stage (explicit stub)."""
        result = {"deferred": True, "reason": "temporal_verify_not_built_mock_stage"}
        self._emit(
            "computer.perception.compare",
            status="deferred",
            before_frame=obs_before.frame.frame_id,
            after_frame=obs_after.frame.frame_id,
        )
        return self._envelope("compare", obs_after, "deferred", result)

    def _envelope(
        self,
        operation: str,
        obs: SceneObservation,
        status: str,
        result: dict[str, Any],
        *,
        candidate_count: int = 0,
        truncated: bool = False,
    ) -> dict[str, Any]:
        return {
            "operation": operation,
            "status": status,
            "observation": {"frame_id": obs.frame.frame_id, "state_sha256": obs.state_sha256},
            "result": result,
            "diagnostics": {"candidate_count": candidate_count, "truncated": truncated},
        }

    def _emit(self, event_type: str, **fields: object) -> None:
        self._event_sink(event_type, **fields)


def _count_blockers(elements: list[UIElement]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for element in elements:
        _available, blocker = _blocker_of(element)
        if blocker is not None:
            counts[blocker] = counts.get(blocker, 0) + 1
    return counts
