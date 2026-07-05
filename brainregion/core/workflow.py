"""Deterministic workflow suggestions built on top of Brain Region routing.

The workflow layer is intentionally advisory. It never calls models, executes
tools, reads memory, or mutates project files. It only turns region matches into
explicit next-step suggestions for the main assistant or user to approve.
"""
from __future__ import annotations

from typing import Any

from .regions import REGIONS_DIR, route_regions

_REGION_ORDER = {
    "memory": 5,
    "planning": 10,
    "debugging": 20,
    "performance": 30,
    "security": 40,
    "unity_ecs": 50,
    "review": 60,
}


def _base_text(*parts: str) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())


def _action(
    *,
    tool: str,
    reason: str,
    suggested_args: dict[str, Any],
    source_regions: list[str],
    confidence: float,
) -> dict:
    return {
        "tool": tool,
        "reason": reason,
        "suggested_args": suggested_args,
        "source_regions": source_regions,
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "requires_user_approval": True,
    }


def _selected_region_ids(routing: dict) -> list[str]:
    return [str(region.get("id", "")) for region in routing.get("selected", []) if region.get("id")]


def _selected_confidence(routing: dict, *region_ids: str) -> float:
    wanted = set(region_ids)
    values = [
        float(region.get("confidence", 0.0))
        for region in routing.get("selected", [])
        if region.get("id") in wanted
    ]
    return max(values) if values else 0.0


def _append_once(actions: list[dict], action: dict) -> None:
    key = (action["tool"], tuple(action.get("source_regions", [])))
    for existing in actions:
        existing_key = (existing["tool"], tuple(existing.get("source_regions", [])))
        if existing_key == key:
            return
    actions.append(action)


def _build_actions(
    region_ids: list[str],
    confidence_by_id: dict[str, float],
    *,
    goal: str,
    problem: str,
    context: str,
    files: dict[str, str],
) -> list[dict]:
    """Turn a set of region ids into explicit next-step actions.

    Pure function over region ids + a confidence map; reused by suggest_workflow
    (deterministic selected set) and wake_gate (woken set incl. sentinel/shadow).
    Never calls models or tools; every action requires user approval.
    """
    selected_set = set(region_ids)
    actions: list[dict] = []
    combined_context = _base_text(context, problem)
    primary_problem = problem or goal or context

    def _conf(*region_ids: str) -> float:
        values = [confidence_by_id.get(rid, 0.0) for rid in region_ids]
        return max(values) if values else 0.0

    if "memory" in selected_set:
        query_text = _base_text(goal, problem, context)
        _append_once(
            actions,
            _action(
                tool="recall_experiences",
                reason="Memory Region matched: recall relevant project experience before planning, consulting, or editing.",
                suggested_args={"text": query_text, "top_k": 5},
                source_regions=["memory"],
                confidence=_conf("memory"),
            ),
        )
        _append_once(
            actions,
            _action(
                tool="inspect",
                reason="Memory Region matched: inspect memory health and recent experience before adding or promoting knowledge.",
                suggested_args={"view": "memory", "memory_preview_k": 8},
                source_regions=["memory"],
                confidence=_conf("memory"),
            ),
        )
        _append_once(
            actions,
            _action(
                tool="record_experience",
                reason="Memory Region matched: after the result is verified, record the distilled lesson as an Experience Memory card.",
                suggested_args={
                    "summary": "",
                    "details": "",
                    "triggers": [],
                    "region": "memory",
                    "status": "pending",
                },
                source_regions=["memory"],
                confidence=_conf("memory"),
            ),
        )

    if "planning" in selected_set:
        suggested_args: dict[str, Any] = {"goal": goal or problem or context}
        if combined_context:
            suggested_args["context"] = combined_context
        if files:
            suggested_args["files"] = files
        _append_once(
            actions,
            _action(
                tool="plan_task",
                reason="Planning Region matched: decompose the goal into milestones, tasks, risks, and acceptance criteria.",
                suggested_args=suggested_args,
                source_regions=["planning"],
                confidence=_conf("planning"),
            ),
        )

    if "debugging" in selected_set:
        suggested_args = {
            "problem": primary_problem,
            "context": context,
            "mode": "debugging",
        }
        source_regions = ["debugging"]
        if "unity_ecs" in selected_set:
            suggested_args["consultants"] = ["debugger", "unity_ecs"]
            source_regions.append("unity_ecs")
        if files:
            suggested_args["files"] = files
        _append_once(
            actions,
            _action(
                tool="consult_problem",
                reason="Debugging Region matched: ask an external debugger for hypotheses and next experiments.",
                suggested_args=suggested_args,
                source_regions=source_regions,
                confidence=_conf(*source_regions),
            ),
        )

    if "performance" in selected_set:
        suggested_args = {
            "problem": primary_problem,
            "context": context,
            "mode": "performance",
        }
        source_regions = ["performance"]
        if "unity_ecs" in selected_set:
            suggested_args["consultants"] = ["performance", "unity_ecs", "critic"]
            source_regions.append("unity_ecs")
        if files:
            suggested_args["files"] = files
        _append_once(
            actions,
            _action(
                tool="consult_problem",
                reason="Performance Region matched: ask a performance specialist to focus on latency, allocation, throughput, or cost.",
                suggested_args=suggested_args,
                source_regions=source_regions,
                confidence=_conf(*source_regions),
            ),
        )

    if "security" in selected_set:
        suggested_args = {
            "problem": primary_problem,
            "context": context,
            "mode": "challenge",
        }
        if files:
            suggested_args["files"] = files
        _append_once(
            actions,
            _action(
                tool="consult_problem",
                reason="Security Region matched: challenge the design around secrets, permissions, privacy, or injection boundaries.",
                suggested_args=suggested_args,
                source_regions=["security"],
                confidence=_conf("security"),
            ),
        )

    if "unity_ecs" in selected_set and not ({"debugging", "performance"} & selected_set):
        suggested_args = {
            "problem": primary_problem,
            "context": context,
            "consultants": ["unity_ecs"],
        }
        if files:
            suggested_args["files"] = files
        _append_once(
            actions,
            _action(
                tool="consult_problem",
                reason="Unity ECS Region matched: ask the Unity ECS consultant for DOTS, Burst, Jobs, and data-oriented guidance.",
                suggested_args=suggested_args,
                source_regions=["unity_ecs"],
                confidence=_conf("unity_ecs"),
            ),
        )

    if "review" in selected_set:
        if files:
            suggested_args = {"files": files}
            if context or goal or problem:
                suggested_args["extra_context"] = _base_text(goal, problem, context)
            tool = "review_code"
            reason = "Review Region matched and files were provided: review the code change before implementation or merge."
        else:
            suggested_args = {
                "content": _base_text(goal, problem, context),
                "document_type": "markdown",
            }
            tool = "review_document"
            reason = "Review Region matched: review the written plan, design, or decision record."
        _append_once(
            actions,
            _action(
                tool=tool,
                reason=reason,
                suggested_args=suggested_args,
                source_regions=["review"],
                confidence=_conf("review"),
            ),
        )

    actions.sort(
        key=lambda item: min(_REGION_ORDER.get(region, 999) for region in item.get("source_regions", [""]))
    )
    return actions


def suggest_workflow(
    *,
    goal: str = "",
    problem: str = "",
    context: str = "",
    files: dict[str, str] | None = None,
    top_k: int = 3,
    min_score: int = 2,
    regions_dir=REGIONS_DIR,
) -> dict:
    """Suggest explicit manual next steps from deterministic region routing."""
    files = files or {}
    routing = route_regions(
        goal=goal,
        problem=problem,
        context=context,
        files=files,
        top_k=top_k,
        min_score=min_score,
        regions_dir=regions_dir,
    )
    selected = list(routing.get("selected", []))
    selected_ids = _selected_region_ids(routing)
    confidence_by_id = {
        str(region.get("id", "")): float(region.get("confidence", 0.0))
        for region in selected
        if region.get("id")
    }
    actions = _build_actions(
        selected_ids,
        confidence_by_id,
        goal=goal,
        problem=problem,
        context=context,
        files=files,
    )
    skipped = [
        {
            "region": region_id,
            "reason": "No executable workflow suggestion is available in this MVP.",
        }
        for region_id in selected_ids
        if region_id not in _REGION_ORDER
    ]
    return {
        "selected_regions": selected,
        "next_actions": actions,
        "skipped_regions": skipped,
        "trace": {
            "strategy": "explicit_workflow_suggestions_v1",
            "routing_strategy": routing.get("trace", {}).get("strategy"),
            "routing": routing.get("trace", {}),
            "auto_execute": False,
            "models_called": False,
            "tools_called": ["route_regions"],
            "requires_user_approval": True,
        },
    }
