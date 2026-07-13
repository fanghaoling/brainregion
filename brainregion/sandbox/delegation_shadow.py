"""Offline replay of candidate delegation gates over content-free progress traces."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ShadowGateDecision:
    activated: bool
    step: int | None = None
    remaining_steps: int | None = None
    reason: str = ""
    signals: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "activated": self.activated,
            "step": self.step,
            "remaining_steps": self.remaining_steps,
            "reason": self.reason,
            "signals": list(self.signals),
        }


@dataclass(frozen=True)
class ShadowGatePolicy:
    name: str
    min_remaining_steps: int = 1
    no_effect_steps: int | None = None
    no_progress_steps: int | None = None
    repeated_operation_steps: int | None = None
    repeated_target_steps: int | None = None
    recent_error_steps: int | None = 2
    deadline_without_effect: int | None = None
    trigger_on_failed_verification: bool = True

    def __post_init__(self) -> None:
        if not self.name or len(self.name) > 100:
            raise ValueError("shadow gate policy name must contain 1..100 characters")
        for field_name in (
            "min_remaining_steps",
            "no_effect_steps",
            "no_progress_steps",
            "repeated_operation_steps",
            "repeated_target_steps",
            "recent_error_steps",
            "deadline_without_effect",
        ):
            value = getattr(self, field_name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{field_name} must be null or a positive integer")

    def replay(self, trace: list[dict[str, Any]], *, max_steps: int) -> ShadowGateDecision:
        for completed in range(1, min(len(trace), max_steps) + 1):
            remaining = max_steps - completed
            if remaining < self.min_remaining_steps:
                continue
            signals = self._signals(trace[:completed], remaining_steps=remaining)
            if signals:
                return ShadowGateDecision(
                    activated=True,
                    step=completed,
                    remaining_steps=remaining,
                    reason=signals[0],
                    signals=tuple(signals),
                )
        return ShadowGateDecision(activated=False)

    def _signals(self, prefix: list[dict[str, Any]], *, remaining_steps: int) -> list[str]:
        signals: list[str] = []
        if self.trigger_on_failed_verification and prefix[-1].get("verification_passed") is False:
            signals.append("verification_failed")
        if self.no_effect_steps and _tail_count(prefix, lambda step: not step["workspace_effect"]) >= self.no_effect_steps:
            signals.append("no_workspace_effect")
        if self.no_progress_steps and _tail_count(prefix, _lacks_progress) >= self.no_progress_steps:
            signals.append("no_new_progress")
        latest_stalled = _lacks_progress(prefix[-1])
        if (
            latest_stalled
            and self.repeated_operation_steps
            and _tail_same(prefix, "operation", self.repeated_operation_steps)
        ):
            signals.append("repeated_operation")
        if latest_stalled and self.repeated_target_steps and _tail_same(
            prefix,
            "target_fingerprint",
            self.repeated_target_steps,
            require_value=True,
        ):
            signals.append("repeated_target")
        if self.recent_error_steps and _tail_count(prefix, lambda step: bool(step["error"])) >= self.recent_error_steps:
            signals.append("repeated_error")
        if (
            self.deadline_without_effect is not None
            and remaining_steps <= self.deadline_without_effect
            and not any(bool(step["workspace_effect"]) for step in prefix)
        ):
            signals.append("deadline_without_effect")
        return signals


DEFAULT_SHADOW_POLICIES: tuple[ShadowGatePolicy, ...] = (
    ShadowGatePolicy(
        name="effect_only_v1",
        min_remaining_steps=2,
        no_effect_steps=2,
        recent_error_steps=2,
    ),
    ShadowGatePolicy(
        name="repetition_only",
        repeated_operation_steps=2,
        repeated_target_steps=2,
        recent_error_steps=2,
    ),
    ShadowGatePolicy(
        name="novelty_stall",
        no_progress_steps=2,
        repeated_target_steps=2,
        recent_error_steps=2,
    ),
    ShadowGatePolicy(
        name="novelty_with_deadline",
        no_progress_steps=2,
        repeated_target_steps=2,
        recent_error_steps=2,
        deadline_without_effect=2,
    ),
)


def summarize_shadow_gates(
    cases: list[dict[str, Any]],
    *,
    policies: tuple[ShadowGatePolicy, ...] = DEFAULT_SHADOW_POLICIES,
) -> dict[str, Any]:
    normalized = [_normalize_case(case) for case in cases]
    if len({case["case_id"] for case in normalized}) != len(normalized):
        raise ValueError("shadow gate case_id values must be unique")
    output: dict[str, Any] = {}
    for policy in policies:
        decisions = [policy.replay(case["progress_trace"], max_steps=case["max_steps"]) for case in normalized]
        activated = [decision for decision in decisions if decision.activated]
        easy_indexes = [index for index, case in enumerate(normalized) if case["solved"]]
        hard_indexes = [index for index, case in enumerate(normalized) if not case["solved"]]
        output[policy.name] = {
            "n_cases": len(normalized),
            "activations": len(activated),
            "activation_rate": _ratio(len(activated), len(normalized)),
            "expert_calls_avoided_vs_always_on": len(normalized) - len(activated),
            "easy_case_false_wake_rate": _indexed_rate(decisions, easy_indexes),
            "hard_case_wake_rate": _indexed_rate(decisions, hard_indexes),
            "mean_trigger_step": _mean([decision.step for decision in activated]),
            "mean_remaining_steps": _mean([decision.remaining_steps for decision in activated]),
            "decisions": [
                {
                    "task_id": case["task_id"],
                    "case_id": case["case_id"],
                    "eventual_solved": case["solved"],
                    "trace_quality": case["trace_quality"],
                    **decision.to_dict(),
                }
                for case, decision in zip(normalized, decisions, strict=True)
            ],
        }
    return {
        "n_cases": len(normalized),
        "label_semantics": "eventual main-only outcome is a calibration proxy, not causal expert need",
        "trace_quality": _quality_counts(normalized),
        "policies": output,
        "models_called": False,
        "contains_reasoning": False,
        "contains_tool_results": False,
    }


def shadow_cases_from_delegation_report(report: dict[str, Any], *, max_steps: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for case in report.get("cases") or []:
        if case.get("arm") != "main_only":
            continue
        main = case.get("main_result") or {}
        diagnostics = main.get("sandbox_diagnostics") or {}
        trace, quality = _trace_from_diagnostics(diagnostics)
        cases.append(
            {
                "task_id": case.get("task_id"),
                "case_id": f"{case.get('task_id')}#repeat={case.get('repeat', 0)}",
                "solved": main.get("solved"),
                "max_steps": max_steps,
                "progress_trace": trace,
                "trace_quality": quality,
            }
        )
    return cases


def render_shadow_gate_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"### delegation shadow gates (cases={summary.get('n_cases', 0)}, models_called=false)"
    ]
    for name, policy in (summary.get("policies") or {}).items():
        lines.append(
            f"  {name}: activation={policy.get('activation_rate')} "
            f"easy_false_wake={policy.get('easy_case_false_wake_rate')} "
            f"hard_wake={policy.get('hard_case_wake_rate')} "
            f"remaining={policy.get('mean_remaining_steps')} "
            f"avoided={policy.get('expert_calls_avoided_vs_always_on')}"
        )
    return "\n".join(lines)


def replay_shadow_report(path: str | Path, *, max_steps: int | None = None) -> dict[str, Any]:
    report = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    configured_steps = (report.get("execution") or {}).get("max_steps")
    resolved_steps = max_steps if max_steps is not None else configured_steps
    if isinstance(resolved_steps, bool) or not isinstance(resolved_steps, int) or resolved_steps <= 0:
        raise ValueError("max_steps is required for reports without execution.max_steps")
    return summarize_shadow_gates(
        shadow_cases_from_delegation_report(report, max_steps=resolved_steps)
    )


def _normalize_case(case: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise ValueError("shadow gate case must be an object")
    task_id = str(case.get("task_id") or "").strip()
    if not task_id or len(task_id) > 200:
        raise ValueError("shadow gate task_id must contain 1..200 characters")
    case_id = str(case.get("case_id") or task_id).strip()
    if not case_id or len(case_id) > 300:
        raise ValueError("shadow gate case_id must contain 1..300 characters")
    if not isinstance(case.get("solved"), bool):
        raise ValueError("shadow gate solved must be a boolean")
    max_steps = case.get("max_steps")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("shadow gate max_steps must be a positive integer")
    trace = case.get("progress_trace")
    if not isinstance(trace, list):
        raise ValueError("shadow gate progress_trace must be an array")
    return {
        "task_id": task_id,
        "case_id": case_id,
        "solved": case["solved"],
        "max_steps": max_steps,
        "progress_trace": [_normalize_step(step, index) for index, step in enumerate(trace)],
        "trace_quality": str(case.get("trace_quality") or "exact"),
    }


def _normalize_step(step: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(step, dict):
        raise ValueError("shadow progress step must be an object")
    operation = str(step.get("operation") or "").strip()
    if not operation or len(operation) > 100:
        raise ValueError("shadow progress operation must contain 1..100 characters")
    verification = step.get("verification_passed")
    if verification is not None and not isinstance(verification, bool):
        raise ValueError("verification_passed must be a boolean or null")
    return {
        "step": index,
        "operation": operation,
        "target_fingerprint": str(step.get("target_fingerprint") or "")[:64],
        "target_is_new": bool(step.get("target_is_new")),
        "workspace_effect": bool(step.get("workspace_effect")),
        "verification_passed": verification,
        "error": bool(step.get("error")),
    }


def _trace_from_diagnostics(diagnostics: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    exact = diagnostics.get("progress_trace")
    if isinstance(exact, list) and exact:
        return exact, "exact"
    operations = list(diagnostics.get("tool_sequence") or [])
    effects_remaining = int(diagnostics.get("workspace_effects") or 0)
    seen: set[str] = set()
    trace = []
    for operation in operations:
        fingerprint = hashlib.sha256(str(operation).encode("utf-8")).hexdigest()[:16]
        effect = operation == "apply_text_patch" and effects_remaining > 0
        if effect:
            effects_remaining -= 1
        verification = None
        if operation == "workspace_run_check":
            verification = diagnostics.get("last_verification_passed")
        trace.append(
            {
                "operation": str(operation),
                "target_fingerprint": fingerprint,
                "target_is_new": fingerprint not in seen,
                "workspace_effect": effect,
                "verification_passed": verification,
                "error": False,
            }
        )
        seen.add(fingerprint)
    return trace, "legacy_approximate"


def _lacks_progress(step: dict[str, Any]) -> bool:
    return not step["workspace_effect"] and not step["target_is_new"]


def _tail_count(trace: list[dict[str, Any]], predicate: Any) -> int:
    count = 0
    for step in reversed(trace):
        if not predicate(step):
            break
        count += 1
    return count


def _tail_same(
    trace: list[dict[str, Any]],
    key: str,
    threshold: int,
    *,
    require_value: bool = False,
) -> bool:
    if len(trace) < threshold:
        return False
    values = [step.get(key) for step in trace[-threshold:]]
    return (not require_value or bool(values[0])) and len(set(values)) == 1


def _indexed_rate(decisions: list[ShadowGateDecision], indexes: list[int]) -> float | None:
    return _ratio(sum(decisions[index].activated for index in indexes), len(indexes))


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: list[int | None]) -> float | None:
    numbers = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(numbers) / len(numbers) if numbers else None


def _quality_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        quality = case["trace_quality"]
        counts[quality] = counts.get(quality, 0) + 1
    return counts


__all__ = [
    "DEFAULT_SHADOW_POLICIES",
    "ShadowGateDecision",
    "ShadowGatePolicy",
    "render_shadow_gate_summary",
    "replay_shadow_report",
    "shadow_cases_from_delegation_report",
    "summarize_shadow_gates",
]
