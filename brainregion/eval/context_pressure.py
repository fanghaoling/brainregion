"""Controlled, content-free reporting for model context-pressure A/B probes."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from brainregion.core.context_pressure import ContextPressureObserver
from brainregion.core.stages.parse import extract_json_object
from brainregion.runtime import normalize_usage

ARM_LOW_LOAD = "low_load"
ARM_HIGH_LOAD = "high_load"
ARM_SAME_LOAD = "same_load"
CONTEXT_PRESSURE_ARMS = (ARM_LOW_LOAD, ARM_HIGH_LOAD)
CONTEXT_PRESSURE_POSITIONS = ("early", "middle", "late")

_POSITION_FRACTIONS = {"early": 0.1, "middle": 0.5, "late": 0.9}


@dataclass(frozen=True)
class ContextPressureProbeSpec:
    case_id: str
    answer: str
    needle_position: str

    def __post_init__(self) -> None:
        if not str(self.case_id or "").strip():
            raise ValueError("case_id cannot be empty")
        if not str(self.answer or "").strip():
            raise ValueError("answer cannot be empty")
        if self.needle_position not in _POSITION_FRACTIONS:
            raise ValueError(f"unknown needle position: {self.needle_position!r}")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _fill_ratio(value: Any, name: str) -> float:
    ratio = float(value)
    if not math.isfinite(ratio) or ratio <= 0 or ratio >= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return ratio


def _probe_answer(run_id: str, repeat: int, position: str) -> str:
    digest = hashlib.sha256(f"{run_id}:{repeat}:{position}".encode()).hexdigest()[:16]
    return f"BR-{digest}"


def _build_probe_prompt(spec: ContextPressureProbeSpec, *, target_tokens: int) -> str:
    target_chars = max(400, target_tokens * 4)
    filler_unit = "neutral-record: alpha beta gamma delta; no answer is present here.\n"
    filler = (filler_unit * ((target_chars // len(filler_unit)) + 1))[:target_chars]
    index = int(len(filler) * _POSITION_FRACTIONS[spec.needle_position])
    needle = (
        f"\nAUTHORITATIVE_RECORD case={spec.case_id} answer={spec.answer}\n"
    )
    context = f"{filler[:index]}{needle}{filler[index:]}"
    return (
        "Read the synthetic records and return JSON only: "
        '{"answer":"the exact AUTHORITATIVE_RECORD answer"}. '
        "Do not infer an answer from neutral records.\n\n"
        f"{context}"
    )


async def _run_probe(
    backend: Any,
    model: str,
    *,
    endpoint_id: str | None,
    arm: str,
    spec: ContextPressureProbeSpec,
    target_tokens: int,
    context_window_tokens: int | None,
    max_tokens: int,
    execution_order: int,
    repeat: int,
    observer: ContextPressureObserver,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = await backend.complete(
        model=model,
        endpoint_id=endpoint_id,
        system=(
            "You are a deterministic context-recall probe. Follow the final user "
            "instruction and return one JSON object without commentary."
        ),
        user=_build_probe_prompt(spec, target_tokens=target_tokens),
        temperature=0.0,
        max_tokens=max_tokens,
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    usage = normalize_usage(getattr(response, "usage", None) or {})
    parsed = extract_json_object(str(getattr(response, "content", "") or ""))
    observed_answer = str((parsed or {}).get("answer") or "").strip()
    error = str(getattr(response, "error", "") or "")[:300]
    correct = bool(not error and observed_answer == spec.answer)
    sample = observer.observe(
        step=execution_order,
        task_id=f"context-pressure:{spec.case_id}",
        assignment_id=f"{repeat}:{arm}:{spec.needle_position}",
        region="context_probe",
        model=model,
        endpoint_id=endpoint_id,
        context_tokens=usage["input_tokens"],
        context_budget_tokens=context_window_tokens or 0,
        usage=usage,
        model_called=True,
        error_observed=bool(error),
        context_status="synthetic_probe",
        lifecycle_state="completed" if not error else "error",
    )
    raw_cost = getattr(response, "cost_usd", None)
    cost = float(raw_cost) if raw_cost is not None else 0.0
    if not math.isfinite(cost) or cost < 0:
        cost = 0.0
    return {
        "arm": arm,
        "repeat": repeat,
        "case_id": spec.case_id,
        "needle_position": spec.needle_position,
        "execution_order": execution_order,
        "target_input_tokens": target_tokens,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "actual_window_fill_ratio": sample.model_window_fill_ratio,
        "pressure_score": sample.pressure_score,
        "pressure_band": sample.pressure_band,
        "correct": correct,
        "parse_ok": parsed is not None,
        "error_observed": bool(error),
        "cost_usd": round(cost, 6),
        "cost_source": getattr(response, "cost_source", None),
        "latency_ms": latency_ms,
        "contains_request_content": False,
        "contains_response_content": False,
        "contains_reasoning": False,
    }


async def run_context_pressure_eval(
    backend: Any,
    model: str,
    *,
    context_window_tokens: int,
    endpoint_id: str | None = None,
    repeats: int = 2,
    low_fill_ratio: float = 0.1,
    high_fill_ratio: float = 0.7,
    needle_positions: Sequence[str] = CONTEXT_PRESSURE_POSITIONS,
    max_probe_tokens: int = 64000,
    max_total_probe_tokens: int = 512000,
    max_tokens: int = 128,
    run_id: str = "",
    context_window_source: str = "explicit_argument",
) -> dict[str, Any]:
    """Run an explicit matched low/high synthetic context-recall experiment."""

    context_window_tokens = _positive_int(
        context_window_tokens, "context_window_tokens"
    )
    repeats = _positive_int(repeats, "repeats")
    max_probe_tokens = _positive_int(max_probe_tokens, "max_probe_tokens")
    max_total_probe_tokens = _positive_int(
        max_total_probe_tokens, "max_total_probe_tokens"
    )
    max_tokens = _positive_int(max_tokens, "max_tokens")
    low_fill_ratio = _fill_ratio(low_fill_ratio, "low_fill_ratio")
    high_fill_ratio = _fill_ratio(high_fill_ratio, "high_fill_ratio")
    if high_fill_ratio <= low_fill_ratio:
        raise ValueError("high_fill_ratio must exceed low_fill_ratio")
    positions = tuple(str(item).strip().casefold() for item in needle_positions)
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("needle_positions must be non-empty and unique")
    unknown = sorted(set(positions) - set(_POSITION_FRACTIONS))
    if unknown:
        raise ValueError(f"unknown needle positions: {unknown}")
    model = str(model or "").strip()
    if not model:
        raise ValueError("model cannot be empty")
    context_window_source = str(context_window_source or "").strip()
    if not context_window_source or len(context_window_source) > 300:
        raise ValueError("context_window_source must be 1..300 characters")
    run_id = run_id or f"context-pressure-eval-{int(time.time() * 1000)}"
    targets = {
        ARM_LOW_LOAD: min(
            max_probe_tokens, max(1, int(context_window_tokens * low_fill_ratio))
        ),
        ARM_HIGH_LOAD: min(
            max_probe_tokens, max(1, int(context_window_tokens * high_fill_ratio))
        ),
    }
    if targets[ARM_HIGH_LOAD] <= targets[ARM_LOW_LOAD]:
        raise ValueError(
            "max_probe_tokens collapses low/high arms; raise the cap or lower ratios"
        )
    planned_input_tokens = repeats * len(positions) * sum(targets.values())
    if planned_input_tokens > max_total_probe_tokens:
        raise ValueError(
            "planned probe input exceeds max_total_probe_tokens; reduce repeats, "
            "positions, ratios, or per-call cap"
        )
    identity = f"{endpoint_id}/{model}" if endpoint_id else model
    observer = ContextPressureObserver(
        model_context_limits={identity: context_window_tokens},
        emit_events=False,
    )
    records: list[dict[str, Any]] = []
    execution_order = 0
    arm_order_counts = {"low_load_first": 0, "high_load_first": 0}
    for repeat in range(repeats):
        ordered_arms = list(CONTEXT_PRESSURE_ARMS)
        if repeat % 2:
            ordered_arms.reverse()
        arm_order_counts[f"{ordered_arms[0]}_first"] += 1
        for position in positions:
            spec = ContextPressureProbeSpec(
                case_id=f"{repeat}:{position}",
                answer=_probe_answer(run_id, repeat, position),
                needle_position=position,
            )
            for arm in ordered_arms:
                execution_order += 1
                records.append(
                    await _run_probe(
                        backend,
                        model,
                        endpoint_id=endpoint_id,
                        arm=arm,
                        spec=spec,
                        target_tokens=targets[arm],
                        context_window_tokens=context_window_tokens,
                        max_tokens=max_tokens,
                        execution_order=execution_order,
                        repeat=repeat,
                        observer=observer,
                    )
                )
    report = summarize_context_pressure_records(records, run_id=run_id)
    report["records"] = records
    actual_total_input_tokens = sum(item["input_tokens"] for item in records)
    report["execution"] = {
        "model": model,
        "endpoint_id": endpoint_id,
        "context_window_tokens": context_window_tokens,
        "repeats": repeats,
        "needle_positions": list(positions),
        "target_input_tokens": targets,
        "requested_fill_ratios": {
            ARM_LOW_LOAD: low_fill_ratio,
            ARM_HIGH_LOAD: high_fill_ratio,
        },
        "target_capped": {
            ARM_LOW_LOAD: targets[ARM_LOW_LOAD]
            < int(context_window_tokens * low_fill_ratio),
            ARM_HIGH_LOAD: targets[ARM_HIGH_LOAD]
            < int(context_window_tokens * high_fill_ratio),
        },
        "max_probe_tokens": max_probe_tokens,
        "max_total_probe_tokens": max_total_probe_tokens,
        "planned_input_tokens": planned_input_tokens,
        "actual_total_input_tokens": actual_total_input_tokens,
        "actual_total_exceeds_target_plan": (
            actual_total_input_tokens > planned_input_tokens
        ),
        "actual_total_exceeds_configured_guard": (
            actual_total_input_tokens > max_total_probe_tokens
        ),
        "calls_exceeding_arm_target": sum(
            int(item["input_tokens"] > targets[item["arm"]]) for item in records
        ),
        "calls_exceeding_per_probe_target_cap": sum(
            int(item["input_tokens"] > max_probe_tokens) for item in records
        ),
        "input_cap_interpretation": (
            "synthetic_target_cap_actual_provider_usage_may_exceed"
        ),
        "capacity_source": context_window_source,
        "provider_capacity_verified_at_runtime": False,
        "arm_order_policy": "alternating_by_repeat",
        "arm_order_counts": arm_order_counts,
        "counterbalanced_order": all(value > 0 for value in arm_order_counts.values()),
        "explicit_model_calls": True,
        "changes_runtime_policy": False,
        "contains_request_content": False,
        "contains_response_content": False,
        "contains_reasoning": False,
    }
    report["context_pressure"] = observer.snapshot()
    return report


async def run_context_stability_control(
    backend: Any,
    model: str,
    *,
    target_input_tokens: int = 2000,
    repeats: int = 3,
    needle_position: str = "middle",
    endpoint_id: str | None = None,
    reference_context_window_tokens: int | None = None,
    max_total_probe_tokens: int = 10000,
    max_tokens: int = 128,
    run_id: str = "",
    context_window_source: str = "unknown",
) -> dict[str, Any]:
    """Repeat one identical synthetic prompt to expose request-order drift."""

    target_input_tokens = _positive_int(target_input_tokens, "target_input_tokens")
    repeats = _positive_int(repeats, "repeats")
    if repeats < 2:
        raise ValueError("repeats must be at least 2 for a stability control")
    max_total_probe_tokens = _positive_int(
        max_total_probe_tokens, "max_total_probe_tokens"
    )
    max_tokens = _positive_int(max_tokens, "max_tokens")
    planned_input_tokens = repeats * target_input_tokens
    if planned_input_tokens > max_total_probe_tokens:
        raise ValueError(
            "planned stability input exceeds max_total_probe_tokens; reduce repeats "
            "or target_input_tokens"
        )
    position = str(needle_position or "").strip().casefold()
    if position not in _POSITION_FRACTIONS:
        raise ValueError(f"unknown needle position: {position!r}")
    model = str(model or "").strip()
    if not model:
        raise ValueError("model cannot be empty")
    if reference_context_window_tokens is not None:
        reference_context_window_tokens = _positive_int(
            reference_context_window_tokens,
            "reference_context_window_tokens",
        )
    context_window_source = str(context_window_source or "").strip()
    if not context_window_source or len(context_window_source) > 300:
        raise ValueError("context_window_source must be 1..300 characters")
    run_id = run_id or f"context-stability-{int(time.time() * 1000)}"
    identity = f"{endpoint_id}/{model}" if endpoint_id else model
    observer = ContextPressureObserver(
        model_context_limits=(
            {identity: reference_context_window_tokens}
            if reference_context_window_tokens is not None
            else {}
        ),
        emit_events=False,
    )
    spec = ContextPressureProbeSpec(
        case_id=f"stability:{position}",
        answer=_probe_answer(run_id, 0, position),
        needle_position=position,
    )
    records: list[dict[str, Any]] = []
    for repeat in range(repeats):
        records.append(
            await _run_probe(
                backend,
                model,
                endpoint_id=endpoint_id,
                arm=ARM_SAME_LOAD,
                spec=spec,
                target_tokens=target_input_tokens,
                context_window_tokens=reference_context_window_tokens,
                max_tokens=max_tokens,
                execution_order=repeat + 1,
                repeat=repeat,
                observer=observer,
            )
        )
    report = summarize_context_stability_records(records, run_id=run_id)
    report["records"] = records
    actual_total_input_tokens = sum(item["input_tokens"] for item in records)
    report["execution"] = {
        "model": model,
        "endpoint_id": endpoint_id,
        "target_input_tokens": target_input_tokens,
        "repeats": repeats,
        "needle_position": position,
        "max_total_probe_tokens": max_total_probe_tokens,
        "planned_input_tokens": planned_input_tokens,
        "actual_total_input_tokens": actual_total_input_tokens,
        "actual_total_exceeds_target_plan": (
            actual_total_input_tokens > planned_input_tokens
        ),
        "actual_total_exceeds_configured_guard": (
            actual_total_input_tokens > max_total_probe_tokens
        ),
        "reference_context_window_tokens": reference_context_window_tokens,
        "capacity_source": context_window_source,
        "provider_capacity_verified_at_runtime": False,
        "prompt_policy": "identical_prompt_across_repeats",
        "explicit_model_calls": True,
        "changes_runtime_policy": False,
        "contains_request_content": False,
        "contains_response_content": False,
        "contains_reasoning": False,
    }
    report["context_pressure"] = observer.snapshot()
    return report


def summarize_context_pressure_records(
    records: Sequence[dict[str, Any]],
    *,
    run_id: str = "",
) -> dict[str, Any]:
    if not records:
        raise ValueError("context pressure records cannot be empty")
    grouped = {arm: [] for arm in CONTEXT_PRESSURE_ARMS}
    pairs: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for record in records:
        arm = str(record.get("arm") or "")
        if arm not in grouped:
            raise ValueError(f"unknown context pressure arm: {arm!r}")
        grouped[arm].append(record)
        key = (int(record.get("repeat", 0)), str(record.get("needle_position") or ""))
        if arm in pairs.setdefault(key, {}):
            raise ValueError(f"duplicate context pressure pair member: {key!r}/{arm}")
        pairs[key][arm] = record
    incomplete = [key for key, value in pairs.items() if set(value) != set(CONTEXT_PRESSURE_ARMS)]
    if incomplete:
        raise ValueError(f"incomplete context pressure pairs: {incomplete}")

    per_arm: dict[str, dict[str, Any]] = {}
    for arm, items in grouped.items():
        fills = [
            float(item["actual_window_fill_ratio"])
            for item in items
            if item.get("actual_window_fill_ratio") is not None
        ]
        per_arm[arm] = {
            "runs": len(items),
            "correct": sum(int(bool(item.get("correct"))) for item in items),
            "correct_rate": round(
                sum(int(bool(item.get("correct"))) for item in items) / len(items), 4
            ),
            "parse_failure_count": sum(
                int(not bool(item.get("parse_ok"))) for item in items
            ),
            "error_count": sum(int(bool(item.get("error_observed"))) for item in items),
            "mean_input_tokens": round(
                sum(int(item.get("input_tokens", 0)) for item in items) / len(items), 2
            ),
            "mean_actual_window_fill_ratio": (
                round(sum(fills) / len(fills), 4) if fills else None
            ),
            "cost_usd": round(
                sum(float(item.get("cost_usd", 0.0)) for item in items), 6
            ),
        }
    low = per_arm[ARM_LOW_LOAD]
    high = per_arm[ARM_HIGH_LOAD]
    fill_increased = (
        low["mean_actual_window_fill_ratio"] is not None
        and high["mean_actual_window_fill_ratio"] is not None
        and high["mean_actual_window_fill_ratio"]
        > low["mean_actual_window_fill_ratio"]
    )
    quality_delta = round(high["correct_rate"] - low["correct_rate"], 4)
    return {
        "run_id": run_id,
        "mode": "controlled_context_pressure_ab",
        "per_arm": per_arm,
        "pair_count": len(pairs),
        "comparison": {
            "quality_delta_high_minus_low": quality_delta,
            "higher_pressure_observed": fill_increased,
            "quality_degradation_observed": quality_delta < 0,
            "supports_pressure_quality_association": fill_increased
            and quality_delta < 0,
            "interpretation": "descriptive_association_not_causal_fatigue_measurement",
        },
        "total_cost_usd": round(
            sum(float(item.get("cost_usd", 0.0)) for item in records), 6
        ),
        "contains_request_content": False,
        "contains_response_content": False,
        "contains_reasoning": False,
    }


def summarize_context_stability_records(
    records: Sequence[dict[str, Any]],
    *,
    run_id: str = "",
) -> dict[str, Any]:
    if len(records) < 2:
        raise ValueError("context stability records require at least 2 repeats")
    if any(str(item.get("arm") or "") != ARM_SAME_LOAD for item in records):
        raise ValueError("context stability records must all use same_load")
    targets = {int(item.get("target_input_tokens", 0)) for item in records}
    positions = {str(item.get("needle_position") or "") for item in records}
    case_ids = {str(item.get("case_id") or "") for item in records}
    if len(targets) != 1 or len(positions) != 1 or len(case_ids) != 1:
        raise ValueError("context stability records must describe one identical prompt")
    correctness = [bool(item.get("correct")) for item in records]
    parse_status = [bool(item.get("parse_ok")) for item in records]
    error_status = [bool(item.get("error_observed")) for item in records]
    input_tokens = [int(item.get("input_tokens", 0)) for item in records]
    latencies = [float(item.get("latency_ms", 0.0)) for item in records]
    correctness_changed = len(set(correctness)) > 1
    parse_status_changed = len(set(parse_status)) > 1
    error_status_changed = len(set(error_status)) > 1
    input_tokens_changed = len(set(input_tokens)) > 1
    instability = any(
        (
            correctness_changed,
            parse_status_changed,
            error_status_changed,
            input_tokens_changed,
        )
    )
    all_calls_failed = all(error_status)
    baseline_usable = (
        all(correctness) and all(parse_status) and not any(error_status)
    )
    if all_calls_failed:
        status = "infrastructure_failed"
    elif instability:
        status = "unstable"
    elif not baseline_usable:
        status = "unusable_baseline"
    else:
        status = "pass"
    return {
        "run_id": run_id,
        "mode": "same_load_order_stability_control",
        "repeat_count": len(records),
        "correct_count": sum(int(value) for value in correctness),
        "correct_rate": round(sum(int(value) for value in correctness) / len(records), 4),
        "parse_failure_count": sum(int(not value) for value in parse_status),
        "error_count": sum(int(value) for value in error_status),
        "input_tokens": {
            "min": min(input_tokens),
            "max": max(input_tokens),
            "spread": max(input_tokens) - min(input_tokens),
        },
        "latency_ms": {
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
            "spread": round(max(latencies) - min(latencies), 2),
        },
        "signals": {
            "correctness_changed": correctness_changed,
            "parse_status_changed": parse_status_changed,
            "error_status_changed": error_status_changed,
            "input_tokens_changed": input_tokens_changed,
        },
        "order_instability_observed": instability,
        "all_calls_failed": all_calls_failed,
        "baseline_usable": baseline_usable,
        "status": status,
        "control_passed": status == "pass",
        "interpretation": "same_prompt_repeatability_not_context_quality_effect",
        "total_cost_usd": round(
            sum(float(item.get("cost_usd", 0.0)) for item in records), 6
        ),
        "contains_request_content": False,
        "contains_response_content": False,
        "contains_reasoning": False,
    }


def render_context_pressure_eval_summary(report: dict[str, Any]) -> str:
    per_arm = report.get("per_arm") or {}
    comparison = report.get("comparison") or {}
    lines = [
        f"Context pressure A/B: {report.get('run_id') or '-'}",
    ]
    for arm in CONTEXT_PRESSURE_ARMS:
        metrics = per_arm.get(arm) or {}
        lines.append(
            f"  {arm}: correct={metrics.get('correct')}/{metrics.get('runs')} "
            f"rate={metrics.get('correct_rate')} "
            f"mean_input={metrics.get('mean_input_tokens')} "
            f"window_fill={metrics.get('mean_actual_window_fill_ratio')} "
            f"cost=${float(metrics.get('cost_usd') or 0.0):.6f}"
        )
    lines.append(
        "  high-low quality delta="
        f"{comparison.get('quality_delta_high_minus_low')} "
        "association="
        f"{comparison.get('supports_pressure_quality_association')}"
    )
    lines.append(
        "  interpretation="
        f"{comparison.get('interpretation') or 'descriptive_only'}"
    )
    return "\n".join(lines)


def render_context_stability_summary(report: dict[str, Any]) -> str:
    tokens = report.get("input_tokens") or {}
    latency = report.get("latency_ms") or {}
    return "\n".join(
        [
            f"Context stability control: {report.get('run_id') or '-'}",
            (
                f"  correct={report.get('correct_count')}/{report.get('repeat_count')} "
                f"rate={report.get('correct_rate')}"
            ),
            (
                f"  input_tokens={tokens.get('min')}..{tokens.get('max')} "
                f"spread={tokens.get('spread')}"
            ),
            (
                f"  latency_ms={latency.get('min')}..{latency.get('max')} "
                f"spread={latency.get('spread')}"
            ),
            (
                f"  order_instability={report.get('order_instability_observed')} "
                f"control_passed={report.get('control_passed')} "
                f"status={report.get('status')}"
            ),
            f"  interpretation={report.get('interpretation')}",
        ]
    )
