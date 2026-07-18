"""Content-free shadow telemetry for region-model context pressure."""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Literal

from brainregion.runtime import emit_event, normalize_usage

ContextPressureBand = Literal["normal", "loaded", "strained", "saturated"]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _bounded_text(value: Any, name: str, *, max_length: int = 200) -> str:
    text = str(value or "").strip()
    if len(text) > max_length:
        raise ValueError(f"{name} cannot exceed {max_length} characters")
    return text


def _ratio(value: int, limit: int) -> float:
    if limit <= 0:
        return 0.0
    return round(min(1.0, max(0.0, value / limit)), 4)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _band(score: float) -> ContextPressureBand:
    if score >= 0.8:
        return "saturated"
    if score >= 0.6:
        return "strained"
    if score >= 0.35:
        return "loaded"
    return "normal"


@dataclass(frozen=True)
class ContextPressureSample:
    index: int
    step: int
    task_id: str
    assignment_id: str
    region: str
    model: str
    endpoint_id: str | None
    context_tokens: int
    context_budget_tokens: int
    context_fill_ratio: float
    blocks_loaded: int
    block_budget: int
    block_fill_ratio: float
    context_truncated: bool
    input_tokens: int
    output_tokens: int
    model_context_limit_tokens: int | None
    model_window_fill_ratio: float | None
    input_growth_ratio: float | None
    attempt: int
    model_called: bool
    error_observed: bool
    context_status: str
    lifecycle_state: str
    pressure_score: float
    pressure_band: ContextPressureBand
    high_pressure_streak: int
    signals: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "step": self.step,
            "task_id": self.task_id,
            "assignment_id": self.assignment_id,
            "region": self.region,
            "model": self.model,
            "endpoint_id": self.endpoint_id,
            "context_tokens": self.context_tokens,
            "context_budget_tokens": self.context_budget_tokens,
            "context_fill_ratio": self.context_fill_ratio,
            "blocks_loaded": self.blocks_loaded,
            "block_budget": self.block_budget,
            "block_fill_ratio": self.block_fill_ratio,
            "context_truncated": self.context_truncated,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_context_limit_tokens": self.model_context_limit_tokens,
            "model_capacity_known": self.model_context_limit_tokens is not None,
            "model_window_fill_ratio": self.model_window_fill_ratio,
            "input_growth_ratio": self.input_growth_ratio,
            "attempt": self.attempt,
            "model_called": self.model_called,
            "error_observed": self.error_observed,
            "context_status": self.context_status,
            "lifecycle_state": self.lifecycle_state,
            "pressure_score": self.pressure_score,
            "pressure_band": self.pressure_band,
            "high_pressure_streak": self.high_pressure_streak,
            "signals": list(self.signals),
            "changes_model_input": False,
            "changes_routing": False,
            "context_content_returned": False,
            "contains_reasoning": False,
        }


@dataclass
class ContextPressureObserver:
    """Observe context saturation without changing execution or routing."""

    max_samples: int = 512
    model_context_limits: dict[str, int] = field(default_factory=dict)
    emit_events: bool = True
    samples: list[ContextPressureSample] = field(default_factory=list)
    _total_observations: int = 0
    _dropped_samples: int = 0
    _previous_input_tokens: dict[tuple[str, str], int] = field(default_factory=dict)
    _high_pressure_streaks: dict[tuple[str, str], int] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.max_samples = _positive_int(self.max_samples, "max_samples")
        normalized: dict[str, int] = {}
        for raw_key, raw_limit in self.model_context_limits.items():
            key = _bounded_text(raw_key, "model context limit key", max_length=400).casefold()
            if not key:
                raise ValueError("model context limit key cannot be empty")
            normalized[key] = _positive_int(raw_limit, "model context limit")
        self.model_context_limits = normalized

    @classmethod
    def from_model_routes(
        cls,
        routes: dict[str, Any],
        **kwargs: Any,
    ) -> ContextPressureObserver:
        """Build capacity lookup from ``list_model_routes``-style output."""

        supplied_limits = kwargs.pop("model_context_limits", None)
        observer = cls(
            model_context_limits=dict(supplied_limits or {}),
            **kwargs,
        )
        observer.register_model_routes(routes, overwrite=False)
        return observer

    def register_model_routes(
        self,
        routes: dict[str, Any],
        *,
        overwrite: bool = False,
    ) -> int:
        """Register verified capacities without retaining the route description."""

        limits: dict[str, int] = {}
        entries = routes.get("resolved_panel") if isinstance(routes, dict) else None
        for entry in entries if isinstance(entries, list) else ():
            if not isinstance(entry, dict):
                continue
            model = str(entry.get("model") or "").strip()
            profile = entry.get("profile")
            if not model or not isinstance(profile, dict):
                continue
            raw_limit = profile.get("context_window_tokens")
            if isinstance(raw_limit, bool):
                continue
            if isinstance(raw_limit, int):
                limit = raw_limit
            elif isinstance(raw_limit, str) and raw_limit.strip().isdigit():
                limit = int(raw_limit.strip())
            else:
                continue
            if limit <= 0:
                continue
            endpoint = str(entry.get("endpoint_id") or "").strip()
            identity = f"{endpoint}/{model}" if endpoint else model
            limits[identity.casefold()] = limit
        registered = 0
        with self._lock:
            for identity, limit in limits.items():
                if not overwrite and identity in self.model_context_limits:
                    continue
                self.model_context_limits[identity] = limit
                registered += 1
        return registered

    def observe(
        self,
        *,
        step: int,
        task_id: str,
        assignment_id: str,
        region: str,
        model: str,
        endpoint_id: str | None = None,
        context_tokens: int = 0,
        context_budget_tokens: int = 0,
        blocks_loaded: int = 0,
        block_budget: int = 0,
        context_truncated: bool = False,
        usage: dict[str, Any] | None = None,
        attempt: int = 1,
        model_called: bool = False,
        error_observed: bool = False,
        context_status: str = "",
        lifecycle_state: str = "",
    ) -> ContextPressureSample:
        step = _nonnegative_int(step, "step")
        context_tokens = _nonnegative_int(context_tokens, "context_tokens")
        context_budget_tokens = _nonnegative_int(
            context_budget_tokens, "context_budget_tokens"
        )
        blocks_loaded = _nonnegative_int(blocks_loaded, "blocks_loaded")
        block_budget = _nonnegative_int(block_budget, "block_budget")
        attempt = _positive_int(attempt, "attempt")
        task_id = _bounded_text(task_id, "task_id")
        assignment_id = _bounded_text(assignment_id, "assignment_id")
        region = _bounded_text(region, "region").casefold()
        model = _bounded_text(model, "model", max_length=300)
        if not task_id or not assignment_id or not region or not model:
            raise ValueError("task_id, assignment_id, region, and model are required")
        endpoint = _bounded_text(endpoint_id, "endpoint_id") or None
        context_status = _bounded_text(
            context_status, "context_status", max_length=80
        )
        lifecycle_state = _bounded_text(
            lifecycle_state, "lifecycle_state", max_length=80
        )
        normalized_usage = normalize_usage(usage or {})
        input_tokens = normalized_usage["input_tokens"]
        output_tokens = normalized_usage["output_tokens"]
        identity = f"{endpoint}/{model}" if endpoint else model
        aggregate_key = (region, identity.casefold())
        context_fill = _ratio(context_tokens, context_budget_tokens)
        block_fill = _ratio(blocks_loaded, block_budget)
        model_limit = self.model_context_limits.get(identity.casefold())
        if model_limit is None:
            model_limit = self.model_context_limits.get(model.casefold())
        model_fill = (
            _ratio(input_tokens, model_limit)
            if model_limit is not None and model_called
            else None
        )

        with self._lock:
            previous_input = self._previous_input_tokens.get(aggregate_key)
            input_growth = (
                round(input_tokens / previous_input, 4)
                if model_called
                and input_tokens > 0
                and previous_input is not None
                and previous_input > 0
                else None
            )
            if model_called and input_tokens > 0:
                self._previous_input_tokens[aggregate_key] = input_tokens

            score = (
                0.35 * context_fill
                + 0.15 * block_fill
                + 0.20 * int(bool(context_truncated))
                + 0.10 * int(attempt > 1)
                + 0.10 * int(bool(error_observed))
                + 0.10 * float(model_fill or 0.0)
            )
            score = round(min(1.0, max(0.0, score)), 4)
            band = _band(score)
            previous_streak = self._high_pressure_streaks.get(aggregate_key, 0)
            streak = previous_streak + 1 if band in {"strained", "saturated"} else 0
            self._high_pressure_streaks[aggregate_key] = streak
            signals: list[str] = []
            if context_fill >= 0.8:
                signals.append("context_budget_near_limit")
            if block_fill >= 0.8:
                signals.append("block_budget_near_limit")
            if context_truncated:
                signals.append("context_truncated")
            if model_fill is not None and model_fill >= 0.7:
                signals.append("model_window_near_limit")
            if input_growth is not None and input_growth >= 1.25:
                signals.append("input_tokens_growing")
            if attempt > 1:
                signals.append("repeated_attempt")
            if error_observed:
                signals.append("error_observed")

            sample = ContextPressureSample(
                index=self._total_observations,
                step=step,
                task_id=task_id,
                assignment_id=assignment_id,
                region=region,
                model=model,
                endpoint_id=endpoint,
                context_tokens=context_tokens,
                context_budget_tokens=context_budget_tokens,
                context_fill_ratio=context_fill,
                blocks_loaded=blocks_loaded,
                block_budget=block_budget,
                block_fill_ratio=block_fill,
                context_truncated=bool(context_truncated),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model_context_limit_tokens=model_limit,
                model_window_fill_ratio=model_fill,
                input_growth_ratio=input_growth,
                attempt=attempt,
                model_called=bool(model_called),
                error_observed=bool(error_observed),
                context_status=context_status,
                lifecycle_state=lifecycle_state,
                pressure_score=score,
                pressure_band=band,
                high_pressure_streak=streak,
                signals=tuple(signals),
            )
            self._total_observations += 1
            if len(self.samples) >= self.max_samples:
                self.samples.pop(0)
                self._dropped_samples += 1
            self.samples.append(sample)

        if self.emit_events:
            emit_event(
                "context.pressure_observed",
                task_id=task_id,
                assignment_id=assignment_id,
                region_id=region,
                model=model,
                endpoint_id=endpoint,
                payload=sample.to_dict(),
            )
        return sample

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self.samples)
            total_observations = self._total_observations
            dropped_samples = self._dropped_samples
        grouped: dict[tuple[str, str, str | None], list[ContextPressureSample]] = {}
        for sample in samples:
            grouped.setdefault(
                (sample.region, sample.model, sample.endpoint_id), []
            ).append(sample)
        region_models: list[dict[str, Any]] = []
        for (region, model, endpoint), items in sorted(
            grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or "")
        ):
            context_ratios = [item.context_fill_ratio for item in items]
            window_ratios = [
                item.model_window_fill_ratio
                for item in items
                if item.model_window_fill_ratio is not None
            ]
            latest = items[-1]
            region_models.append(
                {
                    "region": region,
                    "model": model,
                    "endpoint_id": endpoint,
                    "sample_count": len(items),
                    "model_call_count": sum(int(item.model_called) for item in items),
                    "mean_context_fill_ratio": _mean(context_ratios),
                    "peak_context_fill_ratio": max(context_ratios, default=0.0),
                    "mean_model_window_fill_ratio": _mean(window_ratios),
                    "peak_model_window_fill_ratio": (
                        max(window_ratios) if window_ratios else None
                    ),
                    "peak_input_tokens": max(
                        (item.input_tokens for item in items), default=0
                    ),
                    "truncation_count": sum(
                        int(item.context_truncated) for item in items
                    ),
                    "retry_count": sum(int(item.attempt > 1) for item in items),
                    "error_count": sum(int(item.error_observed) for item in items),
                    "peak_pressure_score": max(
                        (item.pressure_score for item in items), default=0.0
                    ),
                    "latest_pressure_score": latest.pressure_score,
                    "latest_pressure_band": latest.pressure_band,
                    "high_pressure_streak": latest.high_pressure_streak,
                    "model_capacity_known": any(
                        item.model_context_limit_tokens is not None for item in items
                    ),
                }
            )
        model_calls = [item for item in samples if item.model_called]
        capacity_known = [
            item for item in model_calls if item.model_context_limit_tokens is not None
        ]
        by_band = {
            band: sum(int(item.pressure_band == band) for item in samples)
            for band in ("normal", "loaded", "strained", "saturated")
        }
        return {
            "enabled": True,
            "mode": "shadow",
            "policy": "context_pressure_shadow_v1",
            "sample_count": len(samples),
            "total_observations": total_observations,
            "dropped_sample_count": dropped_samples,
            "region_model_count": len(region_models),
            "model_call_count": len(model_calls),
            "model_capacity_known_count": len(capacity_known),
            "model_capacity_coverage_rate": (
                len(capacity_known) / len(model_calls) if model_calls else None
            ),
            "high_pressure_sample_count": sum(
                by_band[band] for band in ("strained", "saturated")
            ),
            "by_band": by_band,
            "region_models": region_models,
            "samples": [sample.to_dict() for sample in samples],
            "score_interpretation": "risk_proxy_not_measured_model_fatigue",
            "changes_model_input": False,
            "changes_routing": False,
            "models_called": False,
            "context_content_returned": False,
            "contains_reasoning": False,
        }


def disabled_context_pressure_metrics() -> dict[str, Any]:
    return {
        "enabled": False,
        "mode": "off",
        "policy": "context_pressure_shadow_v1",
        "sample_count": 0,
        "total_observations": 0,
        "dropped_sample_count": 0,
        "region_model_count": 0,
        "model_call_count": 0,
        "model_capacity_known_count": 0,
        "model_capacity_coverage_rate": None,
        "high_pressure_sample_count": 0,
        "by_band": {
            "normal": 0,
            "loaded": 0,
            "strained": 0,
            "saturated": 0,
        },
        "region_models": [],
        "samples": [],
        "score_interpretation": "risk_proxy_not_measured_model_fatigue",
        "changes_model_input": False,
        "changes_routing": False,
        "models_called": False,
        "context_content_returned": False,
        "contains_reasoning": False,
    }


__all__ = [
    "ContextPressureBand",
    "ContextPressureObserver",
    "ContextPressureSample",
    "disabled_context_pressure_metrics",
]
