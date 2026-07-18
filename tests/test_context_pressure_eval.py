from __future__ import annotations

import asyncio
import json
import re

import pytest

from brainregion.cli import build_parser
from brainregion.eval.context_pressure import (
    ARM_HIGH_LOAD,
    ARM_LOW_LOAD,
    ARM_SAME_LOAD,
    run_context_pressure_eval,
    run_context_stability_control,
    summarize_context_pressure_records,
    summarize_context_stability_records,
)
from brainregion.providers.base import ModelResponse
from brainregion.sandbox.cli import (
    _context_window_for_model,
    _optional_context_window_for_model,
)


class _LoadSensitiveBackend:
    def __init__(self, *, degrade_above_tokens: int) -> None:
        self.degrade_above_tokens = degrade_above_tokens
        self.calls: list[int] = []

    async def complete(self, *, system, user, model, endpoint_id=None, **kwargs):
        del kwargs, endpoint_id
        input_tokens = max(1, (len(system) + len(user)) // 4)
        self.calls.append(input_tokens)
        match = re.search(r"AUTHORITATIVE_RECORD case=\S+ answer=(BR-[0-9a-f]+)", user)
        assert match is not None
        answer = match.group(1) if input_tokens < self.degrade_above_tokens else "wrong"
        return ModelResponse(
            model=model,
            content=json.dumps({"answer": answer}),
            usage={"input_tokens": input_tokens, "output_tokens": 8},
            cost_usd=0.001,
            cost_source="test",
        )


class _SequenceBackend:
    def __init__(self, correctness: list[bool]) -> None:
        self.correctness = correctness
        self.calls = 0

    async def complete(self, *, system, user, model, endpoint_id=None, **kwargs):
        del kwargs, endpoint_id
        match = re.search(r"AUTHORITATIVE_RECORD case=\S+ answer=(BR-[0-9a-f]+)", user)
        assert match is not None
        answer = match.group(1) if self.correctness[self.calls] else "wrong"
        self.calls += 1
        input_tokens = max(1, (len(system) + len(user)) // 4)
        return ModelResponse(
            model=model,
            content=json.dumps({"answer": answer}),
            usage={"input_tokens": input_tokens, "output_tokens": 8},
            cost_usd=0.001,
            cost_source="test",
        )


class _FailedBackend:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, *, model, **kwargs):
        del kwargs
        self.calls += 1
        return ModelResponse(model=model, error="provider unavailable")


def test_context_pressure_eval_detects_matched_quality_degradation():
    backend = _LoadSensitiveBackend(degrade_above_tokens=500)

    report = asyncio.run(
        run_context_pressure_eval(
            backend,
            "expert-model",
            endpoint_id="relay",
            context_window_tokens=1000,
            repeats=2,
            low_fill_ratio=0.1,
            high_fill_ratio=0.8,
            needle_positions=("early",),
            max_probe_tokens=1000,
            run_id="stable-run",
        )
    )

    assert report["pair_count"] == 2
    assert report["per_arm"][ARM_LOW_LOAD]["correct_rate"] == 1.0
    assert report["per_arm"][ARM_HIGH_LOAD]["correct_rate"] == 0.0
    assert report["comparison"] == {
        "quality_delta_high_minus_low": -1.0,
        "higher_pressure_observed": True,
        "quality_degradation_observed": True,
        "supports_pressure_quality_association": True,
        "interpretation": "descriptive_association_not_causal_fatigue_measurement",
    }
    assert report["execution"]["counterbalanced_order"] is True
    assert report["execution"]["target_capped"] == {
        ARM_LOW_LOAD: False,
        ARM_HIGH_LOAD: False,
    }
    assert report["execution"]["actual_total_input_tokens"] == sum(backend.calls)
    assert report["execution"]["actual_total_exceeds_target_plan"] is True
    assert report["execution"]["actual_total_exceeds_configured_guard"] is False
    assert report["execution"]["calls_exceeding_arm_target"] == 4
    assert report["execution"]["provider_capacity_verified_at_runtime"] is False
    assert report["execution"]["input_cap_interpretation"].startswith(
        "synthetic_target_cap"
    )
    assert report["context_pressure"]["model_capacity_coverage_rate"] == 1.0
    assert len(backend.calls) == 4
    public = json.dumps(report)
    assert "AUTHORITATIVE_RECORD" not in public
    assert "BR-" not in public


def test_context_pressure_eval_rejects_collapsed_arms_before_model_calls():
    backend = _LoadSensitiveBackend(degrade_above_tokens=500)

    with pytest.raises(ValueError, match="collapses low/high arms"):
        asyncio.run(
            run_context_pressure_eval(
                backend,
                "expert-model",
                context_window_tokens=1000,
                low_fill_ratio=0.1,
                high_fill_ratio=0.7,
                max_probe_tokens=100,
            )
        )

    assert backend.calls == []


def test_context_pressure_eval_enforces_planned_input_guard_before_calls():
    backend = _LoadSensitiveBackend(degrade_above_tokens=500)

    with pytest.raises(ValueError, match="planned probe input exceeds"):
        asyncio.run(
            run_context_pressure_eval(
                backend,
                "expert-model",
                context_window_tokens=1000,
                repeats=2,
                low_fill_ratio=0.1,
                high_fill_ratio=0.8,
                needle_positions=("early", "middle"),
                max_total_probe_tokens=1000,
            )
        )

    assert backend.calls == []


def test_context_pressure_summary_rejects_incomplete_pairs():
    with pytest.raises(ValueError, match="incomplete context pressure pairs"):
        summarize_context_pressure_records(
            [
                {
                    "arm": ARM_LOW_LOAD,
                    "repeat": 0,
                    "needle_position": "middle",
                    "correct": True,
                }
            ]
        )


def test_context_pressure_eval_cli_defaults_are_bounded_and_explicit():
    args = build_parser().parse_args(
        ["sandbox", "context-pressure-eval", "--main-brain", "relay/model"]
    )

    assert args.sandbox_command == "context-pressure-eval"
    assert args.context_window_tokens is None
    assert args.repeats == 2
    assert args.low_fill_ratio == 0.05
    assert args.high_fill_ratio == 0.25
    assert args.needle_positions == "middle"
    assert args.max_probe_tokens == 32000
    assert args.max_total_probe_tokens == 100000


def test_context_pressure_cli_resolves_capacity_from_exact_endpoint_profile():
    defaults = {
        "endpoints": {
            "relay": {
                "models": [
                    {"id": "model", "context_window_tokens": 128000},
                ]
            }
        }
    }

    limit, source = _context_window_for_model(
        defaults,
        model="model",
        endpoint_id="relay",
        override=None,
    )

    assert limit == 128000
    assert source == "endpoint_model"


def test_context_pressure_cli_requires_capacity_or_explicit_override():
    defaults = {"endpoints": {"relay": {"models": ["model"]}}}

    with pytest.raises(ValueError, match="no verified context_window_tokens"):
        _context_window_for_model(
            defaults,
            model="model",
            endpoint_id="relay",
            override=None,
        )

    assert _context_window_for_model(
        defaults,
        model="model",
        endpoint_id="relay",
        override=64000,
    ) == (64000, "cli_override")


def test_context_stability_control_passes_for_identical_stable_responses():
    backend = _SequenceBackend([True, True, True])

    report = asyncio.run(
        run_context_stability_control(
            backend,
            "expert-model",
            endpoint_id="relay",
            target_input_tokens=100,
            repeats=3,
            max_total_probe_tokens=1000,
            run_id="stable-control",
        )
    )

    assert report["mode"] == "same_load_order_stability_control"
    assert report["control_passed"] is True
    assert report["status"] == "pass"
    assert report["order_instability_observed"] is False
    assert report["correct_rate"] == 1.0
    assert report["input_tokens"]["spread"] == 0
    assert report["execution"]["prompt_policy"] == (
        "identical_prompt_across_repeats"
    )
    assert report["context_pressure"]["model_capacity_coverage_rate"] == 0.0
    public = json.dumps(report)
    assert "AUTHORITATIVE_RECORD" not in public
    assert "BR-" not in public


def test_context_stability_control_detects_order_correctness_drift():
    backend = _SequenceBackend([True, False, True])

    report = asyncio.run(
        run_context_stability_control(
            backend,
            "expert-model",
            target_input_tokens=100,
            repeats=3,
            max_total_probe_tokens=1000,
            run_id="drift-control",
        )
    )

    assert report["correct_rate"] == pytest.approx(2 / 3, abs=0.0001)
    assert report["signals"]["correctness_changed"] is True
    assert report["order_instability_observed"] is True
    assert report["control_passed"] is False
    assert report["status"] == "unstable"


def test_context_stability_control_rejects_consistent_infrastructure_failure():
    backend = _FailedBackend()

    report = asyncio.run(
        run_context_stability_control(
            backend,
            "expert-model",
            target_input_tokens=100,
            repeats=3,
            max_total_probe_tokens=1000,
            run_id="failed-control",
        )
    )

    assert backend.calls == 3
    assert report["error_count"] == 3
    assert report["all_calls_failed"] is True
    assert report["order_instability_observed"] is False
    assert report["baseline_usable"] is False
    assert report["control_passed"] is False
    assert report["status"] == "infrastructure_failed"


def test_context_stability_control_rejects_repeatable_but_unusable_baseline():
    backend = _SequenceBackend([False, False, False])

    report = asyncio.run(
        run_context_stability_control(
            backend,
            "expert-model",
            target_input_tokens=100,
            repeats=3,
            max_total_probe_tokens=1000,
            run_id="unusable-control",
        )
    )

    assert report["correct_rate"] == 0.0
    assert report["order_instability_observed"] is False
    assert report["baseline_usable"] is False
    assert report["control_passed"] is False
    assert report["status"] == "unusable_baseline"


def test_context_stability_control_rejects_costly_plan_before_calls():
    backend = _SequenceBackend([True, True, True])

    with pytest.raises(ValueError, match="planned stability input exceeds"):
        asyncio.run(
            run_context_stability_control(
                backend,
                "expert-model",
                target_input_tokens=1000,
                repeats=3,
                max_total_probe_tokens=2000,
            )
        )

    assert backend.calls == 0


def test_context_stability_summary_requires_one_identical_prompt():
    with pytest.raises(ValueError, match="one identical prompt"):
        summarize_context_stability_records(
            [
                {
                    "arm": ARM_SAME_LOAD,
                    "target_input_tokens": 100,
                    "needle_position": "early",
                    "case_id": "a",
                },
                {
                    "arm": ARM_SAME_LOAD,
                    "target_input_tokens": 100,
                    "needle_position": "late",
                    "case_id": "b",
                },
            ]
        )


def test_context_stability_cli_defaults_do_not_require_capacity():
    args = build_parser().parse_args(
        ["sandbox", "context-stability-control", "--main-brain", "relay/model"]
    )

    assert args.sandbox_command == "context-stability-control"
    assert args.target_input_tokens == 2000
    assert args.repeats == 3
    assert args.max_total_probe_tokens == 10000
    assert args.context_window_tokens is None
    assert _optional_context_window_for_model(
        {"endpoints": {"relay": {"models": ["model"]}}},
        model="model",
        endpoint_id="relay",
        override=None,
    ) == (None, "unknown")
