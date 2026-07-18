from __future__ import annotations

import asyncio
import json
import re

import pytest

from brainregion.cli import build_parser
from brainregion.eval.context_interference import (
    ARM_CLEAN_MEMORY,
    ARM_INTERFERENCE_MEMORY,
    run_context_interference_eval,
    summarize_context_interference_records,
)
from brainregion.providers.base import ModelResponse


class _InterferenceBackend:
    def __init__(self, *, fail_interference: bool = False, always_wrong: bool = False):
        self.fail_interference = fail_interference
        self.always_wrong = always_wrong
        self.calls: list[dict[str, object]] = []

    async def complete(self, *, system, user, model, endpoint_id=None, **kwargs):
        del kwargs, endpoint_id
        active = re.search(
            r"id=(mem-[0-9a-f]+) \| case=\S+ \| status=active \| "
            r"version=7 \| confidence=verified \| answer=(BR-[0-9a-f]+)",
            user,
        )
        assert active is not None
        target = re.search(r"TARGET case=(case-[0-9a-f]+)", user)
        assert target is not None
        is_interference = user.count(f"case={target.group(1)} ") > 1
        input_tokens = max(1, (len(system) + len(user)) // 4)
        self.calls.append(
            {"interference": is_interference, "input_tokens": input_tokens}
        )
        wrong = self.always_wrong or (self.fail_interference and is_interference)
        answer = "wrong" if wrong else active.group(2)
        evidence_id = "wrong" if wrong else active.group(1)
        return ModelResponse(
            model=model,
            content=json.dumps({"answer": answer, "evidence_id": evidence_id}),
            usage={"input_tokens": input_tokens, "output_tokens": 12},
            cost_usd=0.002,
            cost_source="test",
        )


def test_context_interference_eval_detects_semantic_degradation_at_matched_load():
    backend = _InterferenceBackend(fail_interference=True)

    report = asyncio.run(
        run_context_interference_eval(
            backend,
            "expert-model",
            endpoint_id="relay",
            repeats=2,
            target_input_tokens=1000,
            max_total_probe_tokens=5000,
            run_id="interference-run",
        )
    )

    assert report["pair_count"] == 2
    assert report["per_arm"][ARM_CLEAN_MEMORY]["joint_correct_rate"] == 1.0
    assert report["per_arm"][ARM_INTERFERENCE_MEMORY]["joint_correct_rate"] == 0.0
    assert report["comparison"]["load_matched"] is True
    assert report["comparison"]["semantic_interference_observed"] is True
    assert report["execution"]["counterbalanced_order"] is True
    assert len(backend.calls) == 4
    public = json.dumps(report)
    assert "status=stale" not in public
    assert "BR-" not in public
    assert "mem-" not in public


def test_context_interference_eval_does_not_claim_effect_without_usable_baseline():
    backend = _InterferenceBackend(always_wrong=True)

    report = asyncio.run(
        run_context_interference_eval(
            backend,
            "expert-model",
            repeats=2,
            target_input_tokens=800,
            max_total_probe_tokens=4000,
            run_id="wrong-baseline",
        )
    )

    assert report["baseline_usable"] is False
    assert report["comparison"]["semantic_interference_observed"] is False


def test_context_interference_eval_guards_budget_before_model_calls():
    backend = _InterferenceBackend()

    with pytest.raises(ValueError, match="planned interference input exceeds"):
        asyncio.run(
            run_context_interference_eval(
                backend,
                "expert-model",
                repeats=3,
                target_input_tokens=1000,
                max_total_probe_tokens=5000,
            )
        )

    assert backend.calls == []


def test_context_interference_summary_rejects_incomplete_pairs():
    with pytest.raises(ValueError, match="incomplete context interference pairs"):
        summarize_context_interference_records(
            [
                {
                    "arm": ARM_CLEAN_MEMORY,
                    "repeat": 0,
                    "joint_correct": True,
                    "answer_correct": True,
                    "evidence_correct": True,
                    "parse_ok": True,
                    "input_tokens": 100,
                }
            ]
        )


def test_context_interference_cli_defaults_are_bounded():
    args = build_parser().parse_args(
        ["sandbox", "context-interference-eval", "--main-brain", "relay/model"]
    )

    assert args.sandbox_command == "context-interference-eval"
    assert args.target_input_tokens == 4000
    assert args.repeats == 3
    assert args.max_total_probe_tokens == 30000
    assert args.load_match_tolerance == 0.05
