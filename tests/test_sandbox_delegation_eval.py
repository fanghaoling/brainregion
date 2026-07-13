"""Fixture-backed main/single/multi delegation evaluation tests."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from brainregion.cli import build_parser
from brainregion.providers.base import ModelResponse
from brainregion.sandbox import cleanup_run_dir, make_run_dir, materialize_fixture
from brainregion.sandbox.cli import _parse_delegation_experts
from brainregion.sandbox.delegation_eval import (
    SandboxExpertSpec,
    run_fixture_delegation_eval,
)
from brainregion.sandbox.fixtures import get_fixture
from brainregion.sandbox.loop import parse_tool_call
from brainregion.workspace import read_text
from brainregion.workspace.files import scoped_workspace_root


def _response(model: str, content: dict, *, cost: float, tokens: int) -> ModelResponse:
    return ModelResponse(
        model=model,
        content=json.dumps(content),
        usage={
            "input_tokens": max(0, tokens - 5),
            "output_tokens": 5,
            "total_tokens": tokens,
        },
        cost_usd=cost,
        cost_source="provider",
    )


class _DelegationBackend:
    def __init__(self, source_sha: str) -> None:
        self.source_sha = source_sha
        self.expert_calls: list[dict] = []
        self.main_initial_users: list[str] = []
        self.main_calls = 0

    async def complete(self, **kwargs):
        self.expert_calls.append(kwargs)
        region = kwargs["user"].split("REGION: ", 1)[1].splitlines()[0]
        return _response(
            kwargs["model"],
            {
                "state": "done",
                "summary": f"The {region} expert found a bounded off-by-one defect.",
                "implication": "The end value is excluded by the current range.",
                "recommended_action": "Include the end value and run the configured tests.",
                "uncertainty": "Low; objective tests still decide acceptance.",
                "evidence_refs": [],
                "decision_scope": "routine",
                "risk": "low",
                "memory_impact": "supporting",
                "reversible": True,
                "repeated_failure": False,
                "requires_user_choice": False,
                "needs_more_context": False,
                "covered_scope": region,
                "unresolved_questions": [],
                "conflicts_with": [],
                "recommended_followups": ["Run pytest."],
            },
            cost=0.01,
            tokens=15,
        )

    async def complete_messages(self, messages, **kwargs):
        self.main_calls += 1
        initial_user = messages[1]["content"]
        assistant_turns = sum(message["role"] == "assistant" for message in messages)
        if assistant_turns == 0:
            self.main_initial_users.append(initial_user)
            content = {
                "thought": "Apply the minimal bounded fix.",
                "tool": "apply_text_patch",
                "args": {
                    "path": "ranges.py",
                    "expected_sha256": self.source_sha,
                    "replacements": [
                        {
                            "old_text": "range(start, end)",
                            "new_text": "range(start, end + 1)",
                        }
                    ],
                    "dry_run": False,
                },
            }
        elif assistant_turns == 1:
            content = {
                "thought": "Verify the fixture.",
                "tool": "workspace_run_check",
                "args": {"argv": [sys.executable, "-m", "pytest", "-q"]},
            }
        else:
            adopted = [
                assignment_id
                for assignment_id in ("debugging", "review")
                if f'"assignment_id":"{assignment_id}"' in initial_user
            ]
            content = {
                "thought": "The objective tests pass.",
                "done": True,
                "answer": "Applied and verified the fix.",
                "adopted_assignment_ids": adopted,
            }
        return _response(kwargs["model"], content, cost=0.001, tokens=20)


class _TriggeredDelegationBackend(_DelegationBackend):
    async def complete_messages(self, messages, **kwargs):
        self.main_calls += 1
        assistant_turns = sum(message["role"] == "assistant" for message in messages)
        if assistant_turns == 0:
            self.main_initial_users.append(messages[1]["content"])
        has_reports = any(
            message["role"] == "user" and "<expert_reports>" in message["content"]
            for message in messages
        )
        if not has_reports:
            if assistant_turns % 2 == 0:
                content = {
                    "thought": "Locate the relevant range implementation.",
                    "tool": "search_text",
                    "args": {"query": "sum_range", "include_globs": ["*.py"]},
                }
            else:
                content = {
                    "thought": "Read the candidate implementation.",
                    "tool": "read_text",
                    "args": {"path": "ranges.py"},
                }
        elif assistant_turns == 2:
            content = {
                "thought": "Use the expert diagnosis to apply the bounded fix.",
                "tool": "apply_text_patch",
                "args": {
                    "path": "ranges.py",
                    "expected_sha256": self.source_sha,
                    "replacements": [
                        {
                            "old_text": "range(start, end)",
                            "new_text": "range(start, end + 1)",
                        }
                    ],
                    "dry_run": False,
                },
            }
        elif assistant_turns == 3:
            content = {
                "thought": "Verify the fixture.",
                "tool": "workspace_run_check",
                "args": {"argv": [sys.executable, "-m", "pytest", "-q"]},
            }
        else:
            content = {
                "thought": "The objective tests pass.",
                "done": True,
                "answer": "Applied and verified the expert-guided fix.",
                "adopted_assignment_ids": ["debugging"],
            }
        return _response(kwargs["model"], content, cost=0.001, tokens=20)


def _experts() -> list[SandboxExpertSpec]:
    return [
        SandboxExpertSpec(
            assignment_id="debugging",
            region="debugging",
            question="Find the root cause and minimal fix.",
            model="expert-debug",
        ),
        SandboxExpertSpec(
            assignment_id="review",
            region="review",
            question="Independently review correctness and regression risk.",
            model="expert-review",
        ),
    ]


def _materialized_sha(task, path: str) -> str:
    run_dir = make_run_dir(prefix="brainregion-delegation-probe-")
    materialize_fixture(task, Path(run_dir))
    try:
        with scoped_workspace_root(run_dir):
            return read_text(path)["sha256"]
    finally:
        cleanup_run_dir(run_dir)


def test_fixture_delegation_eval_runs_fresh_arms_and_reuses_matched_experts():
    task = get_fixture("off_by_one")
    source_sha = _materialized_sha(task, "ranges.py")
    backend = _DelegationBackend(source_sha)

    report = asyncio.run(
        run_fixture_delegation_eval(
            backend,
            "main-model",
            [task],
            _experts(),
            repeats=2,
            max_steps=5,
            max_cost_usd=1.0,
            bootstrap_samples=50,
        )
    )

    assert report["n_runs"] == 6
    assert all(summary["solve_rate"] == 1.0 for summary in report["per_arm"].values())
    assert all(summary["protocol_completion_rate"] == 1.0 for summary in report["per_arm"].values())
    assert report["per_arm"]["single_expert"]["report_adoption_rate"] == 1.0
    assert report["per_arm"]["multi_expert"]["report_adoption_rate"] == 1.0
    assert report["execution"]["actual_main_runs"] == 6
    assert report["execution"]["actual_main_model_calls"] == 18
    assert report["execution"]["actual_expert_model_calls"] == 4
    assert report["execution"]["expert_cache_entries"] == 4
    assert report["execution"]["actual_expert_cost_usd"] == pytest.approx(0.04)
    assert report["execution"]["contains_trajectories"] is False
    assert report["contains_reasoning"] is False
    assert report["shadow_gates"]["models_called"] is False
    assert len(backend.expert_calls) == 4
    assert backend.main_calls == 18
    assert all("<expert_reports>" not in call["user"] for call in backend.expert_calls)
    assert sum("<expert_reports>" in user for user in backend.main_initial_users) == 4
    assert sum("<expert_reports>" not in user for user in backend.main_initial_users) == 2
    multi_cases = [case for case in report["cases"] if case["arm"] == "multi_expert"]
    assert all(case["main_result"]["adopted_assignment_ids"] == ["debugging", "review"] for case in multi_cases)
    assert all(case["main_result"]["termination_reason"] == "done" for case in report["cases"])
    assert all(
        case["main_result"]["sandbox_diagnostics"]["tool_sequence"]
        == ["apply_text_patch", "workspace_run_check", "done"]
        for case in report["cases"]
    )
    assert all(case["main_result"]["sandbox_diagnostics"]["contains_reasoning"] is False for case in report["cases"])
    for case in report["cases"]:
        progress = case["main_result"]["sandbox_diagnostics"]["progress_trace"]
        assert [step["operation"] for step in progress] == [
            "apply_text_patch",
            "workspace_run_check",
            "done",
        ]
        assert progress[0]["workspace_effect"] is True
        assert progress[1]["verification_passed"] is True
        assert all("thought" not in step and "result" not in step and "args" not in step for step in progress)


def test_done_call_adoption_contract_is_strict_and_deduplicated():
    call, error = parse_tool_call(
        json.dumps(
            {
                "done": True,
                "answer": "ok",
                "adopted_assignment_ids": ["debugging", "debugging", "review"],
            }
        )
    )
    assert error is None
    assert call is not None and call.adopted_assignment_ids == ("debugging", "review")

    for invalid_ids in ("debugging", None, {}):
        call, error = parse_tool_call(json.dumps({"done": True, "adopted_assignment_ids": invalid_ids}))
        assert call is None
        assert error is not None and "must be an array" in error


def test_triggered_fixture_arm_wakes_after_observed_stall_and_then_injects_report():
    task = get_fixture("off_by_one")
    source_sha = _materialized_sha(task, "ranges.py")
    backend = _TriggeredDelegationBackend(source_sha)

    report = asyncio.run(
        run_fixture_delegation_eval(
            backend,
            "main-model",
            [task],
            _experts(),
            arms=["main_only", "triggered_single_expert"],
            max_steps=5,
            max_cost_usd=1.0,
            bootstrap_samples=20,
        )
    )

    assert report["per_arm"]["main_only"]["solve_rate"] == 0.0
    triggered = report["per_arm"]["triggered_single_expert"]
    assert triggered["solve_rate"] == 1.0
    assert triggered["expert_activation_rate"] == 1.0
    assert triggered["report_adoption_rate"] == 1.0
    assert len(backend.expert_calls) == 1
    assert all("<expert_reports>" not in user for user in backend.main_initial_users)

    case = next(case for case in report["cases"] if case["arm"] == "triggered_single_expert")
    diagnostics = case["main_result"]["sandbox_diagnostics"]
    assert diagnostics["delegation_trigger"] == {
        "activated": True,
        "step": 2,
        "reason": "no_workspace_effect",
        "signals": ["no_workspace_effect"],
        "assignment_ids": ["debugging"],
        "reports_available": 1,
        "contains_reasoning": False,
    }
    assert diagnostics["advisory_injections"][0]["step"] == 2
    assert diagnostics["advisory_injections"][0]["contains_advice"] is False
    assert diagnostics["progress_trace"][0]["target_is_new"] is True
    assert diagnostics["progress_trace"][2]["workspace_effect"] is True
    assert diagnostics["tool_sequence"] == [
        "search_text",
        "read_text",
        "apply_text_patch",
        "workspace_run_check",
        "done",
    ]


def test_fixture_delegation_rejects_duplicate_experts_before_model_calls():
    task = get_fixture("off_by_one")
    duplicate = [*_experts(), _experts()[0]]

    with pytest.raises(ValueError, match="assignment ids must be unique"):
        asyncio.run(
            run_fixture_delegation_eval(
                _DelegationBackend("unused"),
                "main",
                [task],
                duplicate,
            )
        )


def test_fixture_delegation_marks_provider_failure_as_invalid_run():
    class FailingBackend:
        async def complete_messages(self, messages, **kwargs):
            return ModelResponse(model=kwargs["model"], error="quota exceeded")

        async def complete(self, **kwargs):
            raise AssertionError("main_only must not call an expert")

    report = asyncio.run(
        run_fixture_delegation_eval(
            FailingBackend(),
            "main-model",
            [get_fixture("off_by_one")],
            _experts(),
            arms=["main_only"],
            max_steps=10,
            consecutive_error_limit=3,
        )
    )

    summary = report["per_arm"]["main_only"]
    main_result = report["cases"][0]["main_result"]
    assert summary["n_valid_runs"] == 0
    assert summary["solve_rate"] is None
    assert summary["infrastructure_failures"] == 1
    assert main_result["infrastructure_error"] is True
    assert main_result["termination_reason"] == "model_error"
    assert main_result["error"] == "sandbox_model_error"


def test_delegation_cli_parses_repeatable_endpoint_qualified_experts():
    args = build_parser().parse_args(
        [
            "sandbox",
            "delegation-eval",
            "--tasks",
            "off_by_one",
            "--main-brain",
            "main-model",
            "--expert",
            "debugging=modelbridge_anthropic/claude-opus-4-8",
            "--expert",
            "review=modelbridge_openai/gpt-5.5",
            "--repeats",
            "2",
        ]
    )

    assert args.sandbox_command == "delegation-eval"
    assert args.expert == [
        "debugging=modelbridge_anthropic/claude-opus-4-8",
        "review=modelbridge_openai/gpt-5.5",
    ]
    assert args.repeats == 2


def test_cli_expert_parser_preserves_resolved_endpoint_identity():
    dd = {
        "endpoints": {
            "modelbridge_anthropic": {"provider": "anthropic"},
        }
    }
    experts = _parse_delegation_experts(
        ["debugging=modelbridge_anthropic/claude-opus-4-8"],
        {"modelbridge_anthropic": object()},
        dd,
    )

    assert experts[0].model == "claude-opus-4-8"
    assert experts[0].endpoint_id == "modelbridge_anthropic"

    same_region = _parse_delegation_experts(
        ["review_opus:review=modelbridge_anthropic/claude-opus-4-8"],
        {"modelbridge_anthropic": object()},
        dd,
    )
    assert same_region[0].assignment_id == "review_opus"
    assert same_region[0].region == "review"
