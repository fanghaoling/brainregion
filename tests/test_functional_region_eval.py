from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from brainregion.cli import build_parser
from brainregion.providers.base import ModelResponse
from brainregion.sandbox.fixtures import get_fixture
from brainregion.sandbox.functional_region_eval import (
    ARM_EVIDENCE_REGION,
    ARM_EVIDENCE_VERIFICATION,
    ARM_INTENT_EVIDENCE_OWNED,
    ARM_MAIN_ONLY,
    ARM_PASSIVE_CONTEXT,
    run_functional_region_eval,
    summarize_functional_region_records,
)
from brainregion.sandbox.isolation import cleanup_run_dir, make_run_dir, materialize_fixture
from brainregion.workspace import read_text
from brainregion.workspace.files import scoped_workspace_root


class _FunctionalBackend:
    def __init__(self, source_sha: str) -> None:
        self.source_sha = source_sha
        self.calls: list[dict] = []

    async def complete_messages(self, messages, **kwargs):
        turn = sum(message["role"] == "assistant" for message in messages)
        rendered = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        has_workbench = "<region_workbench>" in rendered
        auto_verification = "补丁真实落盘后" in messages[0]["content"]
        self.calls.append(
            {
                "turn": turn,
                "has_workbench": has_workbench,
                "auto_verification": auto_verification,
                "messages": rendered,
            }
        )

        if not has_workbench and turn == 0:
            content = {
                "thought": "Read the named source.",
                "tool": "read_text",
                "args": {"path": "ranges.py"},
            }
        elif (not has_workbench and turn == 1) or (has_workbench and turn == 0):
            content = {
                "thought": "Apply the grounded endpoint fix.",
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
        elif (not has_workbench and turn == 2) or (
            has_workbench and not auto_verification and turn == 1
        ):
            content = {
                "thought": "Run objective checks.",
                "tool": "workspace_run_check",
                "args": {"argv": [sys.executable, "-m", "pytest", "-q"]},
            }
        else:
            content = {
                "thought": "The objective checks pass.",
                "done": True,
                "answer": "Fixed and verified.",
            }
        return ModelResponse(
            model=kwargs["model"],
            content=json.dumps(content),
            usage={"input_tokens": 40, "output_tokens": 10, "total_tokens": 50},
            cost_usd=0.001,
            cost_source="provider",
        )


def _materialized_sha() -> str:
    task = get_fixture("off_by_one")
    run_dir = make_run_dir(prefix="brainregion-functional-region-probe-")
    materialize_fixture(task, Path(run_dir))
    try:
        with scoped_workspace_root(run_dir):
            return read_text("ranges.py")["sha256"]
    finally:
        cleanup_run_dir(run_dir)


def test_functional_region_eval_separates_context_ownership_and_verification_effects():
    backend = _FunctionalBackend(_materialized_sha())
    report = asyncio.run(
        run_functional_region_eval(
            backend,
            "main-model",
            [get_fixture("off_by_one")],
            max_steps=4,
            max_cost_usd=1.0,
            bootstrap_samples=20,
        )
    )

    assert report["arms"] == [
        ARM_MAIN_ONLY,
        ARM_PASSIVE_CONTEXT,
        ARM_EVIDENCE_REGION,
        ARM_EVIDENCE_VERIFICATION,
    ]
    assert report["n_runs"] == 4
    assert report["execution"]["actual_model_calls"] == 12
    assert report["execution"]["actual_tool_calls"] == 15
    assert report["execution"]["extraction_mode_counts"] == {"strict_json": 12}
    assert report["execution"]["passive_region_input_contract"] == (
        "model_visible_context_equivalent_v1"
    )
    assert report["execution"]["arm_order_counts"] == {
        "main_only->passive_context->evidence_region->evidence_verification_regions": 1
    }
    assert all(summary["solve_rate"] == 1.0 for summary in report["per_arm"].values())
    assert all(
        summary["extraction_mode_counts"] == {"strict_json": int(summary["mean_steps"])}
        for summary in report["per_arm"].values()
    )

    per_arm = report["per_arm"]
    assert per_arm[ARM_MAIN_ONLY]["mean_steps"] == 4.0
    assert per_arm[ARM_PASSIVE_CONTEXT]["mean_steps"] == 3.0
    assert per_arm[ARM_EVIDENCE_REGION]["mean_steps"] == 3.0
    assert per_arm[ARM_EVIDENCE_VERIFICATION]["mean_steps"] == 2.0
    assert per_arm[ARM_MAIN_ONLY]["mean_main_tool_calls"] == 3.0
    assert per_arm[ARM_PASSIVE_CONTEXT]["mean_context_preparation_tool_calls"] == 2.0
    assert per_arm[ARM_PASSIVE_CONTEXT]["mean_region_tool_calls"] == 0.0
    assert per_arm[ARM_EVIDENCE_REGION]["mean_region_tool_calls"] == 2.0
    assert per_arm[ARM_EVIDENCE_VERIFICATION]["mean_region_tool_calls"] == 3.0
    assert per_arm[ARM_EVIDENCE_VERIFICATION]["mean_main_check_calls"] == 0.0
    by_arm = {case["arm"]: case for case in report["cases"]}
    assert by_arm[ARM_MAIN_ONLY]["region_workbench"]["delivery_mode"] == "disabled"
    assert by_arm[ARM_PASSIVE_CONTEXT]["region_workbench"]["delivery_mode"] == "passive"
    assert by_arm[ARM_EVIDENCE_REGION]["region_workbench"]["delivery_mode"] == "region"
    assert by_arm[ARM_PASSIVE_CONTEXT]["automatic_region_activations"] == 0
    assert by_arm[ARM_EVIDENCE_REGION]["automatic_region_activations"] == 1
    assert by_arm[ARM_EVIDENCE_VERIFICATION]["automatic_region_activations"] == 2

    effects = report["effects"]
    assert effects["context_value"]["raw_deltas"]["steps"] == -1.0
    assert effects["context_value"]["raw_deltas"]["main_read_calls"] == -1.0
    assert effects["context_value"]["raw_deltas"]["total_tool_calls"] == 1.0
    assert effects["context_value"]["bootstrap_deltas"]["steps"]["point"] is None
    assert effects["evidence_ownership"]["raw_deltas"]["steps"] == 0.0
    assert effects["evidence_ownership"]["raw_deltas"]["total_tool_calls"] == 0.0
    assert effects["verification_delegation"]["raw_deltas"]["steps"] == -1.0
    assert effects["verification_delegation"]["raw_deltas"]["main_check_calls"] == -1.0

    first_workbench_calls = [
        call
        for call in backend.calls
        if call["turn"] == 0 and call["has_workbench"] and not call["auto_verification"]
    ]
    assert len(first_workbench_calls) == 2
    assert first_workbench_calls[0]["messages"] == first_workbench_calls[1]["messages"]

    rendered_report = json.dumps(report, ensure_ascii=False)
    assert "for i in range(start, end)" not in rendered_report
    assert "result_preview" not in rendered_report
    assert all(case["contains_context_content"] is False for case in report["cases"])
    assert all(case["contains_tool_results"] is False for case in report["cases"])


def test_functional_region_eval_exposes_opt_in_intent_ownership_arm():
    backend = _FunctionalBackend(_materialized_sha())
    report = asyncio.run(
        run_functional_region_eval(
            backend,
            "main-model",
            [get_fixture("off_by_one")],
            arms=[ARM_EVIDENCE_REGION, ARM_INTENT_EVIDENCE_OWNED],
            max_steps=3,
            max_cost_usd=1.0,
            bootstrap_samples=20,
        )
    )

    by_arm = {case["arm"]: case for case in report["cases"]}
    control = by_arm[ARM_EVIDENCE_REGION]
    treatment = by_arm[ARM_INTENT_EVIDENCE_OWNED]

    assert control["intent_execution"]["enabled"] is False
    assert treatment["intent_execution"]["enabled"] is True
    assert treatment["intent_execution"]["action_owners"] == {
        "read_text": "evidence",
        "search_text": "evidence",
    }
    assert treatment["intent_execution"]["main_denied_actions"] == [
        "read_text",
        "search_text",
    ]
    assert treatment["extraction_mode_counts"] == {"strict_json": 3}
    assert report["effects"]["intent_ownership"]["n_tasks"] == 1


def _record(task_id: str, arm: str, *, solved: bool, infrastructure_error: bool = False) -> dict:
    return {
        "task_id": task_id,
        "repeat": 0,
        "arm": arm,
        "solved": solved,
        "protocol_completed": True,
        "infrastructure_error": infrastructure_error,
        "steps": 3,
        "main_input_tokens": 100,
        "main_total_tokens": 120,
        "main_cost_usd": 0.01,
        "region_cost_usd": 0.0,
        "total_cost_usd": 0.01,
        "main_tool_calls": 2,
        "region_tool_calls": 0,
        "context_preparation_tool_calls": 0,
        "total_tool_calls": 2,
        "main_read_calls": 1,
        "main_check_calls": 1,
        "verification_runs": 0,
        "automatic_region_activations": 0,
        "repeated_target_rate": 0.0,
        "context_preparation_failures": 0,
        "region_workbench": {"enabled": False},
        "main_input_attribution": {},
    }


def test_functional_region_summary_pairs_each_contrast_independently_and_excludes_failures():
    records = []
    for task_id in ("a", "b"):
        records.extend(
            [
                _record(task_id, ARM_MAIN_ONLY, solved=False),
                _record(task_id, ARM_PASSIVE_CONTEXT, solved=True),
                _record(task_id, ARM_EVIDENCE_REGION, solved=True),
                _record(
                    task_id,
                    ARM_EVIDENCE_VERIFICATION,
                    solved=True,
                    infrastructure_error=task_id == "b",
                ),
            ]
        )

    summary = summarize_functional_region_records(
        records,
        run_id="functional-summary",
        bootstrap_samples=20,
    )

    assert summary["effects"]["context_value"]["n_tasks"] == 2
    assert summary["effects"]["context_value"]["raw_deltas"]["solved"] == 1.0
    assert summary["effects"]["evidence_ownership"]["n_tasks"] == 2
    assert summary["effects"]["verification_delegation"]["n_tasks"] == 1
    assert summary["effects"]["end_to_end"]["n_tasks"] == 1


def test_functional_region_eval_rejects_invalid_matrix_configuration():
    task = get_fixture("off_by_one")
    backend = _FunctionalBackend("unused")
    with pytest.raises(ValueError, match="unknown functional Region eval arm"):
        asyncio.run(run_functional_region_eval(backend, "main", [task], arms=["unknown"]))
    with pytest.raises(ValueError, match="cannot contain duplicates"):
        asyncio.run(
            run_functional_region_eval(
                backend,
                "main",
                [task],
                arms=[ARM_MAIN_ONLY, ARM_MAIN_ONLY],
            )
        )
    with pytest.raises(ValueError, match="repeats must be a positive integer"):
        asyncio.run(run_functional_region_eval(backend, "main", [task], repeats=0))
    with pytest.raises(ValueError, match="unknown tool result lifecycle mode"):
        asyncio.run(
            run_functional_region_eval(
                backend,
                "main",
                [task],
                tool_result_lifecycle="unknown",
            )
        )
    assert backend.calls == []


def test_functional_region_eval_redacts_runner_exception_text():
    class _FailingBackend:
        async def complete_messages(self, messages, **kwargs):
            raise RuntimeError("secret/provider/path.py")

    report = asyncio.run(
        run_functional_region_eval(
            _FailingBackend(),
            "main",
            [get_fixture("off_by_one")],
            arms=[ARM_MAIN_ONLY],
            max_steps=1,
        )
    )

    case = report["cases"][0]
    assert case["infrastructure_error"] is True
    assert case["error_type"] == "RuntimeError"
    assert case["error_stage"] == "runner"
    assert "secret/provider/path.py" not in json.dumps(report)


def test_functional_region_eval_cli_contract():
    args = build_parser().parse_args(
        [
            "sandbox",
            "functional-region-eval",
            "--tasks",
            "off_by_one",
            "--main-brain",
            "buzz_anthropic/claude-sonnet-5",
        ]
    )

    assert args.sandbox_command == "functional-region-eval"
    assert args.arms == (
        "main_only,passive_context,evidence_region,evidence_verification_regions"
    )
    assert args.thinking == "off"
    assert args.tool_result_lifecycle == "full"
    assert args.tool_result_live_reads == 3
