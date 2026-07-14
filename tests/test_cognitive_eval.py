from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from brainregion.cli import build_parser
from brainregion.providers.base import ModelResponse
from brainregion.sandbox.cognitive_eval import (
    ARM_COMBINED,
    ARM_EXTERNAL_SCAFFOLD,
    ARM_NATIVE_THINKING,
    ARM_PLAIN,
    run_cognitive_scaffold_eval,
    summarize_cognitive_records,
)
from brainregion.sandbox.fixtures import get_fixture
from brainregion.sandbox.isolation import cleanup_run_dir, make_run_dir, materialize_fixture
from brainregion.workspace import read_text
from brainregion.workspace.files import scoped_workspace_root


class _MatrixBackend:
    def __init__(self, source_sha: str) -> None:
        self.source_sha = source_sha
        self.calls: list[dict] = []
        self.provider_message_keys: list[set[str]] = []

    async def complete_messages(self, messages, **kwargs):
        self.provider_message_keys.extend(set(message) for message in messages)
        system_prompt = messages[0]["content"]
        runtime_scaffold = "Runtime 认知 checkpoint" in system_prompt
        model_managed_scaffold = "思考脚手架已启用" in system_prompt
        scaffold = runtime_scaffold or model_managed_scaffold
        checkpoint = any(
            str(message.get("content", "")).startswith("<runtime_cognitive_checkpoint>")
            for message in messages
        )
        turn = sum(message["role"] == "assistant" for message in messages)
        self.calls.append(
            {
                "thinking": kwargs.get("thinking"),
                "effort": kwargs.get("effort"),
                "scaffold": scaffold,
                "checkpoint": checkpoint,
                "turn": turn,
            }
        )
        if turn == 0:
            content = {
                "thought": "Apply the bounded endpoint fix.",
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
            if model_managed_scaffold:
                content["cognitive_update"] = {
                    "current_subgoal": "Correct the inclusive endpoint.",
                    "hypotheses_upsert": [
                        {
                            "hypothesis_id": "endpoint",
                            "statement": "The range excludes the documented endpoint.",
                            "status": "open",
                            "evidence_refs": ["goal"],
                        }
                    ],
                    "next_action": "Apply the minimal range fix.",
                }
        elif turn == 1:
            content = {
                "thought": "Verify the fixture.",
                "tool": "workspace_run_check",
                "args": {"argv": [sys.executable, "-m", "pytest", "-q"]},
            }
            if model_managed_scaffold:
                content["cognitive_update"] = {
                    "hypotheses_upsert": [
                        {
                            "hypothesis_id": "endpoint",
                            "statement": "The range excluded the documented endpoint.",
                            "status": "supported",
                            "evidence_refs": ["step:0"],
                        }
                    ],
                    "attempts_add": [
                        {
                            "summary": "Applied the inclusive endpoint fix.",
                            "outcome": "succeeded",
                            "evidence_refs": ["step:0"],
                        }
                    ],
                    "next_action": "Run objective verification.",
                }
        else:
            content = {
                "thought": "The configured checks pass.",
                "done": True,
                "answer": "Fixed and verified.",
            }
            if model_managed_scaffold or checkpoint:
                content["cognitive_update"] = {
                    "verification_gap": "",
                    "blocker": "",
                }
        reasoning = 7 if kwargs.get("thinking") else 0
        return ModelResponse(
            model=kwargs["model"],
            content=json.dumps(content),
            usage={
                "input_tokens": 30,
                "output_tokens": 10 + reasoning,
                "total_tokens": 40 + reasoning,
                "completion_tokens_details": {"reasoning_tokens": reasoning},
            },
            cost_usd=0.002 if kwargs.get("thinking") else 0.001,
            cost_source="provider",
        )


def _materialized_sha(task, path: str) -> str:
    run_dir = make_run_dir(prefix="brainregion-cognitive-probe-")
    materialize_fixture(task, Path(run_dir))
    try:
        with scoped_workspace_root(run_dir):
            return read_text(path)["sha256"]
    finally:
        cleanup_run_dir(run_dir)


def test_cognitive_eval_runs_matched_matrix_without_private_state_in_report():
    task = get_fixture("off_by_one")
    source_sha = _materialized_sha(task, "ranges.py")
    backend = _MatrixBackend(source_sha)

    report = asyncio.run(
        run_cognitive_scaffold_eval(
            backend,
            "main-model",
            [task],
            max_steps=3,
            max_cost_usd=1.0,
            effort="medium",
            checkpoint_period=2,
            tool_result_lifecycle="compact",
            tool_result_live_reads=1,
            bootstrap_samples=20,
        )
    )

    assert report["arms"] == [
        ARM_PLAIN,
        ARM_NATIVE_THINKING,
        ARM_EXTERNAL_SCAFFOLD,
        ARM_COMBINED,
    ]
    assert report["n_runs"] == 4
    assert report["execution"]["actual_model_calls"] == 12
    assert report["execution"]["effort"] == "medium"
    assert report["execution"]["scaffold_mode"] == "runtime_checkpoint"
    assert report["execution"]["checkpoint_period"] == 2
    assert report["execution"]["tool_result_lifecycle"] == "compact"
    assert report["execution"]["tool_result_live_reads"] == 1
    assert report["native_thinking_requested"] is True
    assert report["native_thinking_observed"] is True
    assert report["control_reasoning_observed"] is False
    assert report["thinking_telemetry_status"] == "telemetry_confirmed"
    assert report["execution"]["thinking_control"]["adapter_verified"] is False
    assert all(summary["solve_rate"] == 1.0 for summary in report["per_arm"].values())
    assert report["per_arm"][ARM_EXTERNAL_SCAFFOLD]["scaffold_update_success_rate"] == 1.0
    assert report["per_arm"][ARM_COMBINED]["scaffold_update_success_rate"] == 1.0
    assert report["per_arm"][ARM_PLAIN]["scaffold_update_success_rate"] is None
    assert report["per_arm"][ARM_EXTERNAL_SCAFFOLD]["mean_checkpoint_count"] == 1.0
    assert report["per_arm"][ARM_COMBINED]["mean_checkpoint_count"] == 1.0
    assert report["per_arm"][ARM_PLAIN]["mean_checkpoint_count"] is None
    assert all(
        case["main_input_attribution"]["actual_input_tokens"]
        == case["main_input_tokens"]
        for case in report["cases"]
    )
    assert all(
        sum(
            values["actual_input_tokens"]
            for values in case["main_input_attribution"]["categories"].values()
        )
        == case["main_input_tokens"]
        for case in report["cases"]
    )
    assert report["per_arm"][ARM_PLAIN]["input_attribution"][
        "actual_input_tokens"
    ] == 90
    assert "checkpoint" not in report["per_arm"][ARM_PLAIN]["input_attribution"][
        "categories"
    ]
    assert report["per_arm"][ARM_EXTERNAL_SCAFFOLD]["input_attribution"][
        "categories"
    ]["checkpoint"]["actual_input_tokens"] > 0
    assert {(call["thinking"], call["scaffold"]) for call in backend.calls} == {
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    }
    assert all(call["effort"] == "medium" for call in backend.calls if call["thinking"])
    assert all(call["effort"] is None for call in backend.calls if not call["thinking"])
    assert sum(call["checkpoint"] for call in backend.calls) == 2
    assert all(
        not any(key.startswith("_brainregion_") for key in keys)
        for keys in backend.provider_message_keys
    )
    rendered = json.dumps(report, ensure_ascii=False)
    assert "current_subgoal" not in rendered
    assert "evidence_refs" not in rendered
    assert "result_preview" not in rendered
    assert all(case["contains_tool_results"] is False for case in report["cases"])
    assert all(
        case["tool_result_lifecycle"]["contains_result_content"] is False
        for case in report["cases"]
    )


def _record(task_id: str, arm: str, solved: bool) -> dict:
    return {
        "task_id": task_id,
        "repeat": 0,
        "arm": arm,
        "native_thinking": arm in {ARM_NATIVE_THINKING, ARM_COMBINED},
        "external_scaffold": arm in {ARM_EXTERNAL_SCAFFOLD, ARM_COMBINED},
        "solved": solved,
        "protocol_completed": True,
        "infrastructure_error": False,
        "steps": 3,
        "main_input_tokens": 100,
        "main_total_tokens": 120,
        "reasoning_tokens": 10 if arm in {ARM_NATIVE_THINKING, ARM_COMBINED} else 0,
        "cost_usd": 0.01,
        "repeated_target_rate": 0.0,
        "cognitive_scaffold": {"update_attempts": 0, "update_failures": 0},
    }


def test_cognitive_summary_computes_factorial_interaction_at_task_level():
    records = []
    for task_id in ("a", "b"):
        records.extend(
            [
                _record(task_id, ARM_PLAIN, False),
                _record(task_id, ARM_NATIVE_THINKING, False),
                _record(task_id, ARM_EXTERNAL_SCAFFOLD, True),
                _record(task_id, ARM_COMBINED, False),
            ]
        )

    summary = summarize_cognitive_records(records, run_id="factorial", bootstrap_samples=20)

    assert summary["effects"]["scaffold_without_native"]["deltas"]["solved"]["point"] == 1.0
    assert summary["effects"]["scaffold_with_native"]["deltas"]["solved"]["point"] == 0.0
    assert summary["interaction"]["solved"]["point"] == -1.0
    assert summary["interaction"]["solved"]["n"] == 2


def test_cognitive_eval_rejects_unknown_or_duplicate_arms():
    task = get_fixture("off_by_one")
    backend = _MatrixBackend("unused")
    with pytest.raises(ValueError, match="unknown cognitive eval arm"):
        asyncio.run(run_cognitive_scaffold_eval(backend, "main", [task], arms=["unknown"]))
    with pytest.raises(ValueError, match="cannot contain duplicates"):
        asyncio.run(run_cognitive_scaffold_eval(backend, "main", [task], arms=[ARM_PLAIN, ARM_PLAIN]))
    with pytest.raises(ValueError, match="unknown cognitive scaffold mode"):
        asyncio.run(run_cognitive_scaffold_eval(backend, "main", [task], scaffold_mode="unknown"))
    with pytest.raises(ValueError, match="checkpoint_period must be a positive integer"):
        asyncio.run(run_cognitive_scaffold_eval(backend, "main", [task], checkpoint_period=0))
    with pytest.raises(ValueError, match="unknown tool result lifecycle mode"):
        asyncio.run(
            run_cognitive_scaffold_eval(
                backend,
                "main",
                [task],
                tool_result_lifecycle="unknown",
            )
        )
    with pytest.raises(ValueError, match="tool_result_live_reads must be a non-negative integer"):
        asyncio.run(
            run_cognitive_scaffold_eval(
                backend,
                "main",
                [task],
                tool_result_live_reads=-1,
            )
        )
    assert backend.calls == []


def test_cognitive_eval_cli_contract():
    args = build_parser().parse_args(
        [
            "sandbox",
            "cognitive-eval",
            "--tasks",
            "off_by_one",
            "--main-brain",
            "modelbridge_anthropic/claude-sonnet-5",
            "--effort",
            "medium",
        ]
    )
    assert args.sandbox_command == "cognitive-eval"
    assert args.arms == "plain,native_thinking,external_scaffold,combined"
    assert args.effort == "medium"
    assert args.scaffold_mode == "runtime_checkpoint"
    assert args.checkpoint_period == 3
    assert args.tool_result_lifecycle == "full"
    assert args.tool_result_live_reads == 3

    run_args = build_parser().parse_args(
        ["sandbox", "run", "--task", "off_by_one", "--cognitive-scaffold"]
    )
    assert run_args.cognitive_scaffold is True
    assert run_args.cognitive_mode == "runtime_checkpoint"
    assert run_args.checkpoint_period == 3
    assert run_args.tool_result_lifecycle == "full"
    assert run_args.tool_result_live_reads == 3
