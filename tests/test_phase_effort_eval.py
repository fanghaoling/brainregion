from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from brainregion.cli import build_parser
from brainregion.providers.base import ModelResponse
from brainregion.sandbox.fixtures import get_fixture
from brainregion.sandbox.isolation import (
    cleanup_run_dir,
    make_run_dir,
    materialize_fixture,
)
from brainregion.sandbox.phase_effort_eval import (
    ARM_FIXED_OFF,
    ARM_PHASE_ACTIVE,
    run_phase_effort_eval,
)
from brainregion.workspace import read_text
from brainregion.workspace.files import scoped_workspace_root


class _PhaseBackend:
    def __init__(self, source_sha: str) -> None:
        self.source_sha = source_sha
        self.calls: list[dict] = []

    async def complete_messages(self, messages, **kwargs):
        turn = sum(message["role"] == "assistant" for message in messages)
        self.calls.append(
            {
                "model": kwargs["model"],
                "thinking": kwargs.get("thinking"),
                "effort": kwargs.get("effort"),
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
        elif turn == 1:
            content = {
                "thought": "Verify the fixture.",
                "tool": "workspace_run_check",
                "args": {"argv": [sys.executable, "-m", "pytest", "-q"]},
            }
        else:
            content = {
                "thought": "The configured checks pass.",
                "done": True,
                "answer": "Fixed and verified.",
            }
        reasoning = 5 if kwargs.get("thinking") else 0
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


def _materialized_sha() -> str:
    task = get_fixture("off_by_one")
    run_dir = make_run_dir(prefix="brainregion-phase-effort-probe-")
    materialize_fixture(task, Path(run_dir))
    try:
        with scoped_workspace_root(run_dir):
            return read_text("ranges.py")["sha256"]
    finally:
        cleanup_run_dir(run_dir)


def test_phase_effort_eval_runs_rotated_matched_pairs_without_private_content():
    task = get_fixture("off_by_one")
    backend = _PhaseBackend(_materialized_sha())

    report = asyncio.run(
        run_phase_effort_eval(
            backend,
            "claude-sonnet-5",
            [task],
            repeats=2,
            max_steps=3,
            max_cost_usd=0.02,
            bootstrap_samples=20,
        )
    )

    assert report["arms"] == [ARM_FIXED_OFF, ARM_PHASE_ACTIVE]
    assert report["n_runs"] == 4
    assert report["n_matched_pairs"] == 2
    assert report["execution"]["actual_model_calls"] == 12
    assert report["execution"]["actual_cost_usd"] == pytest.approx(0.014)
    assert report["execution"]["arm_order_counts"] == {
        "fixed_off->phase_active": 1,
        "phase_active->fixed_off": 1,
    }
    assert report["execution"]["cost_capped"] is False
    assert report["execution"]["thinking_control"]["adapter_verified"] is True
    assert report["experiment_status"] == "INCONCLUSIVE"
    assert report["status_reasons"] == ["insufficient_independent_task_units"]
    assert report["thinking_telemetry_status"] == "telemetry_confirmed"
    assert report["active_thinking_observed"] is True
    assert report["control_reasoning_observed"] is False
    assert report["per_arm"][ARM_FIXED_OFF]["solve_rate"] == 1.0
    assert report["per_arm"][ARM_PHASE_ACTIVE]["solve_rate"] == 1.0
    assert report["per_arm"][ARM_FIXED_OFF]["mean_effective_thinking_calls"] == 0.0
    assert report["per_arm"][ARM_PHASE_ACTIVE]["mean_effective_thinking_calls"] == 1.0
    assert report["per_arm"][ARM_FIXED_OFF]["mean_main_check_calls"] == 1.0
    assert report["per_arm"][ARM_PHASE_ACTIVE]["mean_main_check_calls"] == 1.0
    assert report["effect"]["raw_deltas"]["cost_usd"] == pytest.approx(0.001)
    assert report["pair_outcomes"]["solve"] == {
        "treatment_wins": 0,
        "control_wins": 0,
        "ties": 2,
    }
    assert all(case["protocol_completed"] for case in report["cases"])
    assert all(case["contains_reasoning"] is False for case in report["cases"])
    assert all(case["effort_routing"]["control_scope"] == "backend_request" for case in report["cases"])
    assert all(case["operation_counts"]["workspace_run_check"] == 1 for case in report["cases"])

    control_calls = [call for call in backend.calls if call["thinking"] is False]
    active_calls = [call for call in backend.calls if call["thinking"] is True]
    assert len(active_calls) == 2
    assert all(call["effort"] == "medium" and call["turn"] == 0 for call in active_calls)
    assert all(call["model"] == "claude-sonnet-5" for call in backend.calls)
    assert len(control_calls) == 10

    rendered = json.dumps(report, ensure_ascii=False)
    assert "Apply the bounded endpoint fix" not in rendered
    assert "result_preview" not in rendered


def test_phase_effort_eval_total_cap_stops_between_complete_pairs():
    task = get_fixture("off_by_one")
    backend = _PhaseBackend(_materialized_sha())

    report = asyncio.run(
        run_phase_effort_eval(
            backend,
            "claude-sonnet-5",
            [task],
            repeats=3,
            max_steps=3,
            max_cost_usd=0.02,
            max_total_cost_usd=0.013,
            bootstrap_samples=10,
        )
    )

    assert report["execution"]["cost_capped"] is True
    assert "planned_matrix_cost_capped" in report["status_reasons"]
    assert report["execution"]["completed_pairs"] == 2
    assert report["execution"]["planned_pairs"] == 3
    assert report["n_runs"] == 4
    assert report["n_matched_pairs"] == 2
    assert all(summary["n_runs"] == 2 for summary in report["per_arm"].values())


def test_phase_effort_eval_rejects_invalid_contracts():
    task = get_fixture("off_by_one")
    duplicate = [task, task]
    backend = _PhaseBackend("unused")

    with pytest.raises(ValueError, match="cannot be empty"):
        asyncio.run(run_phase_effort_eval(backend, "main", []))
    with pytest.raises(ValueError, match="must be unique"):
        asyncio.run(run_phase_effort_eval(backend, "main", duplicate))
    with pytest.raises(ValueError, match="repeats must be a positive integer"):
        asyncio.run(run_phase_effort_eval(backend, "main", [task], repeats=0))
    with pytest.raises(ValueError, match="max_total_cost_usd must be positive"):
        asyncio.run(
            run_phase_effort_eval(
                backend,
                "main",
                [task],
                max_total_cost_usd=0,
            )
        )
    assert backend.calls == []


def test_phase_effort_eval_cli_contract():
    args = build_parser().parse_args(
        [
            "sandbox",
            "phase-effort-eval",
            "--tasks",
            "off_by_one,settings_precedence",
            "--main-brain",
            "buzz_anthropic/claude-sonnet-5",
            "--repeats",
            "2",
            "--max-cost-usd",
            "0.08",
            "--max-total-cost-usd",
            "0.32",
        ]
    )

    assert args.sandbox_command == "phase-effort-eval"
    assert args.repeats == 2
    assert args.max_cost_usd == 0.08
    assert args.max_total_cost_usd == 0.32
    assert args.tool_result_lifecycle == "full"
    assert args.tool_result_live_reads == 3
