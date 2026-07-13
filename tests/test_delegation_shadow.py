from __future__ import annotations

import argparse
import json

import pytest

from brainregion.sandbox.cli import run_delegation_shadow
from brainregion.sandbox.delegation_shadow import (
    shadow_cases_from_delegation_report,
    summarize_shadow_gates,
)


def _step(
    operation: str,
    target: str,
    *,
    new: bool,
    effect: bool = False,
    verification: bool | None = None,
) -> dict:
    return {
        "operation": operation,
        "target_fingerprint": target,
        "target_is_new": new,
        "workspace_effect": effect,
        "verification_passed": verification,
        "error": False,
    }


def test_shadow_policies_separate_normal_discovery_from_repeated_reads():
    cases = [
        {
            "task_id": "easy",
            "solved": True,
            "max_steps": 5,
            "progress_trace": [
                _step("search_text", "q1", new=True),
                _step("read_text", "p1", new=True),
                _step("apply_text_patch", "p1", new=False, effect=True),
                _step("workspace_run_check", "c1", new=True, verification=True),
            ],
        },
        {
            "task_id": "hard",
            "solved": False,
            "max_steps": 5,
            "progress_trace": [
                _step("search_text", "q2", new=True),
                _step("read_text", "p2", new=True),
                _step("read_text", "p2", new=False),
                _step("read_text", "p2", new=False),
            ],
        },
    ]

    summary = summarize_shadow_gates(cases)

    effect_only = summary["policies"]["effect_only_v1"]
    repetition = summary["policies"]["repetition_only"]
    novelty = summary["policies"]["novelty_stall"]
    assert effect_only["activation_rate"] == 1.0
    assert effect_only["easy_case_false_wake_rate"] == 1.0
    assert repetition["activation_rate"] == 0.5
    assert repetition["easy_case_false_wake_rate"] == 0.0
    assert repetition["hard_case_wake_rate"] == 1.0
    assert novelty["hard_case_wake_rate"] == 1.0
    assert summary["models_called"] is False
    assert summary["contains_reasoning"] is False
    assert summary["contains_tool_results"] is False


def test_saved_report_adapter_marks_missing_progress_trace_as_approximate():
    report = {
        "cases": [
            {
                "task_id": "legacy",
                "arm": "main_only",
                "main_result": {
                    "solved": False,
                    "sandbox_diagnostics": {
                        "tool_sequence": ["search_text", "read_text", "read_text"],
                        "workspace_effects": 0,
                        "last_verification_passed": None,
                    },
                },
            }
        ]
    }

    cases = shadow_cases_from_delegation_report(report, max_steps=4)
    summary = summarize_shadow_gates(cases)

    assert cases[0]["trace_quality"] == "legacy_approximate"
    assert summary["trace_quality"] == {"legacy_approximate": 1}
    assert summary["policies"]["repetition_only"]["activations"] == 1


def test_offline_cli_reads_report_without_model_configuration(tmp_path, capsys):
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "execution": {"max_steps": 4},
                "cases": [
                    {
                        "task_id": "x",
                        "arm": "main_only",
                        "main_result": {
                            "solved": True,
                            "sandbox_diagnostics": {
                                "progress_trace": [
                                    _step("apply_text_patch", "p", new=True, effect=True)
                                ]
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = run_delegation_shadow(argparse.Namespace(report=str(path), max_steps=None))

    assert summary["n_cases"] == 1
    assert "models_called=false" in capsys.readouterr().out


def test_shadow_contract_rejects_duplicate_tasks_and_invalid_step_budget():
    case = {"task_id": "x", "solved": False, "max_steps": 4, "progress_trace": []}
    with pytest.raises(ValueError, match="unique"):
        summarize_shadow_gates([case, dict(case)])
    with pytest.raises(ValueError, match="positive integer"):
        summarize_shadow_gates([{**case, "task_id": "bad", "max_steps": 0}])
