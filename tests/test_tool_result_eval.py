from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from brainregion.cli import build_parser
from brainregion.providers.base import ModelResponse
from brainregion.sandbox.isolation import cleanup_run_dir, make_run_dir, materialize_fixture
from brainregion.sandbox.task import SandboxTask
from brainregion.sandbox.tool_result_eval import (
    ARM_COMPACT,
    ARM_FULL,
    run_tool_result_eval,
    summarize_tool_result_records,
)
from brainregion.workspace import read_text
from brainregion.workspace.files import scoped_workspace_root


def _task(task_id: str = "tool_result_probe") -> SandboxTask:
    catalog = "\n".join(
        f"marker item {index:02d}: deliberately verbose searchable fixture text"
        for index in range(60)
    )
    return SandboxTask(
        id=task_id,
        goal="Fix sum_range so the documented inclusive endpoint tests pass.",
        files={
            "ranges.py": (
                "def sum_range(start: int, end: int) -> int:\n"
                "    return sum(range(start, end))\n"
            ),
            "catalog.txt": catalog,
        },
        tests={
            "test_ranges.py": (
                "from ranges import sum_range\n\n"
                "def test_inclusive_endpoint():\n"
                "    assert sum_range(1, 5) == 15\n"
            )
        },
    )


def _materialized_sha(task: SandboxTask) -> str:
    run_dir = make_run_dir(prefix="brainregion-tool-result-probe-")
    materialize_fixture(task, Path(run_dir))
    try:
        with scoped_workspace_root(run_dir):
            return read_text("ranges.py")["sha256"]
    finally:
        cleanup_run_dir(run_dir)


class _UsageAwareBackend:
    def __init__(self, source_sha: str) -> None:
        self.source_sha = source_sha
        self.calls: list[dict] = []

    async def complete_messages(self, messages, **kwargs):
        assert all(not any(key.startswith("_brainregion_") for key in message) for message in messages)
        turn = sum(message["role"] == "assistant" for message in messages)
        input_tokens = max(
            1,
            sum(len(str(message.get("content") or "")) for message in messages) // 4,
        )
        self.calls.append({"turn": turn, "input_tokens": input_tokens})
        if turn == 0:
            content = {
                "thought": "Find broad fixture references before editing.",
                "tool": "search_text",
                "args": {
                    "query": "marker",
                    "include_globs": ["catalog.txt"],
                    "max_results": 60,
                },
            }
        elif turn == 1:
            content = {
                "thought": "Read the target implementation exactly.",
                "tool": "read_text",
                "args": {"path": "ranges.py"},
            }
        elif turn == 2:
            content = {
                "thought": "Apply the inclusive endpoint fix.",
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
        elif turn == 3:
            content = {
                "thought": "Run the objective check.",
                "tool": "workspace_run_check",
                "args": {"argv": [sys.executable, "-m", "pytest", "-q"]},
            }
        else:
            content = {
                "thought": "The objective check passed.",
                "done": True,
                "answer": "Fixed and verified.",
            }
        output_tokens = 24
        return ModelResponse(
            model=kwargs["model"],
            content=json.dumps(content),
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            cost_usd=input_tokens / 1_000_000,
            cost_source="provider",
        )


def test_tool_result_eval_measures_real_input_savings_without_changing_outcome():
    task = _task()
    backend = _UsageAwareBackend(_materialized_sha(task))

    report = asyncio.run(
        run_tool_result_eval(
            backend,
            "usage-aware-main",
            [task],
            max_steps=5,
            max_cost_usd=1.0,
            bootstrap_samples=20,
        )
    )

    assert report["arms"] == [ARM_FULL, ARM_COMPACT]
    assert report["n_runs"] == 2
    assert report["execution"]["counterbalanced_order"] is False
    assert report["execution"]["arm_order_counts"] == {
        "full_first": 1,
        "compact_first": 0,
        "single_arm": 0,
    }
    assert report["execution"]["shared_prefix_turns"] == 2
    assert report["execution"]["actual_model_calls"] == 8
    assert report["execution"]["accounted_model_calls"] == 10
    assert report["execution"]["replayed_model_calls"] == 2
    assert report["execution"]["actual_cost_usd"] < report["execution"][
        "accounted_cost_usd"
    ]
    assert [call["turn"] for call in backend.calls] == [0, 1, 2, 3, 4, 2, 3, 4]
    assert [case["arm"] for case in report["cases"]] == [ARM_FULL, ARM_COMPACT]
    assert all(case["solved"] for case in report["cases"])
    assert all(case["protocol_completed"] for case in report["cases"])

    full = report["per_arm"][ARM_FULL]
    compact = report["per_arm"][ARM_COMPACT]
    assert compact["mean_main_input_tokens"] < full["mean_main_input_tokens"]
    assert (
        compact["mean_tool_transcript_input_tokens"]
        < full["mean_tool_transcript_input_tokens"]
    )
    assert compact["mean_compacted_results"] >= 1
    assert compact["mean_estimated_input_tokens_avoided"] > 0
    effect = report["matched_effect"]
    assert effect["raw_deltas"]["solved"] == 0.0
    assert effect["raw_deltas"]["main_input_tokens"] < 0
    assert effect["raw_deltas"]["tool_transcript_input_tokens"] < 0
    assert effect["bootstrap_deltas"]["main_input_tokens"]["point"] is None
    assert effect["bootstrap_deltas"]["main_input_tokens"]["n"] == 1
    assert report["pair_quality"]["status"] == "pre_exposure_aligned"
    assert report["pair_quality"]["pre_exposure_aligned_pairs"] == 1
    assert report["pair_diagnostics"][0]["first_compaction_step"] == 2
    assert report["pair_diagnostics"][0]["first_divergence_step"] is None
    assert report["pair_diagnostics"][0]["first_observable_divergence_step"] is None
    assert report["pair_diagnostics"][0]["first_post_exposure_divergence_step"] is None
    assert report["pair_diagnostics"][0]["common_prefix_turns"] == 2
    assert report["pair_diagnostics"][0]["prefix_capture_arm"] == ARM_FULL
    assert report["pair_diagnostics"][0]["prefix_replay_arm"] == ARM_COMPACT
    assert report["pair_diagnostics"][0]["prefix_replay_valid"] is True
    assert report["pair_diagnostics"][0]["prefix_covers_pre_exposure"] is True
    assert report["exposure_aligned_effect"]["raw_deltas"]["main_input_tokens"] < 0

    rendered = json.dumps(report, ensure_ascii=False)
    assert "deliberately verbose searchable fixture text" not in rendered
    assert "catalog.txt" not in rendered
    assert "range(start, end + 1)" not in rendered
    assert all(case["contains_result_content"] is False for case in report["cases"])


def test_tool_result_eval_reverses_prefix_roles_with_counterbalanced_repeat(monkeypatch):
    import brainregion.sandbox.loop as sandbox_loop

    monkeypatch.setattr(
        sandbox_loop,
        "verify_solution",
        lambda *args, **kwargs: {"tests_green": False},
    )
    task = _task("counterbalanced-prefix")
    backend = _UsageAwareBackend(_materialized_sha(task))

    report = asyncio.run(
        run_tool_result_eval(
            backend,
            "usage-aware-main",
            [task],
            repeats=2,
            max_steps=2,
            max_cost_usd=1.0,
            bootstrap_samples=20,
        )
    )

    assert report["execution"]["counterbalanced_order"] is True
    assert report["execution"]["arm_order_counts"] == {
        "full_first": 1,
        "compact_first": 1,
        "single_arm": 0,
    }
    assert report["execution"]["accounted_model_calls"] == 8
    assert report["execution"]["actual_model_calls"] == 4
    assert report["execution"]["replayed_model_calls"] == 4
    assert [
        (case["repeat"], case["arm"], case["shared_prefix"]["role"])
        for case in report["cases"]
    ] == [
        (0, ARM_FULL, "capture"),
        (0, ARM_COMPACT, "replay"),
        (1, ARM_COMPACT, "capture"),
        (1, ARM_FULL, "replay"),
    ]


def _record(task_id: str, arm: str, input_tokens: int, solved: bool = True) -> dict:
    return {
        "task_id": task_id,
        "repeat": 0,
        "arm": arm,
        "infrastructure_error": False,
        "solved": solved,
        "protocol_completed": True,
        "steps": 4,
        "workspace_effects": 1,
        "main_input_tokens": input_tokens,
        "main_total_tokens": input_tokens + 20,
        "tool_transcript_input_tokens": input_tokens // 2,
        "reasoning_tokens": 0,
        "cost_usd": input_tokens / 1_000_000,
        "repeated_target_rate": 0.0,
        "retrieval_calls": 2,
        "repeated_retrieval_calls": 0,
        "repeated_retrieval_rate": 0.0,
        "read_calls": 1,
        "repeated_read_calls": 0,
        "repeated_read_rate": 0.0,
        "main_input_attribution": {},
        "tool_result_lifecycle": {},
    }


def test_tool_result_summary_bootstraps_task_level_compact_minus_full_delta():
    records = []
    for task_id in ("a", "b"):
        records.extend(
            [
                _record(task_id, ARM_FULL, 1000),
                _record(task_id, ARM_COMPACT, 700),
            ]
        )

    report = summarize_tool_result_records(
        records,
        run_id="paired-tool-results",
        bootstrap_samples=20,
    )

    effect = report["matched_effect"]
    assert effect["delta_direction"] == "compact_minus_full"
    assert effect["n_tasks"] == 2
    assert effect["n_matched_repeats"] == 2
    assert effect["raw_deltas"]["main_input_tokens"] == -300.0
    assert effect["bootstrap_deltas"]["main_input_tokens"]["point"] == -300.0
    assert effect["bootstrap_deltas"]["main_input_tokens"]["n"] == 2


def test_tool_result_summary_rejects_pre_exposure_divergence_from_aligned_effect():
    full = _record("diverged", ARM_FULL, 1000, solved=True)
    compact = _record("diverged", ARM_COMPACT, 700, solved=False)
    shared = {
        "step": 0,
        "operation": "search_text",
        "target_kind": "query",
        "target_fingerprint": "same",
        "target_is_new": True,
        "workspace_effect": False,
        "verification_passed": None,
        "error": False,
    }
    full["progress_trace"] = [
        shared,
        {**shared, "step": 1, "operation": "read_text", "target_fingerprint": "path-a"},
    ]
    compact["progress_trace"] = [
        shared,
        {**shared, "step": 1, "operation": "read_text", "target_fingerprint": "path-b"},
    ]
    compact["tool_result_lifecycle"] = {"first_compaction_step": 2}

    report = summarize_tool_result_records(
        [full, compact],
        run_id="pre-exposure-divergence",
        bootstrap_samples=20,
    )

    assert report["matched_effect"]["raw_deltas"]["solved"] == -1.0
    assert report["pair_quality"]["status"] == "pre_exposure_diverged"
    assert report["pair_diagnostics"][0]["first_divergence_step"] == 1
    assert report["pair_diagnostics"][0]["first_observable_divergence_step"] == 1
    assert report["pair_diagnostics"][0]["first_post_exposure_divergence_step"] is None
    assert report["exposure_aligned_effect"]["n_tasks"] == 0
    assert report["exposure_aligned_effect"]["raw_deltas"]["solved"] is None


def test_tool_result_summary_locates_first_post_exposure_divergence():
    full = _record("post-exposure", ARM_FULL, 1000)
    compact = _record("post-exposure", ARM_COMPACT, 700)

    def trace(targets: tuple[str, ...]) -> list[dict]:
        return [
            {
                "step": step,
                "operation": "search_text" if step == 0 else "read_text",
                "target_kind": "query" if step == 0 else "path",
                "target_fingerprint": target,
                "target_is_new": True,
                "workspace_effect": False,
                "verification_passed": None,
                "error": False,
            }
            for step, target in enumerate(targets)
        ]

    full["progress_trace"] = trace(("shared-search", "shared-read", "full-read"))
    compact["progress_trace"] = trace(("shared-search", "shared-read", "compact-read"))
    compact["tool_result_lifecycle"] = {"first_compaction_step": 2}

    report = summarize_tool_result_records(
        [full, compact],
        run_id="post-exposure-divergence",
        bootstrap_samples=20,
    )

    diagnostics = report["pair_diagnostics"][0]
    assert diagnostics["pre_exposure_trace_match"] is True
    assert diagnostics["first_divergence_step"] is None
    assert diagnostics["first_observable_divergence_step"] == 2
    assert diagnostics["first_post_exposure_divergence_step"] == 2
    assert report["exposure_aligned_effect"]["n_tasks"] == 1
    assert report["exposure_aligned_effect"]["raw_deltas"]["solved"] == 0.0


def test_tool_result_summary_excludes_request_mismatch_from_aligned_effect():
    full = _record("prefix-mismatch", ARM_FULL, 1000)
    compact = _record("prefix-mismatch", ARM_COMPACT, 700)
    trace = [
        {
            "step": step,
            "operation": operation,
            "target_kind": "query" if step == 0 else "path",
            "target_fingerprint": f"target-{step}",
            "target_is_new": True,
            "workspace_effect": False,
            "verification_passed": None,
            "error": False,
        }
        for step, operation in enumerate(("search_text", "read_text"))
    ]
    full["progress_trace"] = trace
    compact["progress_trace"] = trace
    compact["tool_result_lifecycle"] = {"first_compaction_step": 2}
    full["shared_prefix"] = {
        "role": "capture",
        "captured_calls": 2,
    }
    compact["shared_prefix"] = {
        "role": "replay",
        "replayed_calls": 1,
        "replay_mismatches": 1,
        "replay_shortfalls": 0,
    }

    report = summarize_tool_result_records(
        [full, compact],
        run_id="prefix-mismatch",
        bootstrap_samples=20,
    )

    assert report["pair_quality"]["status"] == "prefix_replay_invalid"
    assert report["pair_quality"]["pre_exposure_aligned_pairs"] == 0
    assert report["pair_quality"]["prefix_replay_invalid_pairs"] == 1
    assert report["exposure_aligned_effect"]["n_tasks"] == 0


def test_tool_result_eval_rejects_invalid_matrix_configuration():
    task = _task()
    backend = _UsageAwareBackend("unused")
    with pytest.raises(ValueError, match="unknown tool-result eval arm"):
        asyncio.run(run_tool_result_eval(backend, "main", [task], arms=["unknown"]))
    with pytest.raises(ValueError, match="cannot contain duplicates"):
        asyncio.run(run_tool_result_eval(backend, "main", [task], arms=[ARM_FULL, ARM_FULL]))
    with pytest.raises(ValueError, match="tool_result_live_reads must be a non-negative integer"):
        asyncio.run(run_tool_result_eval(backend, "main", [task], tool_result_live_reads=-1))
    with pytest.raises(ValueError, match="shared_prefix_turns must be an integer between 0 and 2"):
        asyncio.run(run_tool_result_eval(backend, "main", [task], shared_prefix_turns=3))
    assert backend.calls == []


def test_tool_result_eval_cli_contract():
    args = build_parser().parse_args(
        [
            "sandbox",
            "tool-result-eval",
            "--tasks",
            "off_by_one,settings_precedence",
            "--main-brain",
            "buzz_anthropic/claude-sonnet-5",
            "--cognitive-scaffold",
            "--thinking",
            "on",
            "--effort",
            "medium",
        ]
    )
    assert args.sandbox_command == "tool-result-eval"
    assert args.arms == "full,compact"
    assert args.repeats == 1
    assert args.cognitive_scaffold is True
    assert args.scaffold_mode == "runtime_checkpoint"
    assert args.checkpoint_period == 3
    assert args.tool_result_live_reads == 3
    assert args.shared_prefix_turns == 2
    assert args.thinking == "on"
    assert args.effort == "medium"
