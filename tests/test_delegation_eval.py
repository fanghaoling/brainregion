"""Matched A/B/C task delegation evaluation tests."""

from __future__ import annotations

import asyncio

import pytest

from brainregion.eval.delegation import (
    ARM_MAIN_ONLY,
    ARM_MULTI_EXPERT,
    ARM_SINGLE_EXPERT,
    DelegationEvalTask,
    ExpertEvalResult,
    MainEvalResult,
    build_delegation_plan,
    run_delegation_eval,
    summarize_delegation_records,
)


def _task(index: int) -> DelegationEvalTask:
    return DelegationEvalTask.from_dict(
        {
            "task_id": f"task-{index}",
            "goal": f"Solve task {index}",
            "assignments": [
                {
                    "assignment_id": "debug",
                    "region": "debugging",
                    "question": "Find the failure mechanism",
                },
                {
                    "assignment_id": "architecture",
                    "region": "review",
                    "question": "Check the ownership boundary",
                },
            ],
        }
    )


def _report(summary: str) -> dict:
    return {
        "state": "done",
        "summary": summary,
        "context_state": "ready",
        "decision_scope": "routine",
        "risk": "low",
        "memory_impact": "supporting",
        "reversible": True,
    }


def test_plan_uses_matched_task_repeat_arm_order_and_deterministic_assignments():
    plan = build_delegation_plan(_task(0), repeats=2)

    assert plan["matched_order"] == "task_then_repeat_then_arm"
    assert [run["arm"] for run in plan["runs"][:3]] == [
        ARM_MAIN_ONLY,
        ARM_SINGLE_EXPERT,
        ARM_MULTI_EXPERT,
    ]
    assert plan["runs"][0]["assignment_ids"] == []
    assert plan["runs"][1]["assignment_ids"] == ["debug"]
    assert plan["runs"][2]["assignment_ids"] == ["debug", "architecture"]
    assert plan["models_called"] is False


def test_runner_executes_zero_one_many_experts_and_tracks_split_costs():
    expert_calls = []
    main_calls = []

    async def expert_runner(task, run, assignment):
        expert_calls.append((task.task_id, run.repeat, run.arm, assignment.assignment_id))
        return ExpertEvalResult(
            assignment_id="model-cannot-choose-identity",
            report=_report(f"{assignment.assignment_id} report"),
            usage={"input_tokens": 10, "output_tokens": 5},
            cost_usd=0.01,
        )

    async def main_runner(task, run, reports):
        main_calls.append((task.task_id, run.repeat, run.arm, tuple(r["assignment_id"] for r in reports)))
        adopted = tuple(r["assignment_id"] for r in reports)
        return MainEvalResult(
            solved=run.arm != ARM_MAIN_ONLY,
            score=0.5 if run.arm == ARM_MAIN_ONLY else 1.0,
            steps=3 if run.arm == ARM_MAIN_ONLY else 2,
            repeated_attempts=1 if run.arm == ARM_MAIN_ONLY else 0,
            adopted_assignment_ids=adopted,
            usage={"input_tokens": 100, "output_tokens": 20},
            cost_usd=0.02,
        )

    report = asyncio.run(
        run_delegation_eval(
            [_task(0), _task(1), _task(2)],
            main_runner=main_runner,
            expert_runner=expert_runner,
            run_id="delegation-test",
            bootstrap_samples=200,
        )
    )

    assert len(expert_calls) == 9  # each task:0 + 1 + 2
    assert len(main_calls) == 9
    assert main_calls[0][3] == ()
    assert main_calls[1][3] == ("debug",)
    assert main_calls[2][3] == ("debug", "architecture")
    assert report["per_arm"][ARM_SINGLE_EXPERT]["report_adoption_rate"] == 1.0
    assert report["per_arm"][ARM_MULTI_EXPERT]["mean_expert_cost_usd"] == 0.02
    assert report["per_arm"][ARM_MULTI_EXPERT]["mean_total_tokens"] == 150
    assert report["pairwise"]["main_only_vs_single_expert"]["n_tasks"] == 3
    assert report["bootstrap_unit"] == "task"
    assert report["contains_reasoning"] is False
    assert report["contains_private_context"] is False


def test_expert_failure_is_isolated_and_invalid_report_is_not_sent_to_main():
    seen_reports = []

    async def expert_runner(_task, _run, assignment):
        if assignment.assignment_id == "debug":
            raise RuntimeError("offline")
        return ExpertEvalResult(
            assignment_id=assignment.assignment_id,
            report={**_report("bad"), "chain_of_thought": "forbidden"},
        )

    async def main_runner(_task, _run, reports):
        seen_reports.append(reports)
        return MainEvalResult(solved=False)

    report = asyncio.run(
        run_delegation_eval(
            [_task(0)],
            main_runner=main_runner,
            expert_runner=expert_runner,
            arms=[ARM_MULTI_EXPERT],
            run_id="failure-test",
        )
    )

    assert seen_reports == [()]
    assert report["records"][0]["expert_failures"] == 2
    assert report["cases"][0]["expert_results"][0]["error"].startswith("expert_runner_error")
    assert report["cases"][0]["expert_results"][1]["error"].startswith("invalid_region_report")


def test_summary_bootstraps_by_task_after_aggregating_repeats():
    records = []
    for task_id in ("a", "b", "c"):
        for repeat in (0, 1):
            for arm, solved in ((ARM_MAIN_ONLY, False), (ARM_SINGLE_EXPERT, True)):
                records.append(
                    {
                        "task_id": task_id,
                        "repeat": repeat,
                        "arm": arm,
                        "solved": solved,
                        "score": float(solved),
                        "steps": 1,
                        "repeated_attempts": 0,
                        "reports_produced": int(solved),
                        "reports_adopted": int(solved),
                        "expert_failures": 0,
                        "main_input_tokens": 10,
                        "main_total_tokens": 20,
                        "expert_total_tokens": 5 if solved else 0,
                        "total_tokens": 25 if solved else 20,
                        "main_cost_usd": 0.01,
                        "expert_cost_usd": 0.01 if solved else 0,
                        "total_cost_usd": 0.02 if solved else 0.01,
                        "main_error": False,
                    }
                )

    summary = summarize_delegation_records(records, run_id="cluster-test", bootstrap_samples=200)
    pair = summary["pairwise"]["main_only_vs_single_expert"]

    assert summary["n_runs"] == 12
    assert pair["n_tasks"] == 3
    assert pair["deltas"]["solved_delta"]["point"] == 1.0
    assert pair["gate"]["decision"] == "pilot_GO"


def test_incomplete_multi_arm_does_not_discard_complete_single_expert_pairs():
    records = []
    for task_id in ("a", "b"):
        for arm, solved in ((ARM_MAIN_ONLY, False), (ARM_SINGLE_EXPERT, True)):
            records.append(
                {
                    "task_id": task_id,
                    "repeat": 0,
                    "arm": arm,
                    "solved": solved,
                    "score": float(solved),
                    "steps": 1,
                    "repeated_attempts": 0,
                    "reports_produced": int(solved),
                    "reports_adopted": int(solved),
                    "expert_failures": 0,
                    "main_input_tokens": 10,
                    "main_total_tokens": 20,
                    "expert_total_tokens": 5 if solved else 0,
                    "total_tokens": 25 if solved else 20,
                    "main_cost_usd": 0.01,
                    "expert_cost_usd": 0.01 if solved else 0,
                    "total_cost_usd": 0.02 if solved else 0.01,
                    "main_error": False,
                }
            )
    records.append({**records[-1], "task_id": "a", "arm": ARM_MULTI_EXPERT})

    summary = summarize_delegation_records(records, run_id="incomplete", bootstrap_samples=50)

    assert summary["pairwise"]["main_only_vs_single_expert"]["n_tasks"] == 2
    assert summary["pairwise"]["main_only_vs_multi_expert"]["n_tasks"] == 1


def _paired_record(task_id: str, repeat: int, arm: str, solved: bool) -> dict:
    expert = arm != ARM_MAIN_ONLY
    return {
        "task_id": task_id,
        "repeat": repeat,
        "arm": arm,
        "solved": solved,
        "score": float(solved),
        "steps": 1,
        "repeated_attempts": 0,
        "reports_produced": int(expert),
        "reports_adopted": int(expert),
        "expert_failures": 0,
        "main_input_tokens": 10,
        "main_total_tokens": 20,
        "expert_total_tokens": 5 if expert else 0,
        "total_tokens": 25 if expert else 20,
        "main_cost_usd": 0.01,
        "expert_cost_usd": 0.01 if expert else 0.0,
        "total_cost_usd": 0.02 if expert else 0.01,
        "main_error": False,
    }


def test_pairwise_summary_uses_only_repeat_ids_present_in_both_arms():
    records = [
        _paired_record("a", 0, ARM_MAIN_ONLY, False),
        _paired_record("a", 0, ARM_SINGLE_EXPERT, True),
        _paired_record("a", 1, ARM_MAIN_ONLY, True),
        _paired_record("b", 0, ARM_MAIN_ONLY, False),
        _paired_record("b", 0, ARM_SINGLE_EXPERT, True),
    ]

    summary = summarize_delegation_records(records, run_id="matched-repeat", bootstrap_samples=50)
    pair = summary["pairwise"]["main_only_vs_single_expert"]

    assert pair["n_tasks"] == 2
    assert pair["n_matched_repeats"] == 2
    assert pair["deltas"]["solved_delta"]["point"] == 1.0


def test_summary_rejects_duplicate_runs_and_inconsistent_telemetry():
    record = _paired_record("a", 0, ARM_MAIN_ONLY, False)
    with pytest.raises(ValueError, match="duplicate delegation run"):
        summarize_delegation_records([record, dict(record)])
    with pytest.raises(ValueError, match="total_tokens"):
        summarize_delegation_records([{**record, "total_tokens": 21}])
    with pytest.raises(ValueError, match="main_only"):
        summarize_delegation_records(
            [
                {
                    **record,
                    "expert_total_tokens": 1,
                    "total_tokens": 21,
                }
            ]
        )


def test_summary_requires_complete_metric_schema():
    record = _paired_record("a", 0, ARM_MAIN_ONLY, False)
    record.pop("main_cost_usd")

    with pytest.raises(ValueError, match="missing field"):
        summarize_delegation_records([record])


def test_contracts_fail_fast_for_invalid_tasks_arms_and_main_results():
    with pytest.raises(ValueError, match="unknown field"):
        DelegationEvalTask.from_dict({"task_id": "x", "goal": "g", "raw_context": "forbidden"})
    with pytest.raises(ValueError, match="no expert assignments"):
        build_delegation_plan(
            DelegationEvalTask.from_dict({"task_id": "x", "goal": "g"}),
            arms=[ARM_SINGLE_EXPERT],
        )
    with pytest.raises(ValueError, match="unknown delegation arm"):
        build_delegation_plan(_task(0), arms=["debate"])
    with pytest.raises(ValueError, match="between 0 and 1"):
        MainEvalResult(solved=True, score=2.0)


@pytest.mark.parametrize(
    "updates",
    [
        {"solved": "false"},
        {"score": 2.0},
        {"reports_produced": 0, "reports_adopted": 1},
        {"task_id": ""},
        {"main_error": "false"},
        {"steps": 1.5},
        {"raw_context": "must not be accepted"},
    ],
)
def test_imported_metric_records_are_strictly_validated(updates):
    record = {
        "task_id": "x",
        "repeat": 0,
        "arm": ARM_MAIN_ONLY,
        "solved": False,
        "score": 0.0,
        "steps": 0,
        "repeated_attempts": 0,
        "reports_produced": 0,
        "reports_adopted": 0,
        "expert_failures": 0,
        "main_input_tokens": 0,
        "main_total_tokens": 0,
        "expert_total_tokens": 0,
        "total_tokens": 0,
        "main_cost_usd": 0.0,
        "expert_cost_usd": 0.0,
        "total_cost_usd": 0.0,
        "main_error": False,
    }
    with pytest.raises(ValueError):
        summarize_delegation_records([{**record, **updates}])
