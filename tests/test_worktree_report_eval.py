from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from brainregion.cli import build_parser
from brainregion.providers.base import ModelResponse
from brainregion.sandbox.task import WorktreeTask
from brainregion.sandbox.worktree_memory_eval import WorktreeMemoryExpertSpec
from brainregion.sandbox.worktree_report_eval import (
    ARM_DECISION_CARD,
    ARM_FULL_REPORT,
    ARM_NO_REPORT,
    run_worktree_report_utilization_eval,
    summarize_worktree_report_records,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


@pytest.fixture
def report_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "ranges.py").write_text(
        "def sum_range(start, end):\n"
        "    return sum(range(start, end))\n",
        encoding="utf-8",
    )
    (repo / "test_ranges.py").write_text(
        "from ranges import sum_range\n\n"
        "def test_inclusive():\n"
        "    assert sum_range(1, 3) == 6\n",
        encoding="utf-8",
    )
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "BrainRegion Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    return repo


class _ReportUtilizationBackend:
    def __init__(self, source_sha: str, output_cap: int) -> None:
        self.source_sha = source_sha
        self.output_cap = output_cap
        self.expert_calls = 0
        self.main_calls = {
            ARM_NO_REPORT: 0,
            ARM_FULL_REPORT: 0,
            ARM_DECISION_CARD: 0,
        }

    async def complete(self, **kwargs):
        self.expert_calls += 1
        report = {
            "state": "working",
            "summary": "The range excludes the inclusive end value.",
            "implication": "The configured regression test remains red.",
            "recommended_action": "Include end in the production range bound.",
            "uncertainty": "Run the protected pytest target after patching.",
            "evidence_refs": [
                "worktree:path:ranges.py",
                "memory:id:inclusive-history",
            ],
            "decision_scope": "routine",
            "risk": "low",
            "memory_impact": "supporting",
            "reversible": True,
            "repeated_failure": False,
            "requires_user_choice": False,
            "needs_more_context": False,
            "covered_scope": "inclusive range behavior",
            "unresolved_questions": [],
            "conflicts_with": [],
            "recommended_followups": ["run protected test"],
        }
        return ModelResponse(
            model=kwargs.get("model", "expert"),
            content=json.dumps(report),
            usage={"input_tokens": 100, "output_tokens": 60},
            cost_usd=0.001,
            cost_source="test",
        )

    async def complete_messages(self, messages, **kwargs):
        blob = json.dumps(messages, ensure_ascii=False)
        if "<expert_reports>" not in blob:
            arm = ARM_NO_REPORT
        elif "recommended_followups" in blob:
            arm = ARM_FULL_REPORT
        else:
            arm = ARM_DECISION_CARD
        phase = self.main_calls[arm]
        self.main_calls[arm] += 1

        if arm == ARM_FULL_REPORT:
            return ModelResponse(
                model=kwargs.get("model", "main"),
                content="analysis without a JSON tool call",
                usage={"input_tokens": 50, "output_tokens": self.output_cap},
                cost_usd=0.0001,
                cost_source="test",
            )
        if phase == 0:
            payload = {"thought": "read", "tool": "read_text", "args": {"path": "ranges.py"}}
        elif phase == 1:
            payload = {
                "thought": "patch",
                "tool": "apply_text_patch",
                "args": {
                    "path": "ranges.py",
                    "expected_sha256": self.source_sha,
                    "replacements": [
                        {
                            "old_text": "sum(range(start, end))",
                            "new_text": "sum(range(start, end + 1))",
                        }
                    ],
                    "dry_run": False,
                },
            }
        elif phase == 2:
            payload = {
                "thought": "verify",
                "tool": "workspace_run_check",
                "args": {"argv": [sys.executable, "-m", "pytest", "-q"]},
            }
        else:
            payload = {
                "thought": "done",
                "done": True,
                "answer": "fixed",
                "adopted_assignment_ids": ["debugger"] if arm != ARM_NO_REPORT else [],
            }
        return ModelResponse(
            model=kwargs.get("model", "main"),
            content=json.dumps(payload),
            usage={"input_tokens": 20, "output_tokens": 10},
            cost_usd=0.0001,
            cost_source="test",
        )


def test_report_utilization_reuses_one_report_across_delivery_arms(report_repo: Path):
    source_sha = hashlib.sha256((report_repo / "ranges.py").read_bytes()).hexdigest()
    backend = _ReportUtilizationBackend(source_sha, output_cap=100)
    task = WorktreeTask(
        id="report-utilization",
        goal="Fix production code without changing tests.",
        repo_path=str(report_repo),
        test_args=["-q"],
        bootstrap_commands=[],
        expert_context_paths=["ranges.py", "test_ranges.py"],
        protected_paths=["test_ranges.py"],
        seed_memory=[
            {
                "id": "inclusive-history",
                "region": "debugging",
                "summary": "Range endpoints in this module are contractually inclusive.",
            }
        ],
    )
    expert = WorktreeMemoryExpertSpec(
        assignment_id="debugger",
        region="debugging",
        question="Find the smallest production-only fix.",
        model="expert-model",
    )

    report = asyncio.run(
        run_worktree_report_utilization_eval(
            backend,
            "main-model",
            task,
            expert,
            repeats=1,
            max_steps=6,
            max_tokens=100,
            python_exe=sys.executable,
            run_id="report-utilization-test",
        )
    )

    assert backend.expert_calls == 1
    assert report["expert_generation"]["runs"] == 1
    assert report["expert_generation"]["mean_memory_blocks"] == 1.0
    assert report["execution"]["report_semantics_reused_across_delivery_arms"] is True
    records = {item["arm"]: item for item in report["records"]}
    assert records[ARM_NO_REPORT]["solved"] is True
    assert records[ARM_DECISION_CARD]["solved"] is True
    assert records[ARM_FULL_REPORT]["solved"] is False
    assert records[ARM_FULL_REPORT]["main_diagnostics"]["error_kind_counts"] == {
        "parse_error": 3
    }
    assert records[ARM_FULL_REPORT]["main_diagnostics"]["saturated_output_calls"] == 3
    assert records[ARM_DECISION_CARD]["advisory_chars"] < records[ARM_FULL_REPORT][
        "advisory_chars"
    ]
    assert records[ARM_DECISION_CARD]["adopted_expert_report"] is True
    assert all(item["protected_paths_unchanged"] for item in report["records"])
    public = json.dumps(report, ensure_ascii=False)
    assert "Range endpoints in this module" not in public
    assert "Include end in the production range bound" not in public


def test_report_utilization_summary_requires_complete_repeat():
    with pytest.raises(ValueError, match="incomplete worktree report repeats"):
        summarize_worktree_report_records(
            [{"arm": ARM_NO_REPORT, "repeat": 0, "solved": False}]
        )


def test_report_utilization_cli_is_explicit():
    args = build_parser().parse_args(
        [
            "sandbox",
            "worktree-report-eval",
            "--task-spec",
            "task.json",
        ]
    )

    assert args.sandbox_command == "worktree-report-eval"
    assert args.repeats == 1
    assert args.expert_region == "debugging"
