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
from brainregion.sandbox.cli import _load_worktree_task
from brainregion.sandbox.task import WorktreeTask
from brainregion.sandbox.worktree_memory_eval import (
    ARM_EXPERT_NO_MEMORY,
    ARM_EXPERT_SCOPED_MEMORY,
    ARM_MAIN_ONLY,
    WorktreeMemoryExpertSpec,
    _safe_source_blocks,
    run_worktree_memory_eval,
    summarize_worktree_memory_records,
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
def code_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "ranges.py").write_text(
        "def sum_range(start, end):\n"
        "    total = 0\n"
        "    for i in range(start, end):\n"
        "        total += i\n"
        "    return total\n",
        encoding="utf-8",
    )
    (repo / "test_ranges.py").write_text(
        "from ranges import sum_range\n\n"
        "def test_inclusive_end():\n"
        "    assert sum_range(1, 3) == 6\n",
        encoding="utf-8",
    )
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "BrainRegion Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    return repo


class _WorktreeMemoryBackend:
    def __init__(self, source_sha: str) -> None:
        self.source_sha = source_sha
        self.main_calls = 0
        self.expert_calls: list[dict] = []

    async def complete(self, **kwargs):
        self.expert_calls.append(kwargs)
        user = str(kwargs.get("user") or "")
        refs = ["worktree:path:ranges.py"]
        memory_used = "memory:id:inclusive-range-history" in user
        if memory_used:
            refs.append("memory:id:inclusive-range-history")
        report = {
            "state": "working",
            "summary": "The implementation excludes the inclusive end value.",
            "implication": "The loop bound must include end.",
            "recommended_action": "Change range(start, end) to range(start, end + 1).",
            "uncertainty": "Run the configured test after patching.",
            "evidence_refs": refs,
            "decision_scope": "routine",
            "risk": "low",
            "memory_impact": "supporting" if memory_used else "none",
            "reversible": True,
            "repeated_failure": False,
            "requires_user_choice": False,
            "needs_more_context": False,
            "covered_scope": "ranges.py inclusive endpoint",
            "unresolved_questions": [],
            "conflicts_with": [],
            "recommended_followups": [],
        }
        return ModelResponse(
            model=kwargs.get("model", "expert"),
            content=json.dumps(report),
            usage={"input_tokens": 100, "output_tokens": 50},
            cost_usd=0.001,
            cost_source="test",
        )

    async def complete_messages(self, messages, **kwargs):
        phase = self.main_calls % 4
        self.main_calls += 1
        has_report = "<expert_reports>" in json.dumps(messages, ensure_ascii=False)
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
                            "old_text": "for i in range(start, end):",
                            "new_text": "for i in range(start, end + 1):",
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
                "adopted_assignment_ids": ["debugger"] if has_report else [],
            }
        return ModelResponse(
            model=kwargs.get("model", "main"),
            content=json.dumps(payload),
            usage={"input_tokens": 20, "output_tokens": 10},
            cost_usd=0.0001,
            cost_source="test",
        )


def test_worktree_memory_eval_runs_three_isolated_real_repo_arms(code_repo: Path):
    source = (code_repo / "ranges.py").read_bytes()
    backend = _WorktreeMemoryBackend(hashlib.sha256(source).hexdigest())
    private_memory = "A previous incident confirmed that the endpoint is inclusive."
    task = WorktreeTask(
        id="inclusive-range",
        goal="Fix the inclusive endpoint bug and make test_ranges.py pass.",
        repo_path=str(code_repo),
        test_args=["-q"],
        bootstrap_commands=[],
        expert_context_paths=["ranges.py", "test_ranges.py"],
        seed_memory=[
            {
                "id": "inclusive-range-history",
                "region": "debugging",
                "status": "active",
                "summary": private_memory,
            },
            {
                "id": "unrelated-security-note",
                "region": "security",
                "summary": "This should be excluded by deterministic region scope.",
            },
        ],
    )
    expert = WorktreeMemoryExpertSpec(
        assignment_id="debugger",
        region="debugging",
        question="Identify the smallest test-backed fix.",
        model="expert-model",
    )

    report = asyncio.run(
        run_worktree_memory_eval(
            backend,
            "main-model",
            task,
            expert,
            repeats=1,
            max_steps=6,
            python_exe=sys.executable,
            run_id="worktree-memory-test",
        )
    )

    assert report["pair_count"] == 1
    assert report["per_arm"][ARM_MAIN_ONLY]["solve_rate"] == 1.0
    assert report["per_arm"][ARM_EXPERT_NO_MEMORY]["solve_rate"] == 1.0
    assert report["per_arm"][ARM_EXPERT_SCOPED_MEMORY]["solve_rate"] == 1.0
    records = {item["arm"]: item for item in report["records"]}
    assert records[ARM_MAIN_ONLY]["expert"]["model_called"] is False
    assert records[ARM_EXPERT_NO_MEMORY]["expert"]["memory_blocks"] == 0
    scoped = records[ARM_EXPERT_SCOPED_MEMORY]["expert"]
    assert scoped["memory_blocks"] == 1
    assert scoped["memory_records_excluded_by_scope"] == 1
    assert records[ARM_EXPERT_NO_MEMORY]["adopted_expert_report"] is True
    assert records[ARM_EXPERT_SCOPED_MEMORY]["adopted_expert_report"] is True
    public = json.dumps(report, ensure_ascii=False)
    assert private_memory not in public
    assert "for i in range(start, end):" not in public
    assert report["execution"]["contains_memory_content"] is False
    assert report["execution"]["contains_diff_content"] is False


def test_worktree_context_paths_reject_escape(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes worktree"):
        _safe_source_blocks(str(root), ["../secret.txt"])


def test_worktree_memory_summary_requires_complete_three_arm_repeat():
    with pytest.raises(ValueError, match="incomplete worktree memory repeats"):
        summarize_worktree_memory_records(
            [{"arm": ARM_MAIN_ONLY, "repeat": 0, "solved": True}]
        )


def test_worktree_memory_cli_and_task_spec_are_explicit(tmp_path: Path):
    spec = tmp_path / "task.json"
    spec.write_text(
        json.dumps(
            {
                "id": "real-task",
                "goal": "Fix the bounded bug.",
                "repo_path": str(tmp_path),
                "expert_context_paths": ["module.py", "tests/test_module.py"],
                "seed_memory": [
                    {
                        "id": "lesson-1",
                        "region": "debugging",
                        "summary": "Prior bounded lesson.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    task = _load_worktree_task(str(spec))
    args = build_parser().parse_args(
        ["sandbox", "worktree-memory-eval", "--task-spec", str(spec)]
    )

    assert task.expert_context_paths == ["module.py", "tests/test_module.py"]
    assert task.seed_memory[0]["id"] == "lesson-1"
    assert args.sandbox_command == "worktree-memory-eval"
    assert args.repeats == 2
    assert args.expert_model is None
    assert args.expert_region == "debugging"
