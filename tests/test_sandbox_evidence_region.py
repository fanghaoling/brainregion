from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from brainregion.core.intent import IntentCompiler
from brainregion.providers.base import ModelResponse
from brainregion.sandbox import cleanup_run_dir, make_run_dir, materialize_fixture, run_agent
from brainregion.sandbox.fixtures import get_fixture
from brainregion.sandbox.regions import EvidenceRegion, VerificationOptionRegion
from brainregion.sandbox.task import SandboxTask
from brainregion.workspace import read_text
from brainregion.workspace.files import scoped_workspace_root


class ScriptBackend:
    def __init__(self, script: list[str]) -> None:
        self.script = script
        self.index = 0
        self.message_history: list[list[dict]] = []

    async def complete_messages(self, messages, **kwargs):
        self.message_history.append([dict(item) for item in messages])
        content = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        return ModelResponse(
            model=kwargs.get("model", "mock"),
            content=content,
            usage={},
            cost_usd=0.0,
        )


def _json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False)


def test_evidence_region_selects_only_explicit_bounded_relative_text_paths():
    task = SandboxTask(
        id="paths",
        goal=(
            "Inspect src/parser.py, tests/test_parser.py, ../secret.py, "
            "C:\\private\\hidden.py and image.png"
        ),
        test_args=["tests/test_parser.py", "-q"],
    )
    region = EvidenceRegion(max_files=4, max_bytes_per_file=321)

    requests = region.requests(task)

    assert [request.path for request in requests] == ["src/parser.py", "tests/test_parser.py"]
    assert all(request.max_bytes == 321 for request in requests)


def test_evidence_region_turns_successful_reads_into_grounded_blocks():
    region = EvidenceRegion()
    task = SandboxTask(id="one", goal="Read parser.py", test_args=[])
    request = region.requests(task)[0]
    region.observe(
        request,
        result={
            "relative_path": "parser.py",
            "sha256": "abc",
            "start_line": 1,
            "end_line": 2,
            "total_lines": 2,
            "truncated": False,
            "text": "def parse():\n    pass\n",
        },
    )

    block = region.blocks()[0]

    assert block.source == "evidence_region"
    assert block.metadata == {
        "kind": "source_snapshot",
        "path": "parser.py",
        "sha": "abc",
        "region": "evidence",
    }
    assert json.loads(block.content)["text"].startswith("def parse")
    assert region.snapshot()["uses_model"] is False


def test_evidence_region_executes_assignment_scoped_search_and_sanitizes_results():
    task = SandboxTask(id="search", goal="Inspect parser behavior", test_args=[])
    assignment = IntentCompiler().compile(
        {
            "intent_id": task.id,
            "objective": task.goal,
            "required_capabilities": ["code_evidence"],
            "resource_hints": ["src/parser.py"],
            "search_queries": ["parse_config"],
        }
    ).assignment_for("code_evidence")
    assert assignment is not None
    region = EvidenceRegion(max_results_per_search=5)
    requests = region.requests(task, assignment)

    assert [request.action for request in requests] == ["read_text", "search_text"]
    search_request = requests[1]
    region.observe(
        search_request,
        result={
            "matches": [
                {
                    "path": "D:/private/work/src/parser.py",
                    "relative_path": "src/parser.py",
                    "line": 12,
                    "text": "def parse_config():",
                    "context": [{"line": 12, "text": "def parse_config():"}],
                    "root": {"path": "D:/private/work"},
                }
            ],
            "matched_files": 1,
            "scanned_files": 8,
            "truncated": False,
        },
    )

    block = region.blocks()[0]
    content = json.loads(block.content)

    assert block.metadata["kind"] == "search_results"
    assert content["matches"][0]["path"] == "src/parser.py"
    assert "D:/private" not in block.content
    assert region.snapshot()["searches_succeeded"] == 1


def test_evidence_region_validates_and_budgets_main_brain_follow_up_requests():
    task = SandboxTask(id="follow-up", goal="Diagnose behavior", test_args=[])
    assignment = IntentCompiler().compile(
        {
            "intent_id": task.id,
            "objective": task.goal,
            "required_capabilities": ["code_evidence"],
        }
    ).assignment_for("code_evidence")
    assert assignment is not None
    region = EvidenceRegion(max_files=1, max_searches=1, max_follow_ups=2)

    search = region.follow_up_request(
        task,
        assignment,
        {"action": "search_text", "query": "def inclusive_total"},
    )
    read = region.follow_up_request(
        task,
        assignment,
        {"action": "read_text", "path": "range_calc.py"},
    )

    assert search.max_results == region.max_results_per_search
    assert read.max_bytes == region.max_bytes_per_file
    assert region.snapshot()["follow_up_remaining"] == 0
    with pytest.raises(ValueError, match="follow-up limit"):
        region.follow_up_request(
            task,
            assignment,
            {"action": "read_text", "path": "other.py"},
        )

    duplicate_region = EvidenceRegion()
    duplicate_region.follow_up_request(
        task,
        assignment,
        {"action": "search_text", "query": "inclusive_total"},
    )
    with pytest.raises(ValueError, match="already requested"):
        duplicate_region.follow_up_request(
            task,
            assignment,
            {"action": "search_text", "query": "inclusive_total"},
        )
    with pytest.raises(ValueError, match="unknown field"):
        EvidenceRegion().follow_up_request(
            task,
            assignment,
            {"action": "read_text", "path": "range_calc.py", "max_bytes": 999999},
        )
    with pytest.raises(ValueError, match="relative text file"):
        EvidenceRegion().follow_up_request(
            task,
            assignment,
            {"action": "read_text", "path": "../secret.py"},
        )


def test_main_brain_can_request_bounded_evidence_without_owning_reads_or_searches():
    source = "def inclusive_total(start, end):\n    return sum(range(start, end))\n"
    test_source = (
        "from range_calc import inclusive_total\n\n"
        "def test_includes_end():\n"
        "    assert inclusive_total(1, 3) == 6\n"
    )
    task = SandboxTask(
        id="follow-up-loop",
        goal="Fix the inclusive total behavior without assuming which implementation file owns it.",
        files={"range_calc.py": source},
        tests={"tests/test_behavior.py": test_source},
        test_args=["tests/test_behavior.py", "-q"],
    )
    compiled = IntentCompiler().compile(
        {
            "intent_id": task.id,
            "objective": task.goal,
            "required_capabilities": ["code_evidence"],
        }
    )
    run_dir = make_run_dir(prefix="brainregion-evidence-follow-up-")
    materialize_fixture(task, Path(run_dir))
    with scoped_workspace_root(run_dir):
        source_sha = read_text("range_calc.py")["sha256"]
    backend = ScriptBackend(
        [
            _json(
                {
                    "thought": "Find the implementation named by the test.",
                    "tool": "request_evidence",
                    "args": {"action": "search_text", "query": "def inclusive_total"},
                }
            ),
            _json(
                {
                    "thought": "Read the implementation found by the evidence region.",
                    "tool": "request_evidence",
                    "args": {"action": "read_text", "path": "range_calc.py"},
                }
            ),
            _json(
                {
                    "thought": "Apply the grounded repair.",
                    "tool": "apply_text_patch",
                    "args": {
                        "path": "range_calc.py",
                        "expected_sha256": source_sha,
                        "replacements": [
                            {
                                "old_text": "range(start, end)",
                                "new_text": "range(start, end + 1)",
                            }
                        ],
                        "dry_run": False,
                    },
                }
            ),
            _json({"thought": "Verification passed.", "done": True, "answer": "fixed"}),
        ]
    )
    try:
        trajectory = asyncio.run(
            run_agent(
                backend,
                "mock",
                task,
                run_dir=run_dir,
                max_steps=5,
                compiled_intent=compiled,
                evidence_region=EvidenceRegion(),
                option_region=VerificationOptionRegion(),
                option_continuous=True,
                python_exe=sys.executable,
            )
        )
    finally:
        cleanup_run_dir(run_dir)

    first_input = "\n".join(message["content"] for message in backend.message_history[0])
    second_input = "\n".join(message["content"] for message in backend.message_history[1])
    third_input = "\n".join(message["content"] for message in backend.message_history[2])
    assert "- request_evidence(" in first_input
    assert "tests/test_behavior.py" in first_input
    assert "Search evidence: def inclusive_total" in second_input
    assert "Source snapshot: range_calc.py" in third_input
    assert trajectory.tests_green is True, [
        (step.tool, step.error, step.result_preview) for step in trajectory.steps
    ]
    assert [step.tool for step in trajectory.steps] == [
        "request_evidence",
        "request_evidence",
        "apply_text_patch",
        None,
    ]
    assert not {"read_text", "search_text"} & {
        step.tool for step in trajectory.steps if step.tool
    }
    assert trajectory.region_tool_calls == 4
    assert trajectory.automatic_region_activations == 4
    assert trajectory.region_workbench["evidence_follow_up_requests"] == 2
    assert trajectory.region_workbench["evidence_follow_up_remaining"] == 1


def test_empty_evidence_region_still_reports_enabled_workbench():
    task = SandboxTask(id="no-path", goal="Diagnose the current behavior", test_args=[])
    run_dir = make_run_dir(prefix="brainregion-empty-workbench-")
    backend = ScriptBackend(
        [_json({"thought": "No explicit evidence", "done": True, "answer": "stopped"})]
    )
    try:
        trajectory = asyncio.run(
            run_agent(
                backend,
                "mock",
                task,
                run_dir=run_dir,
                max_steps=1,
                evidence_region=EvidenceRegion(),
            )
        )
    finally:
        cleanup_run_dir(run_dir)

    assert trajectory.region_workbench["enabled"] is True
    assert trajectory.region_workbench["entries"] == 0
    assert trajectory.region_workbench["blocks_loaded"] == 0
    assert trajectory.region_tool_calls == 0


def test_evidence_and_verification_regions_share_one_main_brain_workbench():
    task = get_fixture("off_by_one")
    run_dir = make_run_dir(prefix="brainregion-region-workbench-")
    materialize_fixture(task, Path(run_dir))
    with scoped_workspace_root(run_dir):
        sha = read_text("ranges.py")["sha256"]
    backend = ScriptBackend(
        [
            _json(
                {
                    "thought": "Use grounded source snapshot",
                    "tool": "apply_text_patch",
                    "args": {
                        "path": "ranges.py",
                        "expected_sha256": sha,
                        "replacements": [
                            {"old_text": "range(start, end)", "new_text": "range(start, end + 1)"}
                        ],
                        "dry_run": False,
                    },
                }
            ),
            _json({"thought": "Objective verification passed", "done": True, "answer": "fixed"}),
        ]
    )
    try:
        trajectory = asyncio.run(
            run_agent(
                backend,
                "mock",
                task,
                run_dir=run_dir,
                max_steps=3,
                evidence_region=EvidenceRegion(),
                option_region=VerificationOptionRegion(),
                option_continuous=True,
            )
        )
    finally:
        cleanup_run_dir(run_dir)

    first_input = "\n".join(message["content"] for message in backend.message_history[0])
    second_input = "\n".join(message["content"] for message in backend.message_history[1])
    assert "<region_workbench>" in first_input
    assert "Source snapshot: ranges.py" in first_input
    assert "Source snapshot: test_ranges.py" in first_input
    assert sha in first_input
    assert '<region_execution actor="evidence_region"' not in first_input
    assert "Objective verification: passed" in second_input
    assert '<region_execution actor="verification_region"' not in second_input

    assert trajectory.tests_green is True
    assert trajectory.workspace_effects == 1
    assert trajectory.verification_runs == 1
    assert trajectory.region_tool_calls == 3
    assert trajectory.automatic_region_activations == 2
    assert [item["region"] for item in trajectory.option_activations] == [
        "evidence",
        "verification",
    ]
    assert trajectory.region_workbench["by_region"] == {"evidence": 2, "verification": 1}
    assert trajectory.to_dict()["region_workbench"]["contains_context_content"] is False


def test_compiled_intent_gives_evidence_region_read_search_ownership():
    task = get_fixture("off_by_one")
    compiled = IntentCompiler().compile(
        {
            "intent_id": task.id,
            "objective": task.goal,
            "required_capabilities": ["code_evidence"],
            "resource_hints": ["ranges.py", "test_ranges.py"],
            "search_queries": ["range(start, end)"],
        }
    )
    run_dir = make_run_dir(prefix="brainregion-intent-ownership-")
    materialize_fixture(task, Path(run_dir))
    backend = ScriptBackend(
        [
            _json(
                {
                    "thought": "Try to duplicate delegated evidence work",
                    "tool": "read_text",
                    "args": {"path": "ranges.py"},
                }
            ),
            _json({"thought": "Respect the ownership boundary", "done": True, "answer": "stopped"}),
        ]
    )
    try:
        trajectory = asyncio.run(
            run_agent(
                backend,
                "mock",
                task,
                run_dir=run_dir,
                max_steps=2,
                compiled_intent=compiled,
                evidence_region=EvidenceRegion(),
            )
        )
    finally:
        cleanup_run_dir(run_dir)

    first_input = "\n".join(message["content"] for message in backend.message_history[0])
    assert "read_text->evidence" in first_input
    assert "- read_text(" not in first_input
    assert "- search_text(" not in first_input
    assert "Search evidence: range(start, end)" in first_input
    assert trajectory.region_tool_calls == 3
    assert trajectory.steps[0].tool is None
    assert trajectory.steps[0].error_kind == "protocol_error"
    assert "unavailable to this actor" in str(trajectory.steps[0].error)
    assert trajectory.intent_execution["action_owners"] == {
        "read_text": "evidence",
        "search_text": "evidence",
    }
    assert trajectory.intent_execution["main_denied_actions"] == [
        "read_text",
        "search_text",
    ]


def test_compiled_evidence_intent_requires_evidence_region():
    task = SandboxTask(id="owned", goal="Inspect parser.py", test_args=[])
    compiled = IntentCompiler().compile(
        {
            "intent_id": task.id,
            "objective": task.goal,
            "required_capabilities": ["code_evidence"],
        }
    )

    run_dir = make_run_dir(prefix="brainregion-missing-evidence-region-")
    try:
        with pytest.raises(ValueError, match="requires evidence_region"):
            asyncio.run(
                run_agent(
                    ScriptBackend([_json({"thought": "done", "done": True})]),
                    "mock",
                    task,
                    run_dir=run_dir,
                    max_steps=1,
                    compiled_intent=compiled,
                )
            )
    finally:
        cleanup_run_dir(run_dir)


def test_evidence_region_cli_flag():
    from brainregion.cli import build_parser

    args = build_parser().parse_args(
        ["sandbox", "run", "--main-brain", "mock", "--evidence-region"]
    )
    assert args.evidence_region is True
