from __future__ import annotations

import asyncio
import json
from pathlib import Path

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


def test_evidence_region_cli_flag():
    from brainregion.cli import build_parser

    args = build_parser().parse_args(
        ["sandbox", "run", "--main-brain", "mock", "--evidence-region"]
    )
    assert args.evidence_region is True
