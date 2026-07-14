from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from brainregion.providers.base import ModelResponse
from brainregion.sandbox.cognitive_state import MainCognitiveState, RuntimeCognitiveState
from brainregion.sandbox.fixtures import get_fixture
from brainregion.sandbox.isolation import cleanup_run_dir, make_run_dir, materialize_fixture
from brainregion.sandbox.loop import parse_tool_call, run_agent


def test_state_applies_evidence_linked_updates_and_revises_hypotheses():
    state = MainCognitiveState()
    state = state.apply_update(
        {
            "current_subgoal": "Inspect the range implementation.",
            "facts_upsert": [
                {
                    "fact_id": "goal-contract",
                    "statement": "The requested range includes both endpoints.",
                    "evidence_refs": ["goal"],
                }
            ],
            "hypotheses_upsert": [
                {
                    "hypothesis_id": "off-by-one",
                    "statement": "The loop excludes the upper endpoint.",
                    "status": "open",
                    "evidence_refs": [],
                }
            ],
            "next_action": "Read the implementation.",
        },
        valid_evidence_refs={"goal"},
    )
    state = state.apply_update(
        {
            "hypotheses_upsert": [
                {
                    "hypothesis_id": "off-by-one",
                    "statement": "The loop excludes the upper endpoint.",
                    "status": "supported",
                    "evidence_refs": ["step:0"],
                }
            ],
            "attempts_add": [
                {
                    "summary": "Inspected the range loop.",
                    "outcome": "succeeded",
                    "evidence_refs": ["step:0"],
                }
            ],
            "verification_gap": "The fix has not been applied or tested.",
        },
        valid_evidence_refs={"goal", "step:0"},
    )

    assert state.revision == 2
    assert state.facts[0].evidence_refs == ("goal",)
    assert state.hypotheses[0].status == "supported"
    assert state.hypotheses[0].evidence_refs == ("step:0",)
    assert state.attempts[0].outcome == "succeeded"
    assert state.public_metrics()["contains_state_content"] is False
    assert state.public_metrics()["contains_reasoning"] is False


def test_state_rejects_unknown_fields_and_unavailable_evidence_without_mutation():
    state = MainCognitiveState()
    with pytest.raises(ValueError, match="unknown field"):
        state.apply_update({"private_thought": "forbidden"}, valid_evidence_refs={"goal"})
    with pytest.raises(ValueError, match="unavailable reference"):
        state.apply_update(
            {
                "facts_upsert": [
                    {
                        "fact_id": "future",
                        "statement": "Unsupported fact.",
                        "evidence_refs": ["step:9"],
                    }
                ]
            },
            valid_evidence_refs={"goal"},
        )

    failed = state.record_failed_update("missing cognitive_update")
    assert failed.revision == 0
    assert failed.update_attempts == 1
    assert failed.update_failures == 1
    metrics = failed.public_metrics()
    assert metrics["recent_update_error_categories"] == {"missing_update": 1}
    assert "missing cognitive_update" not in json.dumps(metrics)


def test_state_public_metrics_categorize_errors_without_exposing_details():
    state = MainCognitiveState()
    state = state.record_failed_update("fact evidence_refs contains unavailable reference(s): ['step:9']")
    state = state.record_failed_update("cognitive_update unknown field(s): ['private_thought']")

    metrics = state.public_metrics()

    assert metrics["recent_update_error_categories"] == {
        "unavailable_evidence": 1,
        "unknown_field": 1,
    }
    rendered = json.dumps(metrics)
    assert "step:9" not in rendered
    assert "private_thought" not in rendered


def test_runtime_state_reduces_objective_events_and_triggers_sparse_checkpoints():
    state = RuntimeCognitiveState()
    state = state.observe(
        step=0,
        operation="search_text",
        target_kind="query",
        target_label="sum_range",
        target_fingerprint="q1",
        target_is_new=True,
    )
    state = state.observe(
        step=1,
        operation="read_text",
        target_kind="path",
        target_label="ranges.py",
        target_fingerprint="p1",
        target_is_new=True,
    )
    assert state.checkpoint_reason(period=3) is None
    state = state.observe(
        step=2,
        operation="read_text",
        target_kind="path",
        target_label="test_ranges.py",
        target_fingerprint="p2",
        target_is_new=True,
    )
    assert state.checkpoint_reason(period=3) == "periodic"

    state, error = state.complete_checkpoint(
        "periodic",
        {
            "current_subgoal": "Fix the inclusive range endpoint.",
            "hypotheses_upsert": [
                {
                    "hypothesis_id": "off-by-one",
                    "statement": "The implementation excludes the documented endpoint.",
                    "status": "supported",
                    "evidence_refs": ["step:1"],
                }
            ],
            "next_action": "Apply the bounded fix.",
        },
        valid_evidence_refs={"goal", "step:0", "step:1", "step:2"},
    )
    assert error is None
    assert state.checkpoint_count == 1
    assert state.strategy.revision == 1
    assert state.checkpoint_reason(period=3) is None

    state = state.observe(
        step=3,
        operation="apply_text_patch",
        target_kind="path",
        target_label="ranges.py",
        target_fingerprint="p1",
        target_is_new=False,
        workspace_effect=True,
    )
    assert state.pending_verification is True
    assert state.checkpoint_reason(period=3) is None
    state = state.observe(
        step=4,
        operation="workspace_run_check",
        target_kind="command",
        target_label="pytest -q",
        target_fingerprint="c1",
        target_is_new=True,
        verification_passed=False,
    )
    assert state.pending_verification is False
    assert state.checkpoint_reason(period=3) == "verification_failed"
    assert state.public_metrics()["contains_state_content"] is False
    assert state.public_metrics()["checkpoint_reason_counts"] == {"periodic": 1}


def test_runtime_checkpoint_rejects_model_authored_facts_and_isolates_failure():
    state = RuntimeCognitiveState().observe(step=0, operation="read_text")
    state, error = state.complete_checkpoint(
        "periodic",
        {
            "facts_upsert": [
                {
                    "fact_id": "invented",
                    "statement": "The runtime should own objective facts.",
                    "evidence_refs": ["step:0"],
                }
            ]
        },
        valid_evidence_refs={"goal", "step:0"},
    )

    assert error is not None and "unknown field" in error
    assert state.checkpoint_count == 1
    assert state.strategy.revision == 0
    assert state.strategy.update_failures == 1
    assert state.public_metrics()["recent_update_error_categories"] == {
        "unknown_field": 1
    }


def test_runtime_checkpoint_missing_update_and_urgent_error_are_observable():
    state = RuntimeCognitiveState().observe(
        step=0,
        operation="read_text",
        error=True,
    )
    assert state.checkpoint_reason(period=3) == "tool_error"

    state, error = state.complete_checkpoint(
        "tool_error",
        None,
        valid_evidence_refs={"goal", "step:0"},
    )

    assert error == "missing cognitive_update at runtime checkpoint"
    assert state.public_metrics()["recent_update_error_categories"] == {
        "missing_update": 1
    }


def test_tool_parser_accepts_only_object_cognitive_updates():
    call, error = parse_tool_call(
        json.dumps(
            {
                "thought": "Inspect.",
                "tool": "read_text",
                "args": {"path": "ranges.py"},
                "cognitive_update": {"next_action": "Read the implementation."},
            }
        )
    )
    assert error is None
    assert call is not None and call.cognitive_update == {
        "next_action": "Read the implementation."
    }

    call, error = parse_tool_call(
        json.dumps(
            {
                "thought": "Inspect.",
                "tool": "read_text",
                "args": {"path": "ranges.py"},
                "cognitive_update": "free-form",
            }
        )
    )
    assert call is None
    assert error == "'cognitive_update' must be a JSON object"


class _ScaffoldBackend:
    def __init__(self, *, include_update: bool = True) -> None:
        self.include_update = include_update
        self.messages: list[list[dict]] = []

    async def complete_messages(self, messages, **kwargs):
        self.messages.append([dict(message) for message in messages])
        turn = len(self.messages) - 1
        if turn == 0:
            content = {
                "thought": "Inspect the implementation.",
                "tool": "read_text",
                "args": {"path": "ranges.py"},
            }
            if self.include_update:
                content["cognitive_update"] = {
                    "current_subgoal": "Inspect ranges.py.",
                    "hypotheses_upsert": [
                        {
                            "hypothesis_id": "off-by-one",
                            "statement": "The loop may exclude the endpoint.",
                            "status": "open",
                            "evidence_refs": [],
                        }
                    ],
                    "next_action": "Read ranges.py.",
                }
        else:
            content = {
                "thought": "Record the observed implementation.",
                "done": True,
                "answer": "Inspection complete.",
                "cognitive_update": {
                    "facts_upsert": [
                        {
                            "fact_id": "range-loop",
                            "statement": "The implementation uses an exclusive upper bound.",
                            "evidence_refs": ["step:0"],
                        }
                    ],
                    "verification_gap": "No patch or objective verification was completed.",
                },
            }
        return ModelResponse(model=kwargs["model"], content=json.dumps(content))


def _run_scaffold(backend: _ScaffoldBackend):
    task = get_fixture("off_by_one")
    run_dir = make_run_dir(prefix="brainregion-cognitive-state-")
    materialize_fixture(task, Path(run_dir))
    try:
        return asyncio.run(
            run_agent(
                backend,
                "mock",
                task,
                run_dir=run_dir,
                max_steps=2,
                max_cost_usd=1.0,
                cognitive_scaffold=True,
            )
        )
    finally:
        cleanup_run_dir(run_dir)


def test_run_agent_replaces_state_message_and_persists_evidence_across_turns():
    backend = _ScaffoldBackend()
    trajectory = _run_scaffold(backend)

    assert trajectory.cognitive_state is not None
    assert trajectory.cognitive_state.revision == 2
    assert trajectory.cognitive_state.facts[0].evidence_refs == ("step:0",)
    assert trajectory.cognitive_state.update_failures == 0
    assert all(step.cognitive_update_applied for step in trajectory.steps)
    for messages in backend.messages:
        state_messages = [
            message for message in messages if message["content"].startswith("<cognitive_state>")
        ]
        assert len(state_messages) == 1
    assert '"revision":1' in next(
        message["content"]
        for message in backend.messages[1]
        if message["content"].startswith("<cognitive_state>")
    )


def test_missing_state_update_is_observed_but_does_not_block_tool_execution():
    trajectory = _run_scaffold(_ScaffoldBackend(include_update=False))

    assert trajectory.steps[0].tool == "read_text"
    assert trajectory.steps[0].cognitive_update_error == "missing cognitive_update"
    assert trajectory.cognitive_state is not None
    assert trajectory.cognitive_state.update_failures == 1


class _RuntimeCheckpointBackend:
    def __init__(self, *, include_update: bool = True) -> None:
        self.include_update = include_update
        self.messages: list[list[dict]] = []

    async def complete_messages(self, messages, **kwargs):
        self.messages.append([dict(message) for message in messages])
        turn = len(self.messages) - 1
        if turn == 0:
            content = {
                "thought": "Read the implementation.",
                "tool": "read_text",
                "args": {"path": "ranges.py"},
            }
        elif turn == 1:
            content = {
                "thought": "Read the objective tests.",
                "tool": "read_text",
                "args": {"path": "test_ranges.py"},
            }
        elif turn == 2:
            content = {
                "thought": "Confirm the relevant symbol.",
                "tool": "search_text",
                "args": {"query": "sum_range", "include_globs": ["*.py"]},
            }
        else:
            content = {
                "thought": "Record the checkpoint decision.",
                "done": True,
                "answer": "Checkpoint observed.",
            }
            if self.include_update:
                content["cognitive_update"] = {
                    "current_subgoal": "Correct the inclusive endpoint.",
                    "hypotheses_upsert": [
                        {
                            "hypothesis_id": "off-by-one",
                            "statement": "The implementation may exclude the endpoint.",
                            "status": "open",
                            "evidence_refs": ["step:0"],
                        }
                    ],
                    "next_action": "Apply the minimal endpoint fix.",
                }
        return ModelResponse(model=kwargs["model"], content=json.dumps(content))


def _run_runtime_checkpoint(backend: _RuntimeCheckpointBackend):
    task = get_fixture("off_by_one")
    run_dir = make_run_dir(prefix="brainregion-runtime-checkpoint-")
    materialize_fixture(task, Path(run_dir))
    try:
        return asyncio.run(
            run_agent(
                backend,
                "mock",
                task,
                run_dir=run_dir,
                max_steps=4,
                max_cost_usd=1.0,
                cognitive_scaffold=True,
                cognitive_scaffold_mode="runtime_checkpoint",
                cognitive_checkpoint_period=3,
            )
        )
    finally:
        cleanup_run_dir(run_dir)


def test_runtime_checkpoint_is_one_shot_and_does_not_bloat_normal_turns():
    backend = _RuntimeCheckpointBackend()
    trajectory = _run_runtime_checkpoint(backend)

    assert isinstance(trajectory.cognitive_state, RuntimeCognitiveState)
    assert trajectory.cognitive_state.revision == 3
    assert trajectory.cognitive_state.checkpoint_count == 1
    assert trajectory.cognitive_state.strategy.revision == 1
    assert trajectory.steps[-1].cognitive_update_applied is True
    checkpoint_counts = [
        sum(
            message["content"].startswith("<runtime_cognitive_checkpoint>")
            for message in messages
        )
        for messages in backend.messages
    ]
    assert checkpoint_counts == [0, 0, 0, 1]
    assert all(
        not any(message["content"].startswith("<cognitive_state>") for message in messages)
        for messages in backend.messages
    )


def test_runtime_checkpoint_missing_update_isolated_without_blocking_completion():
    trajectory = _run_runtime_checkpoint(
        _RuntimeCheckpointBackend(include_update=False)
    )

    assert trajectory.done is True
    assert isinstance(trajectory.cognitive_state, RuntimeCognitiveState)
    assert trajectory.cognitive_state.checkpoint_count == 1
    assert trajectory.cognitive_state.strategy.update_failures == 1
    assert trajectory.steps[-1].cognitive_update_error == (
        "missing cognitive_update at runtime checkpoint"
    )
