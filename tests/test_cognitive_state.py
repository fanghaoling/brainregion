from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from brainregion.providers.base import ModelResponse
from brainregion.sandbox.cognitive_state import MainCognitiveState
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
