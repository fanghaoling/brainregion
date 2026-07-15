from __future__ import annotations

import asyncio
import json

import pytest

from brainregion.sandbox.epistemic_ledger import EpistemicLedger
from brainregion.sandbox.epistemic_transcript import EpistemicTranscriptLifecycle
from brainregion.sandbox.input_attribution import attributed_message, provider_messages
from brainregion.sandbox.loop import run_agent, scoped_env
from brainregion.sandbox.task import SandboxTask


def _claim(hypothesis_id: str) -> dict:
    return {
        "hypothesis_id": hypothesis_id,
        "rule": "action1 changes a small visible component",
        "scope": "while the level remains in the current phase",
        "replaces": "",
        "predicts": {
            "change_scale": "local",
            "level_delta": 0,
            "state": "NOT_FINISHED",
        },
    }


def _resolve(ledger: EpistemicLedger, *, change_scale: str) -> None:
    prepared = ledger.prepare(_claim("action1-effect"), action="action1")
    ledger.resolve(
        prepared,
        change_scale=change_scale,
        changed_cells=1 if change_scale != "none" else 0,
        total_cells=100,
        level_delta=0,
        state="NOT_FINISHED",
    )


def test_suppress_mode_unloads_refuted_turn_before_provider_boundary():
    ledger = EpistemicLedger()
    _resolve(ledger, change_scale="none")
    lifecycle = EpistemicTranscriptLifecycle(mode="suppress", ledger=ledger)
    message = attributed_message(
        "assistant",
        '{"thought":"wrong private reasoning","tool":"act","args":{"epistemic":{}}}',
        "model_transcript",
    )
    lifecycle.mark(message, hypothesis_id="action1-effect", step=0)
    messages = [message]

    lifecycle.apply(messages, next_step=1)

    visible = provider_messages(messages)[0]
    assert "wrong private reasoning" not in visible["content"]
    assert "status=\"refuted\"" in visible["content"]
    assert set(visible) == {"role", "content"}
    metrics = lifecycle.public_metrics()
    assert metrics["suppressed_turns"] == 1
    assert metrics["contains_reasoning"] is False


def test_suppress_mode_keeps_open_and_supported_turns():
    ledger = EpistemicLedger()
    _resolve(ledger, change_scale="local")
    lifecycle = EpistemicTranscriptLifecycle(mode="suppress", ledger=ledger)
    message = attributed_message("assistant", "keep this rule", "model_transcript")
    lifecycle.mark(message, hypothesis_id="action1-effect", step=0)

    lifecycle.apply([message], next_step=1)

    assert message["content"] == "keep this rule"
    assert lifecycle.public_metrics()["suppressed_turns"] == 0


def test_suppress_mode_unloads_runtime_rejected_turn_without_ledger_mutation():
    ledger = EpistemicLedger()
    lifecycle = EpistemicTranscriptLifecycle(mode="suppress", ledger=ledger)
    message = attributed_message("assistant", "invalid attempted rewrite", "model_transcript")
    lifecycle.mark(
        message,
        hypothesis_id="action1-effect",
        step=2,
        rejected=True,
    )

    lifecycle.apply([message], next_step=3)

    assert "invalid attempted rewrite" not in message["content"]
    assert lifecycle.public_metrics()["suppressed_by_status"] == {"rejected": 1}


def test_suppress_mode_requires_a_ledger_but_full_mode_is_noop():
    with pytest.raises(ValueError, match="requires an EpistemicLedger"):
        EpistemicTranscriptLifecycle(mode="suppress")

    lifecycle = EpistemicTranscriptLifecycle(mode="full")
    message = attributed_message("assistant", "unchanged", "model_transcript")
    lifecycle.mark(message, hypothesis_id="h1", step=0, rejected=True)
    lifecycle.apply([message], next_step=1)

    assert message["content"] == "unchanged"
    assert lifecycle.public_metrics()["enabled"] is False


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content
        self.error = None
        self.usage = {}
        self.cost_usd = 0.0
        self.cost_source = None

    @property
    def ok(self) -> bool:
        return True


class _Backend:
    def __init__(self) -> None:
        self.turn = 0
        self.second_messages: list[dict] = []

    async def complete_messages(self, messages, **kwargs):
        del kwargs
        self.turn += 1
        if self.turn == 1:
            return _Response(
                json.dumps(
                    {
                        "thought": "wrong private reasoning",
                        "tool": "act",
                        "args": {
                            "action": "action1",
                            "epistemic": _claim("action1-effect"),
                        },
                    }
                )
            )
        self.second_messages = [dict(message) for message in messages]
        return _Response(json.dumps({"thought": "stop", "done": True, "answer": "done"}))


class _LedgerEnv:
    action_vocab = ("action1",)
    supports_action_data = True
    supports_epistemic_update = True
    solved = False
    ego_actions = False

    def __init__(self) -> None:
        self.epistemic_ledger = EpistemicLedger()

    def observation(self) -> str:
        return json.dumps({"epistemic_ledger": self.epistemic_ledger.working_view()})

    def render(self) -> str:
        return self.observation()

    def step(self, action, *, data=None, epistemic_update=None):
        del data
        prepared = self.epistemic_ledger.prepare(epistemic_update, action=action)
        feedback = self.epistemic_ledger.resolve(
            prepared,
            change_scale="none",
            changed_cells=0,
            total_cells=100,
            level_delta=0,
            state="NOT_FINISHED",
        )
        return self.observation(), 0.0, False, {"epistemic_feedback": feedback}


def test_run_agent_suppresses_refuted_turn_before_next_model_call(tmp_path):
    backend = _Backend()
    env = _LedgerEnv()
    task = SandboxTask(id="epistemic-suppression", goal="test one rule")

    def verify(*_args, **_kwargs):
        return {"tests_green": False, "solve_status": "tests_fail", "pytest": None, "gold_diff": ""}

    with scoped_env(env):
        trajectory = asyncio.run(
            run_agent(
                backend,
                "mock-model",
                task,
                run_dir=str(tmp_path),
                max_steps=2,
                max_cost_usd=1.0,
                system_prompt="Use the provided protocol.",
                verify_fn=verify,
                initial_observation=env.observation(),
                epistemic_transcript_lifecycle="suppress",
            )
        )

    visible = "\n".join(str(message.get("content") or "") for message in backend.second_messages)
    assert "wrong private reasoning" not in visible
    assert "epistemic_turn_receipt" in visible
    assert trajectory.epistemic_transcript_lifecycle["suppressed_turns"] == 1
