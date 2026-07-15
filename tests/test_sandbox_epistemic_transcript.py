from __future__ import annotations

import asyncio
import json

import pytest

from brainregion.core.task_coordination import TaskCoordinationBoard
from brainregion.sandbox.epistemic_ledger import EpistemicLedger
from brainregion.sandbox.epistemic_evidence import EpistemicEvidenceWorkspace
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


def test_evidence_mode_keeps_only_allowlisted_runtime_outcome():
    ledger = EpistemicLedger()
    _resolve(ledger, change_scale="none")
    lifecycle = EpistemicTranscriptLifecycle(mode="evidence", ledger=ledger)
    message = attributed_message(
        "assistant",
        '{"thought":"wrong private reasoning","prediction":"secret model rule"}',
        "model_transcript",
    )
    lifecycle.mark(
        message,
        hypothesis_id="action1-effect",
        step=0,
        evidence={
            "action": "action1",
            "matched": False,
            "mismatch_fields": ["change_scale", "model_secret"],
            "expected": {"rule": "secret model rule"},
            "actual": {
                "change_scale": "none",
                "changed_cells": 0,
                "total_cells": 100,
                "level_delta": 0,
                "state": "NOT_FINISHED",
                "private_note": "must not escape",
            },
        },
    )

    messages = [message]
    lifecycle.apply(messages, next_step=1)

    visible = "\n".join(item["content"] for item in provider_messages(messages))
    assert "wrong private reasoning" not in visible
    assert "secret model rule" not in visible
    assert "private_note" not in visible
    assert "model_secret" not in visible
    assert "epistemic_evidence_pointer" in message["content"]
    assert 'evidence_ref="evidence-' in message["content"]
    assert '"action":"action1"' not in message["content"]
    assert "epistemic_evidence_workspace" in visible
    assert '"action":"action1"' in visible
    assert '"change_scale":"none"' in visible
    assert '"changed_cells":0' in visible
    metrics = lifecycle.public_metrics()
    assert metrics["policy"] == "objective_evidence_workspace_v1"
    assert metrics["receipt_mode"] == "workspace_pointer"
    assert metrics["evidence_receipts"] == 1
    assert metrics["evidence_workspace"]["events"] == 1
    assert metrics["contains_objective_evidence"] is True


def test_selective_mode_sleeps_until_an_explicit_bounded_wake():
    lifecycle = EpistemicTranscriptLifecycle(
        mode="selective",
        ledger=EpistemicLedger(),
        selective_wake_live_reads=2,
    )
    message = attributed_message("assistant", "private rule", "model_transcript")
    lifecycle.mark(
        message,
        hypothesis_id="action1-effect",
        step=0,
        rejected=True,
        evidence={
            "action": "action1",
            "matched": True,
            "actual": {
                "change_scale": "local",
                "changed_cells": 2,
                "total_cells": 100,
                "level_delta": 0,
                "state": "NOT_FINISHED",
            },
        },
    )
    messages = [message]

    lifecycle.apply(messages, next_step=1)
    assert "epistemic_evidence_workspace" not in "\n".join(
        item["content"] for item in provider_messages(messages)
    )

    assert lifecycle.request_wake("explicit_recall") is True
    lifecycle.apply(messages, next_step=2)
    assert "epistemic_evidence_workspace" in "\n".join(
        item["content"] for item in provider_messages(messages)
    )
    lifecycle.apply(messages, next_step=3)
    assert "epistemic_evidence_workspace" in "\n".join(
        item["content"] for item in provider_messages(messages)
    )
    lifecycle.apply(messages, next_step=4)
    assert "epistemic_evidence_workspace" not in "\n".join(
        item["content"] for item in provider_messages(messages)
    )

    metrics = lifecycle.public_metrics()
    assert metrics["policy"] == "objective_evidence_selective_wake_v1"
    assert metrics["workspace_refreshes"] == 2
    assert metrics["workspace_skips"] == 2
    assert metrics["selective_wake"] == {
        "enabled": True,
        "live_reads": 2,
        "active": False,
        "reads_remaining": 0,
        "requests": 1,
        "activations": 1,
        "requests_by_reason": {"explicit_recall": 1},
        "contains_focus_content": False,
    }
    assert metrics["event_attention"] == {
        "enabled": True,
        "max_selected_events": 4,
        "selection_passes": 2,
        "selected_events": 2,
        "omitted_events": 0,
        "empty_wakes": 0,
        "last_candidate_events": 1,
        "last_selected_events": 1,
        "last_omitted_events": 0,
        "contains_event_content": False,
        "contains_focus_content": False,
    }


def test_selective_mode_wakes_on_contradiction_and_action_focus_change():
    lifecycle = EpistemicTranscriptLifecycle(
        mode="selective",
        ledger=EpistemicLedger(),
    )
    messages = []
    for step, action, matched in (
        (0, "action1", True),
        (1, "action1", False),
        (2, "action2", True),
    ):
        message = attributed_message("assistant", f"private {step}", "model_transcript")
        lifecycle.mark(
            message,
            hypothesis_id=f"h{step}",
            step=step,
            rejected=True,
            evidence={
                "action": action,
                "matched": matched,
                "actual": {
                    "change_scale": "none",
                    "changed_cells": 0,
                    "total_cells": 100,
                    "level_delta": 0,
                    "state": "NOT_FINISHED",
                },
            },
        )
        messages.append(message)

    lifecycle.apply(messages, next_step=3)

    visible = "\n".join(item["content"] for item in provider_messages(messages))
    assert "epistemic_evidence_workspace" in visible
    wake = lifecycle.public_metrics()["selective_wake"]
    assert wake["requests"] == 2
    assert wake["activations"] == 1
    assert wake["requests_by_reason"] == {
        "action_focus_change": 1,
        "objective_contradiction": 1,
    }


def test_selective_wake_is_inert_for_other_modes_and_validates_enabled_requests():
    lifecycle = EpistemicTranscriptLifecycle(
        mode="full",
        selective_wake_live_reads=0,
        selective_max_events=0,
    )
    assert lifecycle.request_wake("arbitrary disabled request") is False
    assert lifecycle.public_metrics()["selective_wake"]["requests"] == 0

    selective = EpistemicTranscriptLifecycle(
        mode="selective",
        ledger=EpistemicLedger(),
    )
    with pytest.raises(ValueError, match="unknown selective evidence wake reason"):
        selective.request_wake("arbitrary")
    with pytest.raises(ValueError, match="live_reads must be positive"):
        selective.request_wake("expert_request", live_reads=0)
    with pytest.raises(ValueError, match="selective_wake_live_reads"):
        EpistemicTranscriptLifecycle(
            mode="selective",
            ledger=EpistemicLedger(),
            selective_wake_live_reads=0,
        )
    with pytest.raises(ValueError, match="selective_max_events"):
        EpistemicTranscriptLifecycle(
            mode="selective",
            ledger=EpistemicLedger(),
            selective_max_events=0,
        )


def test_external_wake_delivery_is_assignment_scoped_with_provider_read_ttl():
    board = TaskCoordinationBoard()
    board.create_task({"task_id": "root", "goal": "g"})
    board.delegate(
        "root",
        {"assignment_id": "parser", "region": "debugging", "question": "q1"},
    )
    board.delegate(
        "root",
        {"assignment_id": "architecture", "region": "review", "question": "q2"},
    )

    def lifecycle_for(assignment_id: str, action: str):
        lifecycle = EpistemicTranscriptLifecycle(
            mode="selective",
            ledger=EpistemicLedger(),
            task_id="root",
            assignment_id=assignment_id,
            evidence_wake_source=board,
        )
        message = attributed_message(
            "assistant", f"private {assignment_id}", "model_transcript"
        )
        lifecycle.mark(
            message,
            hypothesis_id=f"{assignment_id}-hypothesis",
            step=0,
            rejected=True,
            evidence={
                "action": action,
                "matched": True,
                "actual": {
                    "change_scale": "local",
                    "changed_cells": 1,
                    "total_cells": 10,
                    "level_delta": 0,
                    "state": "NOT_FINISHED",
                },
            },
        )
        return lifecycle, [message]

    parser, parser_messages = lifecycle_for("parser", "parse")
    architecture, architecture_messages = lifecycle_for(
        "architecture", "inspect_boundary"
    )
    board.request_evidence_wake(
        "root",
        "parser",
        reason="expert_request",
        source="region_expert",
        ttl_reads=2,
    )

    architecture.apply(architecture_messages, next_step=1)

    assert architecture.public_metrics()["external_wake_delivery"]["deliveries"] == 0
    assert board.status("root")["evidence_wakes"][0]["remaining_reads"] == 2
    assert "epistemic_evidence_workspace" not in "\n".join(
        item["content"] for item in provider_messages(architecture_messages)
    )

    board.request_evidence_wake(
        "root",
        "architecture",
        reason="task_focus_change",
        source="runtime_policy",
    )
    architecture.apply(architecture_messages, next_step=2)
    parser.apply(parser_messages, next_step=1)
    parser.apply(parser_messages, next_step=2)

    assert "inspect_boundary" in "\n".join(
        item["content"] for item in provider_messages(architecture_messages)
    )
    assert "parse" in "\n".join(
        item["content"] for item in provider_messages(parser_messages)
    )
    assert board.status("root")["evidence_wake_count"] == 0

    parser.apply(parser_messages, next_step=3)
    assert "epistemic_evidence_workspace" not in "\n".join(
        item["content"] for item in provider_messages(parser_messages)
    )
    parser_delivery = parser.public_metrics()["external_wake_delivery"]
    assert parser_delivery == {
        "enabled": True,
        "configured": True,
        "task_id": "root",
        "assignment_id": "parser",
        "batches": 2,
        "deliveries": 2,
        "deliveries_by_reason": {"expert_request": 2},
        "deliveries_by_source": {"region_expert": 2},
        "contains_context_content": False,
        "authorization_boundary": False,
    }


def test_external_wake_source_is_not_touched_when_selective_mode_is_disabled():
    class ExplodingWakeSource:
        calls = 0

        def consume_evidence_wakes(self, task_id: str, assignment_id: str) -> dict:
            self.calls += 1
            raise AssertionError("disabled lifecycle touched wake source")

    source = ExplodingWakeSource()
    lifecycle = EpistemicTranscriptLifecycle(
        mode="full",
        evidence_wake_source=source,
    )
    message = attributed_message("assistant", "unchanged", "model_transcript")

    lifecycle.apply([message], next_step=1)

    assert source.calls == 0
    assert lifecycle.public_metrics()["external_wake_delivery"]["configured"] is False


def test_selective_external_wake_requires_an_exact_assignment_binding():
    with pytest.raises(ValueError, match="requires task_id and assignment_id"):
        EpistemicTranscriptLifecycle(
            mode="selective",
            ledger=EpistemicLedger(),
            task_id="root",
            evidence_wake_source=TaskCoordinationBoard(),
        )


def test_suppress_mode_requires_a_ledger_but_full_mode_is_noop():
    with pytest.raises(ValueError, match="requires an EpistemicLedger"):
        EpistemicTranscriptLifecycle(mode="suppress")
    with pytest.raises(ValueError, match="requires an EpistemicLedger"):
        EpistemicTranscriptLifecycle(mode="evidence")
    with pytest.raises(ValueError, match="requires an EpistemicLedger"):
        EpistemicTranscriptLifecycle(mode="selective")

    lifecycle = EpistemicTranscriptLifecycle(mode="full")
    message = attributed_message("assistant", "unchanged", "model_transcript")
    lifecycle.mark(message, hypothesis_id="h1", step=0, rejected=True)
    lifecycle.apply([message], next_step=1)

    assert message["content"] == "unchanged"
    assert lifecycle.public_metrics()["enabled"] is False


def test_evicted_evidence_event_does_not_leave_a_dangling_pointer():
    lifecycle = EpistemicTranscriptLifecycle(
        mode="evidence",
        ledger=EpistemicLedger(),
        evidence_workspace=EpistemicEvidenceWorkspace(max_events=1),
    )
    messages = []
    for step, action, scale in ((0, "action1", "local"), (1, "action2", "none")):
        message = attributed_message(
            "assistant", f"private reasoning {step}", "model_transcript"
        )
        lifecycle.mark(
            message,
            hypothesis_id=f"h{step}",
            step=step,
            rejected=True,
            evidence={
                "action": action,
                "matched": False,
                "mismatch_fields": ["change_scale"],
                "actual": {
                    "change_scale": scale,
                    "changed_cells": 2 if scale == "local" else 0,
                    "total_cells": 100,
                    "level_delta": 0,
                    "state": "NOT_FINISHED",
                },
            },
        )
        messages.append(message)

    lifecycle.apply(messages, next_step=2)

    assert 'evidence_ref=""' in messages[0]["content"]
    assert 'evidence_ref="evidence-' in messages[1]["content"]
    observed = lifecycle.observe(messages)
    assert observed[0]["evidence_ref"] == ""
    assert observed[1]["evidence_ref"].startswith("evidence-")
    visible = "\n".join(item["content"] for item in provider_messages(messages))
    assert '"action":"action1"' not in visible
    assert '"action":"action2"' in visible
    metrics = lifecycle.public_metrics()
    assert metrics["expired_evidence_receipts"] == 1
    assert metrics["evidence_receipts"] == 1


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


class _RejectedTurnBackend:
    def __init__(self) -> None:
        self.turn = 0
        self.third_messages: list[dict] = []

    async def complete_messages(self, messages, **kwargs):
        del kwargs
        self.turn += 1
        if self.turn == 1:
            update = _claim("first-effect")
        elif self.turn == 2:
            update = _claim("rejected-effect")
            update["predicts"]["change_scale"] = "invalid"
        else:
            self.third_messages = [dict(message) for message in messages]
            return _Response(
                json.dumps({"thought": "stop", "done": True, "answer": "done"})
            )
        return _Response(
            json.dumps(
                {
                    "thought": f"model reasoning turn {self.turn}",
                    "tool": "act",
                    "args": {"action": "action1", "epistemic": update},
                }
            )
        )


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


def test_run_agent_preserves_objective_evidence_without_model_rule(tmp_path):
    backend = _Backend()
    env = _LedgerEnv()
    task = SandboxTask(id="epistemic-evidence", goal="test one rule")

    def verify(*_args, **_kwargs):
        return {
            "tests_green": False,
            "solve_status": "tests_fail",
            "pytest": None,
            "gold_diff": "",
        }

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
                epistemic_transcript_lifecycle="evidence",
            )
        )

    visible = "\n".join(
        str(message.get("content") or "") for message in backend.second_messages
    )
    assert "wrong private reasoning" not in visible
    assert "action1 changes a small visible component" not in visible
    assert "epistemic_evidence_pointer" in visible
    assert "epistemic_evidence_workspace" in visible
    assert '"action":"action1"' in visible
    assert '"change_scale":"none"' in visible
    assert trajectory.epistemic_transcript_lifecycle["evidence_receipts"] == 1


def test_rejected_action_does_not_reuse_previous_runtime_evidence(tmp_path):
    backend = _RejectedTurnBackend()
    env = _LedgerEnv()
    task = SandboxTask(id="epistemic-no-stale-evidence", goal="reject the second update")

    with scoped_env(env):
        trajectory = asyncio.run(
            run_agent(
                backend,
                "mock-model",
                task,
                run_dir=str(tmp_path),
                max_steps=3,
                max_cost_usd=1.0,
                system_prompt="Use the provided protocol.",
                verify_fn=lambda *_args, **_kwargs: {
                    "tests_green": False,
                    "solve_status": "tests_fail",
                    "pytest": None,
                    "gold_diff": "",
                },
                initial_observation=env.observation(),
                epistemic_transcript_lifecycle="evidence",
            )
        )

    visible = "\n".join(
        str(message.get("content") or "") for message in backend.third_messages
    )
    assert "model reasoning turn 1" not in visible
    assert "model reasoning turn 2" not in visible
    assert visible.count('evidence_ref=""') == 1
    assert visible.count("epistemic_evidence_workspace") == 2
    assert trajectory.epistemic_transcript_lifecycle["suppressed_turns"] == 2
    assert trajectory.epistemic_transcript_lifecycle["evidence_receipts"] == 1
    assert trajectory.epistemic_transcript_lifecycle["evidence_workspace"]["events"] == 1
