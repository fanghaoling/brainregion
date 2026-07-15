from __future__ import annotations

import json

import pytest

from brainregion.cli import build_parser
from brainregion.sandbox.envs.rule_shift import RuleShiftEnv
from brainregion.sandbox.epistemic_ledger import (
    classify_change_scale,
    classify_epistemic_error,
)


def _claim(
    hypothesis_id: str,
    *,
    scale: str,
    rule: str,
    scope: str,
    replaces: str = "",
) -> dict:
    return {
        "hypothesis_id": hypothesis_id,
        "rule": rule,
        "scope": scope,
        "replaces": replaces,
        "predicts": {
            "change_scale": scale,
            "level_delta": 0,
            "state": "",
        },
    }


def test_change_scale_uses_public_resolution_relative_boundaries():
    assert classify_change_scale(0, 100) == "none"
    assert classify_change_scale(2, 100) == "local"
    assert classify_change_scale(3, 100) == "regional"
    assert classify_change_scale(25, 100) == "regional"
    assert classify_change_scale(26, 100) == "global"

    with pytest.raises(ValueError, match="between zero and total_cells"):
        classify_change_scale(2, 1)
    with pytest.raises(ValueError, match="must be an integer"):
        classify_change_scale(True, 100)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "an existing epistemic hypothesis_id cannot silently change rule or scope",
            "mutated_hypothesis",
        ),
        (
            "duplicate live epistemic rule; reuse hypothesis_id 'h1'",
            "duplicate_live_rule",
        ),
        (
            "epistemic replaces must reference an existing hypothesis",
            "invalid_replacement",
        ),
        ("epistemic ledger hypothesis limit reached", "hypothesis_limit"),
        ("unknown rule-shift action 'left'", "other_tool_error"),
        (None, ""),
    ],
)
def test_epistemic_error_classifier_is_content_free(message, expected):
    assert classify_epistemic_error(message) == expected


def test_rule_shift_observation_does_not_reveal_schedule_or_phase():
    env = RuleShiftEnv(shift_after=2)

    payload = json.loads(env.observation())
    prompt = env.build_system_prompt("revise a rule")

    assert payload["state"] == "NOT_FINISHED"
    assert payload["available_actions"] == ["action1"]
    assert len(payload["frame"]) == 10
    assert "shift_after" not in payload
    assert "action_count" not in payload
    assert "phase" not in payload
    assert "third action" not in prompt.casefold()
    assert "after 2" not in prompt.casefold()
    assert "replaces" in prompt
    assert "always set level_delta to 0" in prompt
    assert "state to an empty string" in prompt


def test_rule_shift_drives_supported_refuted_replacement_superseded_chain():
    env = RuleShiftEnv(shift_after=2)
    local = _claim(
        "local-effect",
        scale="local",
        rule="action1 changes exactly two visible cells",
        scope="while the observed local effect continues",
    )

    _obs, _reward, _done, first = env.step("action1", epistemic_update=local)
    _obs, _reward, _done, second = env.step("action1", epistemic_update=local)
    _obs, _reward, _done, surprise = env.step("action1", epistemic_update=local)

    assert first["epistemic_feedback"]["status"] == "open"
    assert second["epistemic_feedback"]["status"] == "supported"
    assert surprise["epistemic_feedback"]["matched"] is False
    assert surprise["epistemic_feedback"]["status"] == "refuted"
    assert surprise["epistemic_feedback"]["mismatch_fields"] == ["change_scale"]

    global_rule = _claim(
        "global-effect",
        scale="global",
        rule="action1 changes thirty visible cells after the observed effect shift",
        scope="after the local-effect contradiction",
        replaces="local-effect",
    )
    _obs, _reward, _done, candidate = env.step(
        "action1", epistemic_update=global_rule
    )
    final_obs, reward, done, verified = env.step(
        "action1", epistemic_update=global_rule
    )

    assert candidate["epistemic_feedback"]["status"] == "candidate"
    assert verified["epistemic_feedback"]["status"] == "supported"
    assert verified["epistemic_feedback"]["mismatch_fields"] == []
    assert reward == 1.0
    assert done is True
    assert env.solved is True
    metrics = env.epistemic_ledger.public_metrics()
    assert metrics["surprises"] == 1
    assert metrics["verified_insights"] == 1
    assert metrics["supersessions"] == 1
    view = json.loads(final_obs)["epistemic_ledger"]
    assert view["active_hypotheses"][0]["hypothesis_id"] == "global-effect"
    assert view["suppressed_hypotheses"][0]["hypothesis_id"] == "local-effect"
    assert view["suppressed_hypotheses"][0]["status"] == "superseded"


def test_rule_shift_rejects_missing_ledger_update_before_environment_effect():
    env = RuleShiftEnv()
    before = env.observation()

    with pytest.raises(ValueError, match="args.epistemic"):
        env.step("action1")

    assert env.observation() == before
    assert env.action_trace == []
    assert env.epistemic_ledger.public_metrics()["predictions"] == 0


def test_rule_shift_does_not_accept_replacing_a_never_supported_candidate():
    env = RuleShiftEnv(shift_after=2)
    wrong = _claim(
        "wrong-first-guess",
        scale="global",
        rule="action1 changes most visible cells",
        scope="before any action evidence exists",
    )
    local_candidate = _claim(
        "local-candidate",
        scale="local",
        rule="action1 changes exactly two visible cells",
        scope="after the first observed local effect",
        replaces="wrong-first-guess",
    )
    global_replacement = _claim(
        "global-replacement",
        scale="global",
        rule="action1 changes thirty visible cells after the later effect shift",
        scope="after the local candidate contradiction",
        replaces="local-candidate",
    )

    env.step("action1", epistemic_update=wrong)
    env.step("action1", epistemic_update=local_candidate)
    env.step("action1", epistemic_update=local_candidate)
    env.step("action1", epistemic_update=global_replacement)
    _obs, reward, done, _info = env.step(
        "action1", epistemic_update=global_replacement
    )

    assert done is True
    assert reward == 0.0
    assert env.epistemic_ledger.public_metrics()["verified_insights"] == 1
    assert env.solved is False


def test_rule_shift_cli_defaults_and_transcript_lifecycle_arms():
    parser = build_parser()
    defaults = parser.parse_args(
        ["sandbox", "rule-shift", "--main-brain", "mock/model"]
    )
    suppressed = parser.parse_args(
        [
            "sandbox",
            "rule-shift",
            "--main-brain",
            "mock/model",
            "--epistemic-transcript-lifecycle",
            "suppress",
            "--shift-after",
            "3",
        ]
    )
    evidence = parser.parse_args(
        [
            "sandbox",
            "rule-shift",
            "--main-brain",
            "mock/model",
            "--epistemic-transcript-lifecycle",
            "evidence",
        ]
    )

    assert defaults.sandbox_command == "rule-shift"
    assert defaults.shift_after == 3
    assert defaults.max_steps == 10
    assert defaults.tool_result_lifecycle == "compact"
    assert defaults.tool_result_live_reads == 0
    assert defaults.epistemic_transcript_lifecycle == "full"
    assert suppressed.shift_after == 3
    assert suppressed.epistemic_transcript_lifecycle == "suppress"
    assert evidence.epistemic_transcript_lifecycle == "evidence"
