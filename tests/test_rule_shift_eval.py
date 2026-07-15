from __future__ import annotations

import asyncio
import json

import pytest

from brainregion.cli import build_parser
from brainregion.sandbox.epistemic_ledger import EpistemicLedger
from brainregion.sandbox.rule_shift_eval import (
    ARM_FULL,
    ARM_SUPPRESS,
    run_rule_shift_eval,
    summarize_rule_shift_records,
)


class _Response:
    def __init__(self, content: str, *, input_tokens: int) -> None:
        self.content = content
        self.error = None
        self.usage = {
            "input_tokens": input_tokens,
            "output_tokens": 50,
            "total_tokens": input_tokens + 50,
        }
        self.cost_usd = 0.001
        self.cost_source = "test"

    @property
    def ok(self) -> bool:
        return True


class _LedgerAwareBackend:
    def __init__(self) -> None:
        self.calls = 0

    async def complete_messages(self, messages, **kwargs):
        del kwargs
        self.calls += 1
        observation = _latest_observation(messages)
        input_tokens = max(
            1,
            sum(len(str(message.get("content") or "")) for message in messages) // 4,
        )
        if observation["state"] == "WIN":
            content = json.dumps(
                {"thought": "terminal state observed", "done": True, "answer": "done"}
            )
            return _Response(content, input_tokens=input_tokens)

        ledger = observation["epistemic_ledger"]
        active = ledger["active_hypotheses"]
        suppressed = ledger["suppressed_hypotheses"]
        if active:
            hypothesis = active[0]
            scale = "global" if hypothesis["hypothesis_id"] == "global-effect" else "local"
            update = _claim(
                hypothesis["hypothesis_id"],
                rule=hypothesis["rule"],
                scope=hypothesis["scope"],
                replaces=hypothesis["replaces"],
                scale=scale,
            )
        else:
            refuted = [item for item in suppressed if item["status"] == "refuted"]
            if not refuted:
                update = _claim(
                    "initial-guess",
                    rule="action1 changes most visible cells",
                    scope="before observing an action effect",
                    scale="global",
                )
            elif any(item["hypothesis_id"] == "local-effect" for item in refuted):
                update = _claim(
                    "global-effect",
                    rule="action1 changes thirty visible cells after the effect shift",
                    scope="after the supported local rule was contradicted",
                    replaces="local-effect",
                    scale="global",
                )
            else:
                update = _claim(
                    "local-effect",
                    rule="action1 changes exactly two visible cells",
                    scope="after the initial calibration action",
                    replaces=refuted[-1]["hypothesis_id"],
                    scale="local",
                )
        content = json.dumps(
            {
                "thought": "test the current falsifiable rule",
                "tool": "act",
                "args": {"action": "action1", "epistemic": update},
            }
        )
        return _Response(content, input_tokens=input_tokens)


def _latest_observation(messages: list[dict]) -> dict:
    for message in reversed(messages):
        content = str(message.get("content") or "")
        if "<visual>" not in content:
            continue
        payload = content.split("<visual>", 1)[1].split("</visual>", 1)[0].strip()
        return json.loads(payload)
    raise AssertionError("rule-shift observation was not supplied to the model")


def _claim(
    hypothesis_id: str,
    *,
    rule: str,
    scope: str,
    scale: str,
    replaces: str = "",
) -> dict:
    return {
        "hypothesis_id": hypothesis_id,
        "rule": rule,
        "scope": scope,
        "replaces": replaces,
        "predicts": {"change_scale": scale, "level_delta": 0, "state": ""},
    }


def test_working_view_exposes_replacement_id_needed_for_exact_reuse():
    ledger = EpistemicLedger()
    old = _claim(
        "old",
        rule="action1 changes most visible cells",
        scope="initial observation",
        scale="global",
    )
    prepared = ledger.prepare(old, action="action1")
    ledger.resolve(
        prepared,
        change_scale="local",
        changed_cells=2,
        total_cells=100,
        level_delta=0,
        state="NOT_FINISHED",
    )
    replacement = _claim(
        "replacement",
        rule="action1 changes exactly two visible cells",
        scope="after calibration",
        replaces="old",
        scale="local",
    )
    prepared = ledger.prepare(replacement, action="action1")
    ledger.resolve(
        prepared,
        change_scale="local",
        changed_cells=2,
        total_cells=100,
        level_delta=0,
        state="NOT_FINISHED",
    )

    active = ledger.working_view()["active_hypotheses"][0]
    assert active["hypothesis_id"] == "replacement"
    assert active["replaces"] == "old"


def test_rule_shift_eval_counterbalances_and_replays_safe_first_turn(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backend = _LedgerAwareBackend()

    report = asyncio.run(
        run_rule_shift_eval(
            backend,
            "fake-model",
            repeats=2,
            max_steps=8,
            max_cost_usd=1.0,
            shared_prefix_turns=1,
            bootstrap_samples=20,
            run_id="rule-shift-test",
        )
    )

    assert report["execution"]["arm_order_counts"] == {
        "full_first": 1,
        "suppress_first": 1,
    }
    assert report["execution"]["counterbalanced_order"] is True
    assert report["execution"]["replayed_model_calls"] == 2
    assert report["execution"]["actual_provider_calls"] == backend.calls
    assert backend.calls == 26
    assert report["pair_quality"] == {
        "status": "matched",
        "complete_pairs": 2,
        "valid_pairs": 2,
        "treatment_exposed_pairs": 2,
        "prefix_replay_invalid_pairs": 0,
        "first_action_diverged_pairs": 0,
        "infrastructure_failed_pairs": 0,
    }
    assert report["per_arm"][ARM_FULL]["solve_rate"] == 1.0
    assert report["per_arm"][ARM_SUPPRESS]["solve_rate"] == 1.0
    assert report["matched_effect"]["n_pairs"] == 2
    assert report["matched_effect"]["bootstrap_deltas"]["total_tokens"]["n"] == 2
    assert report["exposure_aligned_effect"]["n_pairs"] == 2
    assert all(case["contains_rule_content"] is False for case in report["cases"])
    serialized = json.dumps(report)
    assert "action1 changes exactly two visible cells" not in serialized
    assert "action1 changes thirty visible cells" not in serialized


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"repeats": 0}, "repeats"),
        ({"shared_prefix_turns": 2}, "shared_prefix_turns"),
        ({"max_total_cost_usd": 0.0}, "max_total_cost_usd"),
        ({"bootstrap_samples": 0}, "bootstrap_samples"),
    ],
)
def test_rule_shift_eval_rejects_invalid_experiment_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        asyncio.run(run_rule_shift_eval(_LedgerAwareBackend(), "fake", **kwargs))


def test_rule_shift_summary_excludes_pair_with_recovered_provider_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = asyncio.run(
        run_rule_shift_eval(
            _LedgerAwareBackend(),
            "fake-model",
            repeats=2,
            max_steps=8,
            max_cost_usd=1.0,
            bootstrap_samples=20,
        )
    )
    records = report["cases"]
    degraded = next(
        record
        for record in records
        if record["repeat"] == 0 and record["arm"] == ARM_FULL
    )
    degraded["infrastructure_error"] = True
    degraded["infrastructure_degraded"] = True
    degraded["error_kind_counts"] = {"model_error": 1}

    summary = summarize_rule_shift_records(
        records,
        run_id="degraded-pair",
        bootstrap_samples=20,
    )

    assert summary["per_arm"][ARM_FULL]["n_valid_runs"] == 1
    assert summary["per_arm"][ARM_SUPPRESS]["n_valid_runs"] == 2
    assert summary["matched_effect"]["n_pairs"] == 1
    assert summary["matched_effect"]["bootstrap_deltas"]["total_tokens"]["point"] is None
    assert summary["pair_quality"]["status"] == "matched_with_infrastructure_failures"
    assert summary["pair_quality"]["infrastructure_failed_pairs"] == 1


def test_rule_shift_eval_cli_defaults_are_bounded_and_explicit():
    args = build_parser().parse_args(
        ["sandbox", "rule-shift-eval", "--main-brain", "fake/model"]
    )

    assert args.sandbox_command == "rule-shift-eval"
    assert args.repeats == 2
    assert args.shift_after == 3
    assert args.max_steps == 10
    assert args.max_cost_usd == 0.08
    assert args.max_total_cost_usd is None
    assert args.shared_prefix_turns == 1
    assert args.tool_result_lifecycle == "compact"
