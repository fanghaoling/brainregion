from __future__ import annotations

import pytest

from brainregion.sandbox.epistemic_ledger import EpistemicLedger


def _claim(
    hypothesis_id: str,
    *,
    rule: str = "action1 advances the marker",
    replaces: str = "",
    change_scale: str = "local",
    level_delta: int = 0,
    state: str = "NOT_FINISHED",
) -> dict:
    return {
        "hypothesis_id": hypothesis_id,
        "rule": rule,
        "scope": "current level",
        "replaces": replaces,
        "predicts": {
            "change_scale": change_scale,
            "level_delta": level_delta,
            "state": state,
        },
    }


def _resolve(ledger: EpistemicLedger, raw: dict, *, change_scale: str = "local"):
    prepared = ledger.prepare(raw, action="action1")
    return ledger.resolve(
        prepared,
        change_scale=change_scale,
        changed_cells=1 if change_scale != "none" else 0,
        total_cells=100,
        level_delta=0,
        state="NOT_FINISHED",
    )


def test_hypothesis_requires_two_observed_matches_before_support():
    ledger = EpistemicLedger()

    first = _resolve(ledger, _claim("h1"))
    second = _resolve(ledger, _claim("h1"))

    assert first["matched"] is True and first["status"] == "open"
    assert second["matched"] is True and second["status"] == "supported"
    assert ledger.public_metrics()["prediction_accuracy"] == 1.0
    assert ledger.public_metrics()["supported_hypotheses"] == 1


def test_surprise_refutes_claim_and_hides_rule_from_working_set():
    ledger = EpistemicLedger()

    result = _resolve(ledger, _claim("wrong"), change_scale="none")
    view = ledger.working_view()

    assert result["matched"] is False and result["status"] == "refuted"
    assert view["active_hypotheses"] == []
    assert view["suppressed_hypotheses"][0]["status"] == "refuted"
    assert "rule" not in view["suppressed_hypotheses"][0]
    assert ledger.public_metrics()["surprises"] == 1


def test_replacement_does_not_supersede_old_claim_until_independently_verified():
    ledger = EpistemicLedger()
    _resolve(ledger, _claim("old"))
    replacement = _claim("new", rule="action1 moves a different object", replaces="old")

    first = _resolve(ledger, replacement)
    assert first["status"] == "candidate"
    assert ledger.hypotheses["old"].status == "open"

    second = _resolve(ledger, replacement)
    assert second["status"] == "supported"
    assert ledger.hypotheses["old"].status == "superseded"
    assert ledger.hypotheses["old"].superseded_by == "new"
    assert ledger.public_metrics()["verified_insights"] == 1
    assert ledger.public_metrics()["supersessions"] == 1


def test_candidate_that_later_fails_is_counted_without_suppressing_old_claim():
    ledger = EpistemicLedger()
    _resolve(ledger, _claim("old"))
    replacement = _claim("new", rule="action1 always changes the frame", replaces="old")

    _resolve(ledger, replacement)
    failed = _resolve(ledger, replacement, change_scale="none")

    assert failed["status"] == "refuted"
    assert ledger.hypotheses["old"].status == "open"
    assert ledger.public_metrics()["false_insights"] == 1


def test_prepare_is_transactional_and_rejects_silent_rule_rewrite():
    ledger = EpistemicLedger()
    _resolve(ledger, _claim("h1"))

    with pytest.raises(ValueError, match="silently change"):
        ledger.prepare(_claim("h1", rule="a different rule"), action="action1")

    assert ledger.public_metrics()["predictions"] == 1
    assert ledger.hypotheses["h1"].rule == "action1 advances the marker"


def test_replacement_must_reference_existing_hypothesis():
    ledger = EpistemicLedger()

    with pytest.raises(ValueError, match="reference an existing"):
        ledger.prepare(_claim("new", replaces="missing"), action="action1")


def test_placeholder_and_duplicate_live_rules_are_rejected():
    ledger = EpistemicLedger()

    with pytest.raises(ValueError, match="placeholder"):
        ledger.prepare(_claim("h1", rule="bounded public rule"), action="action1")
    with pytest.raises(ValueError, match="schema placeholder"):
        ledger.prepare(_claim("STABLE_ID"), action="action1")

    _resolve(ledger, _claim("h1"))
    with pytest.raises(ValueError, match="reuse hypothesis_id 'h1'"):
        ledger.prepare(_claim("h2"), action="action1")


def test_exact_refuted_rule_cannot_be_revived_under_a_new_id():
    ledger = EpistemicLedger()
    _resolve(ledger, _claim("wrong"), change_scale="none")

    with pytest.raises(ValueError, match="matches refuted hypothesis 'wrong'"):
        ledger.prepare(_claim("same-wrong-rule"), action="action1")


def test_reset_removes_episode_state_and_metrics():
    ledger = EpistemicLedger()
    _resolve(ledger, _claim("h1"))

    ledger.reset()

    assert ledger.working_view()["active_hypotheses"] == []
    assert ledger.public_metrics()["predictions"] == 0
    assert ledger.public_metrics()["persistent"] is False
