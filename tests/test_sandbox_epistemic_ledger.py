from __future__ import annotations

import pytest

from brainregion.sandbox.epistemic_ledger import EpistemicLedger


def _claim(
    hypothesis_id: str,
    *,
    rule: str = "action1 advances the marker",
    replaces: str = "",
    frame_change: str = "changed",
    level_delta: int = 0,
    state: str = "NOT_FINISHED",
) -> dict:
    return {
        "hypothesis_id": hypothesis_id,
        "rule": rule,
        "scope": "current level",
        "replaces": replaces,
        "predicts": {
            "frame_change": frame_change,
            "level_delta": level_delta,
            "state": state,
        },
    }


def _resolve(ledger: EpistemicLedger, raw: dict, *, changed: bool = True):
    prepared = ledger.prepare(raw, action="action1")
    return ledger.resolve(
        prepared,
        frame_changed=changed,
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


def test_surprise_refutes_claim_and_hides_rule_from_working_set():
    ledger = EpistemicLedger()

    result = _resolve(ledger, _claim("wrong"), changed=False)
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
    failed = _resolve(ledger, replacement, changed=False)

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


def test_reset_removes_episode_state_and_metrics():
    ledger = EpistemicLedger()
    _resolve(ledger, _claim("h1"))

    ledger.reset()

    assert ledger.working_view()["active_hypotheses"] == []
    assert ledger.public_metrics()["predictions"] == 0
    assert ledger.public_metrics()["persistent"] is False
