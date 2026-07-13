"""Metadata-only context export authorization tests."""

from __future__ import annotations

import pytest

from brainregion.core.context import ContextBlock
from brainregion.core.context_export import (
    endpoint_context_trust,
    evaluate_context_export,
)


def _block(source="memory", sensitivity=None):
    metadata = {} if sensitivity is None else {"sensitivity": sensitivity}
    return ContextBlock(source=source, title="t", content="never inspected", metadata=metadata)


def test_off_is_true_bypass_without_classifying_blocks(monkeypatch):
    from brainregion.core import context_export

    monkeypatch.setattr(
        context_export,
        "_classify_block",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("classified")),
    )

    decision = evaluate_context_export(
        (_block(),),
        policy={"mode": "off"},
        endpoint_trust="invalid-is-not-read",
    )

    assert decision.permits_call is True
    assert decision.to_dict() == {
        "mode": "off",
        "evaluated": False,
        "action": "bypass",
        "allowed": True,
        "endpoint_trust": "uninspected",
        "highest_sensitivity": None,
        "block_counts": {},
        "denied_sensitivities": [],
        "reason": "policy_off",
        "context_modified": False,
    }


def test_audit_observes_private_memory_but_never_blocks():
    decision = evaluate_context_export(
        (_block(source="memory"),),
        policy={"mode": "audit"},
        endpoint_trust="external",
    )

    assert decision.action == "would_deny"
    assert decision.allowed is False
    assert decision.permits_call is True
    assert decision.to_dict()["block_counts"] == {"private": 1}


def test_enforce_denies_private_context_to_external_endpoint():
    decision = evaluate_context_export(
        (_block(sensitivity="private"),),
        policy={"mode": "enforce"},
        endpoint_trust="external",
    )

    assert decision.action == "deny"
    assert decision.permits_call is False
    assert decision.denied_sensitivities == ("private",)


def test_enforce_allows_original_private_context_for_trusted_endpoint():
    block = _block(sensitivity="private")
    decision = evaluate_context_export(
        (block,),
        policy={"mode": "enforce"},
        endpoint_trust="trusted",
    )

    assert decision.action == "allow"
    assert decision.permits_call is True
    assert block.content == "never inspected"
    assert decision.to_dict()["context_modified"] is False


def test_endpoint_trust_supports_inline_and_policy_override():
    endpoints = {"relay": {"context_trust": "trusted"}}

    assert endpoint_context_trust("relay", endpoints, {}) == "trusted"
    assert endpoint_context_trust("relay", endpoints, {"endpoint_trust": {"relay": "local"}}) == "local"
    assert endpoint_context_trust(None, endpoints, {}) == "external"


@pytest.mark.parametrize("mode", ["filter", "enabled", "strict"])
def test_unknown_mode_fails_fast(mode):
    with pytest.raises(ValueError, match="off, audit, or enforce"):
        evaluate_context_export((), policy={"mode": mode}, endpoint_trust="external")
