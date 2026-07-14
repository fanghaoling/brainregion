from __future__ import annotations

from brainregion.sandbox.input_attribution import provider_messages
from brainregion.sandbox.tool_result_lifecycle import (
    ToolResultLifecycle,
    tool_result_message,
)


def _result(tool: str, step: int, *, error: bool = False) -> dict:
    return tool_result_message(
        f'<tool_result tool="{tool}">\n' + ("evidence " * 200) + "\n</tool_result>",
        tool=tool,
        step=step,
        target_kind="path",
        target_fingerprint=f"target-{step}",
        error=error,
    )


def test_search_result_gets_one_guaranteed_consumer_turn_then_receipt():
    messages = [_result("search_text", 0)]
    lifecycle = ToolResultLifecycle(mode="compact")

    lifecycle.apply(messages, next_step=1)
    assert "tool_result_receipt" not in messages[0]["content"]

    lifecycle.apply(messages, next_step=2)
    assert "tool_result_receipt" in messages[0]["content"]
    assert "evidence evidence" not in messages[0]["content"]
    assert provider_messages(messages)[0] == {
        "role": "user",
        "content": messages[0]["content"],
    }
    metrics = lifecycle.public_metrics()
    assert metrics["compacted_results"] == 1
    assert metrics["compacted_by_tool"] == {"search_text": 1}
    assert metrics["body_estimated_tokens_removed"] > 0
    assert metrics["estimated_input_tokens_avoided"] > 0
    assert metrics["contains_result_content"] is False


def test_recent_read_working_set_stays_full_while_older_reads_unload():
    messages = [_result("read_text", step) for step in range(4)]
    lifecycle = ToolResultLifecycle(mode="compact", live_read_results=2)

    lifecycle.apply(messages, next_step=5)

    assert ["tool_result_receipt" in message["content"] for message in messages] == [
        True,
        True,
        False,
        False,
    ]
    assert lifecycle.public_metrics()["active_receipts"] == 2


def test_patch_and_latest_verification_are_pinned_until_superseded():
    patch = _result("apply_text_patch", 0)
    messages = [patch]
    lifecycle = ToolResultLifecycle(mode="compact")

    lifecycle.apply(messages, next_step=2)
    assert "tool_result_receipt" not in patch["content"]

    check = _result("workspace_run_check", 2)
    messages.append(check)
    lifecycle.apply(messages, next_step=4)
    assert "tool_result_receipt" in patch["content"]
    assert "tool_result_receipt" not in check["content"]

    later_patch = _result("apply_text_patch", 4)
    messages.append(later_patch)
    lifecycle.apply(messages, next_step=6)
    assert "tool_result_receipt" in check["content"]
    assert "tool_result_receipt" not in later_patch["content"]


def test_error_result_gets_extra_recovery_turn_and_full_mode_is_inert():
    error_result = _result("search_text", 0, error=True)
    compact = ToolResultLifecycle(mode="compact")

    compact.apply([error_result], next_step=2)
    assert "tool_result_receipt" not in error_result["content"]
    compact.apply([error_result], next_step=3)
    assert "tool_result_receipt" in error_result["content"]

    untouched = _result("search_text", 0)
    full = ToolResultLifecycle(mode="full")
    full.apply([untouched], next_step=10)
    assert "tool_result_receipt" not in untouched["content"]
    assert full.public_metrics()["enabled"] is False
    assert full.public_metrics()["tool_results_observed"] == 1
