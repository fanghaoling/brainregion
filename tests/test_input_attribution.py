from __future__ import annotations

from brainregion.sandbox.input_attribution import (
    attributed_message,
    capture_input_attribution,
    compound_message,
    merge_input_attributions,
    provider_messages,
    reconcile_input_attribution,
)


def test_compound_message_attributes_parts_without_crossing_provider_boundary():
    messages = [
        compound_message(
            "system",
            [
                ("system", "base system"),
                ("scaffold", " runtime checkpoint"),
            ],
        ),
        attributed_message("user", "do the task", "task"),
    ]

    captured = capture_input_attribution(messages)
    outbound = provider_messages(messages)

    assert outbound == [
        {"role": "system", "content": "base system runtime checkpoint"},
        {"role": "user", "content": "do the task"},
    ]
    assert captured["categories"]["system"]["estimated_tokens"] > 0
    assert captured["categories"]["scaffold"]["estimated_tokens"] > 0
    assert captured["categories"]["task"]["estimated_tokens"] > 0
    assert captured["categories"]["protocol"]["estimated_tokens"] == 11
    assert captured["contains_content"] is False


def test_reconciliation_allocates_exact_provider_total_without_content():
    captured = capture_input_attribution(
        [
            attributed_message("system", "s" * 40, "system"),
            attributed_message("user", "t" * 80, "tool_transcript"),
            attributed_message("user", "c" * 20, "checkpoint"),
        ]
    )

    report = reconcile_input_attribution(
        captured,
        {
            "input_tokens": 101,
            "prompt_tokens_details": {"cached_tokens": 23},
        },
    )

    assert report["actual_input_tokens"] == 101
    assert report["cached_input_tokens"] == 23
    assert report["allocation_status"] == "provider_aligned"
    assert sum(
        category["actual_input_tokens"] for category in report["categories"].values()
    ) == 101
    assert report["categories"]["tool_transcript"]["actual_input_tokens"] > report[
        "categories"
    ]["checkpoint"]["actual_input_tokens"]
    assert report["contains_content"] is False
    assert report["contains_reasoning"] is False
    assert report["contains_tool_results"] is False


def test_reconciliation_reads_responses_cached_tokens():
    captured = capture_input_attribution(
        [attributed_message("user", "inspect the repository", "task")]
    )

    report = reconcile_input_attribution(
        captured,
        {
            "input_tokens": 8_242,
            "input_tokens_details": {"cached_tokens": 5_888},
        },
    )

    assert report["actual_input_tokens"] == 8_242
    assert report["cached_input_tokens"] == 5_888


def test_estimate_only_and_aggregate_reports_remain_additive():
    first = reconcile_input_attribution(
        capture_input_attribution([attributed_message("user", "alpha", "task")]),
        {"input_tokens": 0},
    )
    second = reconcile_input_attribution(
        capture_input_attribution(
            [attributed_message("user", "beta" * 20, "tool_transcript")]
        ),
        {"input_tokens": 50},
    )

    merged = merge_input_attributions([first, second])

    assert first["allocation_status"] == "estimate_only"
    assert merged["calls"] == 2
    assert merged["provider_reported_calls"] == 1
    assert merged["actual_input_tokens"] == 50
    assert sum(
        category["actual_input_tokens"] for category in merged["categories"].values()
    ) == 50
    assert merged["categories"]["task"]["estimated_tokens"] > 0
    assert merged["categories"]["tool_transcript"]["actual_input_tokens"] > 0
