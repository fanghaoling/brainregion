from __future__ import annotations

import asyncio

from brainregion.providers.base import ModelResponse
from brainregion.sandbox.prefix_replay import ModelPrefixTape, PrefixReplayBackend


class _Backend:
    def __init__(self) -> None:
        self.calls = 0

    async def complete_messages(self, messages, **kwargs):
        self.calls += 1
        return ModelResponse(
            model=kwargs["model"],
            content=f"response-{self.calls}",
            usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            cost_usd=0.01,
            cost_source="provider",
        )


def _request(content: str = "same") -> tuple[list[dict], dict]:
    return ([{"role": "user", "content": content}], {"model": "main", "max_tokens": 50})


def test_prefix_backend_replays_deep_copied_response_without_provider_call():
    backend = _Backend()
    tape = ModelPrefixTape(turn_limit=2)
    capture = PrefixReplayBackend(backend, tape, role="capture")
    messages, kwargs = _request()

    captured = asyncio.run(capture.complete_messages(messages, **kwargs))
    replay = PrefixReplayBackend(backend, tape, role="replay")
    replayed = asyncio.run(replay.complete_messages(messages, **kwargs))

    assert backend.calls == 1
    assert replayed.content == captured.content
    assert replayed is not captured
    replayed.usage["input_tokens"] = 999
    assert captured.usage["input_tokens"] == 10
    assert capture.public_metrics()["captured_calls"] == 1
    assert replay.public_metrics()["replayed_calls"] == 1
    assert replay.public_metrics()["provider_calls"] == 0
    assert replay.public_metrics()["provider_cost_usd"] == 0.0
    assert replay.public_metrics()["replayed_accounted_cost_usd"] == 0.01
    assert replay.public_metrics()["contains_response_content"] is False


def test_prefix_backend_request_mismatch_falls_back_to_provider_and_is_audited():
    backend = _Backend()
    tape = ModelPrefixTape(turn_limit=1)
    capture = PrefixReplayBackend(backend, tape, role="capture")
    messages, kwargs = _request("capture")
    asyncio.run(capture.complete_messages(messages, **kwargs))

    replay = PrefixReplayBackend(backend, tape, role="replay")
    different_messages, kwargs = _request("different")
    response = asyncio.run(replay.complete_messages(different_messages, **kwargs))

    assert backend.calls == 2
    assert response.content == "response-2"
    metrics = replay.public_metrics()
    assert metrics["replayed_calls"] == 0
    assert metrics["replay_mismatches"] == 1
    assert metrics["provider_calls"] == 1
    assert metrics["accounted_calls"] == 1


def test_prefix_backend_shortfall_falls_back_without_inventing_a_response():
    backend = _Backend()
    replay = PrefixReplayBackend(backend, ModelPrefixTape(turn_limit=1), role="replay")
    messages, kwargs = _request()

    response = asyncio.run(replay.complete_messages(messages, **kwargs))

    assert response.content == "response-1"
    assert replay.public_metrics()["replay_shortfalls"] == 1
    assert replay.public_metrics()["provider_calls"] == 1
