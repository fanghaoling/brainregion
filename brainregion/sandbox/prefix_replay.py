"""In-memory capture/replay for a bounded model-response prefix."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

PrefixReplayRole = Literal["disabled", "capture", "replay"]


@dataclass
class ModelPrefixTape:
    """Private response tape shared by one matched pair; never serialize it."""

    turn_limit: int
    request_fingerprints: list[str] = field(default_factory=list, repr=False)
    responses: list[Any] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.turn_limit, bool)
            or not isinstance(self.turn_limit, int)
            or self.turn_limit < 0
        ):
            raise ValueError("model prefix turn_limit must be a non-negative integer")


class PrefixReplayBackend:
    """Capture or replay an exact response prefix while auditing real provider use."""

    def __init__(
        self,
        backend: Any,
        tape: ModelPrefixTape,
        *,
        role: PrefixReplayRole,
    ) -> None:
        if role not in {"disabled", "capture", "replay"}:
            raise ValueError(f"unknown prefix replay role: {role!r}")
        self._backend = backend
        self._tape = tape
        self.role = role
        self.accounted_calls = 0
        self.provider_calls = 0
        self.captured_calls = 0
        self.replayed_calls = 0
        self.replay_mismatches = 0
        self.replay_shortfalls = 0
        self.provider_cost_usd = 0.0
        self.replayed_accounted_cost_usd = 0.0

    async def complete_messages(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        turn = self.accounted_calls
        self.accounted_calls += 1
        fingerprint = _request_fingerprint(messages, kwargs)

        if self.role == "replay" and turn < self._tape.turn_limit:
            if turn < len(self._tape.responses):
                if fingerprint == self._tape.request_fingerprints[turn]:
                    response = copy.deepcopy(self._tape.responses[turn])
                    self.replayed_calls += 1
                    self.replayed_accounted_cost_usd += float(
                        getattr(response, "cost_usd", None) or 0.0
                    )
                    return response
                self.replay_mismatches += 1
            else:
                self.replay_shortfalls += 1

        response = await self._backend.complete_messages(messages, **kwargs)
        self.provider_calls += 1
        self.provider_cost_usd += float(getattr(response, "cost_usd", None) or 0.0)
        if self.role == "capture" and turn < self._tape.turn_limit:
            self._tape.request_fingerprints.append(fingerprint)
            self._tape.responses.append(copy.deepcopy(response))
            self.captured_calls += 1
        return response

    def public_metrics(self) -> dict[str, Any]:
        return {
            "enabled": self.role != "disabled" and self._tape.turn_limit > 0,
            "role": self.role,
            "turn_limit": self._tape.turn_limit,
            "accounted_calls": self.accounted_calls,
            "provider_calls": self.provider_calls,
            "captured_calls": self.captured_calls,
            "replayed_calls": self.replayed_calls,
            "replay_mismatches": self.replay_mismatches,
            "replay_shortfalls": self.replay_shortfalls,
            "provider_cost_usd": self.provider_cost_usd,
            "replayed_accounted_cost_usd": self.replayed_accounted_cost_usd,
            "accounted_cost_usd": (
                self.provider_cost_usd + self.replayed_accounted_cost_usd
            ),
            "contains_request_content": False,
            "contains_response_content": False,
            "contains_reasoning": False,
        }


def _request_fingerprint(messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"messages": messages, "request": kwargs},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["ModelPrefixTape", "PrefixReplayBackend", "PrefixReplayRole"]
