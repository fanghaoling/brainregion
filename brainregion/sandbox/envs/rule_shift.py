"""Deterministic visual probe for evidence-gated rule replacement."""

from __future__ import annotations

import json
from typing import Any

from ..epistemic_ledger import (
    EpistemicLedger,
    classify_change_scale,
    epistemic_action_contract,
)


class RuleShiftEnv:
    """Hide one action-effect regime shift behind otherwise neutral observations."""

    action_vocab = ("action1",)
    supports_action_data = True
    supports_epistemic_update = True
    ego_actions = False
    size = 10

    def __init__(self, *, shift_after: int = 3, distractor_steps: int = 0) -> None:
        if isinstance(shift_after, bool) or not isinstance(shift_after, int):
            raise ValueError("shift_after must be an integer")
        if shift_after < 2:
            raise ValueError("shift_after must be at least 2")
        if (
            isinstance(distractor_steps, bool)
            or not isinstance(distractor_steps, int)
            or distractor_steps < 0
        ):
            raise ValueError("distractor_steps must be a non-negative integer")
        self.shift_after = shift_after
        self.distractor_steps = distractor_steps
        self.action_vocab = (
            ("action1", "action2") if distractor_steps > 0 else ("action1",)
        )
        self.epistemic_ledger = EpistemicLedger()
        self._cells: list[list[int]] = []
        self._action_count = 0
        self._distractor_remaining = 0
        self._terminated = False
        self.total_reward = 0.0
        self.frames: list[str] = []
        self.action_trace: list[dict[str, Any]] = []
        self.reset()

    @property
    def solved(self) -> bool:
        if not self._terminated:
            return False
        supported_fingerprints: set[str] = set()
        refuted_after_support: set[str] = set()
        for event in self.action_trace:
            fingerprint = str(event.get("epistemic_hypothesis_fingerprint") or "")
            status = event.get("epistemic_status")
            if status == "supported":
                supported_fingerprints.add(fingerprint)
            elif status == "refuted" and fingerprint in supported_fingerprints:
                refuted_after_support.add(fingerprint)
        for hypothesis in self.epistemic_ledger.hypotheses.values():
            if hypothesis.status != "supported" or not hypothesis.replaces:
                continue
            replaced = self.epistemic_ledger.hypotheses[hypothesis.replaces]
            if replaced.fingerprint in refuted_after_support:
                return True
        return False

    def _observation_payload(self) -> dict[str, Any]:
        return {
            "state": "WIN" if self._terminated else "NOT_FINISHED",
            "available_actions": self._available_actions(),
            "frame_encoding": "base36_grid",
            "frame": ["".join(str(cell) for cell in row) for row in self._cells],
            "epistemic_ledger": self.epistemic_ledger.working_view(),
        }

    def observation(self) -> str:
        return json.dumps(self._observation_payload(), ensure_ascii=True, separators=(",", ":"))

    def snapshot(self) -> dict[str, Any]:
        return self._observation_payload()

    def render(self) -> str:
        return self.observation()

    def reset(self, *, seed: int | None = None) -> str:
        if seed is not None:
            raise ValueError("RuleShiftEnv is deterministic and does not accept a seed")
        self._cells = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self._action_count = 0
        self._distractor_remaining = 0
        self._terminated = False
        self.total_reward = 0.0
        self.epistemic_ledger.reset()
        rendered = self.observation()
        self.frames = [rendered]
        self.action_trace = []
        return rendered

    def step(
        self,
        action: str,
        *,
        data: dict[str, Any] | None = None,
        epistemic_update: dict[str, Any] | None = None,
    ) -> tuple[str, float, bool, dict[str, Any]]:
        if self._terminated:
            return self.observation(), 0.0, True, {"already_done": True}
        normalized = str(action or "").strip().lower()
        if normalized not in self.action_vocab:
            raise ValueError(f"unknown rule-shift action {action!r}")
        if normalized not in self._available_actions():
            raise ValueError(
                f"rule-shift action {action!r} is not currently available; "
                f"use {self._available_actions()}"
            )
        if data not in (None, {}):
            raise ValueError("RuleShiftEnv does not accept action data")

        prepared = self.epistemic_ledger.prepare(epistemic_update, action=normalized)
        before = [row[:] for row in self._cells]
        if normalized == "action1":
            self._action_count += 1
            if self._action_count <= self.shift_after:
                changed_positions = ((0, 0), (1, 0))
            else:
                changed_positions = tuple(
                    (index % self.size, index // self.size) for index in range(30)
                )
            if (
                self.distractor_steps > 0
                and self._action_count == self.shift_after + 1
            ):
                self._distractor_remaining = self.distractor_steps
        else:
            changed_positions = ()
            self._distractor_remaining -= 1
        for x, y in changed_positions:
            self._cells[y][x] = 1 - self._cells[y][x]

        changed_cells = sum(
            before[y][x] != self._cells[y][x]
            for y in range(self.size)
            for x in range(self.size)
        )
        total_cells = self.size * self.size
        change_scale = classify_change_scale(changed_cells, total_cells)
        self._terminated = (
            self._action_count >= self.shift_after + 3
            and self._distractor_remaining == 0
        )
        state = "WIN" if self._terminated else "NOT_FINISHED"
        feedback = self.epistemic_ledger.resolve(
            prepared,
            change_scale=change_scale,
            changed_cells=changed_cells,
            total_cells=total_cells,
            level_delta=0,
            state=state,
        )
        reward = 1.0 if self.solved else 0.0
        self.total_reward += reward
        rendered = self.observation()
        self.frames.append(rendered)
        self.action_trace.append(
            {
                "index": len(self.action_trace),
                "action": normalized,
                "changed_cells": changed_cells,
                "change_scale": change_scale,
                "state": state,
                "epistemic_prediction_matched": feedback["matched"],
                "epistemic_hypothesis_fingerprint": feedback["hypothesis_fingerprint"],
                "epistemic_status": feedback["status"],
                "epistemic_mismatch_fields": feedback["mismatch_fields"],
            }
        )
        info = {
            "state": state,
            "available_actions": self._available_actions(),
            "epistemic_feedback": feedback,
        }
        return rendered, reward, self._terminated, info

    def _available_actions(self) -> list[str]:
        if self._distractor_remaining > 0:
            return ["action2"]
        return ["action1"]

    def build_system_prompt(self, goal: str, *, navigation: bool = False) -> str:
        if navigation:
            raise ValueError("RuleShiftEnv does not support navigation regions")
        epistemic_prompt, act_example = epistemic_action_contract()
        distractor_prompt = ""
        if self.distractor_steps > 0:
            distractor_prompt = (
                "The available action may temporarily change to action2. Treat action2 as an independent "
                "effect with its own hypothesis id; its observations do not revise an action1 rule. "
                "When action1 becomes available again, recover the earlier action1 evidence before predicting.\n"
            )
        return (
            "You control an unfamiliar deterministic visual environment with no provided action rules. "
            "Infer action effects only from observations. Do not assume an effect stays stable after contrary "
            "evidence.\n"
            f"Objective: {goal}\n"
            "Build a supported falsifiable rule through repeated predictions. If later evidence refutes that "
            "rule, create a genuinely revised hypothesis that replaces it and verify the replacement. "
            "The environment ends automatically after a bounded interaction.\n"
            "The current observation is provided before the first decision. Act directly from the latest frame; "
            "the observe tool is unnecessary.\n"
            f"{epistemic_prompt}"
            f"{distractor_prompt}"
            "For this probe, the hypothesis concerns visible change scale only: always set level_delta to 0 "
            "and state to an empty string. Terminal timing is not evidence for the action-effect rule.\n"
            f"Reply with exactly one JSON object per turn. Act with {act_example}. "
            "Use only actions listed by the latest observation. Mark done after termination. "
            "The base36 frame is an exact color-index grid, not text."
        )

    def close(self) -> None:
        return None


__all__ = ["RuleShiftEnv"]
