"""Optional ARC-AGI-3 adapter for BrainRegion's generic environment loop.

The module deliberately avoids importing ``arc_agi`` or ``arcengine`` at import
time. BrainRegion keeps Python 3.10 support while the official SDK currently
requires Python 3.12 or newer.
"""
from __future__ import annotations

import json
import logging
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..epistemic_ledger import EpistemicLedger


_BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"
_TERMINAL_STATES = frozenset({"WIN", "GAME_OVER"})


def _state_name(frame: Any) -> str:
    state = getattr(frame, "state", "NOT_PLAYED")
    return str(getattr(state, "value", getattr(state, "name", state)))


def _frame_rows(frame: Any) -> tuple[str, list[Any], list[int]]:
    raw_frames = list(getattr(frame, "frame", ()) or ())
    if not raw_frames:
        return "none", [], []
    raw = raw_frames[-1]
    rows = raw.tolist() if hasattr(raw, "tolist") else raw
    normalized = [[int(value) for value in row] for row in rows]
    palette = sorted({value for row in normalized for value in row})
    if all(0 <= value < len(_BASE36) for value in palette):
        return (
            "base36_grid",
            ["".join(_BASE36[value] for value in row) for row in normalized],
            palette,
        )
    return "integer_grid", normalized, palette


def _frame_change_summary(before: Any, after: Any) -> tuple[int, int, str]:
    """Return changed cells, frame size, and a resolution-independent scale."""

    _before_encoding, before_rows, _before_palette = _frame_rows(before)
    _after_encoding, after_rows, _after_palette = _frame_rows(after)
    before_cells = {
        (x, y): value
        for y, row in enumerate(before_rows)
        for x, value in enumerate(row)
    }
    after_cells = {
        (x, y): value
        for y, row in enumerate(after_rows)
        for x, value in enumerate(row)
    }
    coordinates = set(before_cells) | set(after_cells)
    changed_cells = sum(before_cells.get(cell) != after_cells.get(cell) for cell in coordinates)
    total_cells = len(coordinates)
    if changed_cells == 0:
        return 0, total_cells, "none"
    local_limit = max(1, (total_cells + 49) // 50)
    regional_limit = max(local_limit, (total_cells + 3) // 4)
    if changed_cells <= local_limit:
        return changed_cells, total_cells, "local"
    if changed_cells <= regional_limit:
        return changed_cells, total_cells, "regional"
    return changed_cells, total_cells, "global"


@dataclass
class ArcAgiEnv:
    """Adapt an official EnvironmentWrapper without embedding game knowledge."""

    wrapper: Any
    game_id: str
    arcade: Any | None = None
    frames: list[str] = field(default_factory=list)
    action_trace: list[dict[str, Any]] = field(default_factory=list)
    total_reward: float = 0.0
    supports_action_data: bool = True
    visibility_radius: None = None
    ego_actions: bool = False
    _last_frame: Any | None = None
    _terminated: bool = False
    epistemic_ledger: EpistemicLedger | None = None

    @classmethod
    def create(
        cls,
        game_id: str,
        *,
        seed: int = 0,
        root: str | Path = ".brain-region/arc-agi",
        api_key: str = "",
        epistemic_ledger: EpistemicLedger | None = None,
    ) -> "ArcAgiEnv":
        """Create an SDK-backed public environment with all artifacts isolated."""

        try:
            from arc_agi import Arcade
        except ImportError as exc:
            raise RuntimeError(
                "ARC-AGI-3 support requires Python 3.12+ and arc-agi==0.9.9"
            ) from exc

        base = Path(root)
        logger = logging.getLogger("brainregion.arc_agi.sdk")
        logger.handlers = [logging.NullHandler()]
        logger.propagate = False
        logger.setLevel(logging.WARNING)
        arcade = Arcade(
            arc_api_key=api_key,
            environments_dir=str(base / "environment_files"),
            recordings_dir=str(base / "recordings"),
            logger=logger,
        )
        initial_frames: list[Any] = []
        wrapper = arcade.make(
            game_id,
            seed=seed,
            include_frame_data=True,
            save_recording=True,
            renderer=lambda _steps, frame: initial_frames.append(frame),
        )
        if wrapper is None:
            arcade.close_scorecard()
            raise RuntimeError(f"ARC-AGI-3 game could not be created: {game_id}")
        env = cls(
            wrapper=wrapper,
            game_id=game_id,
            arcade=arcade,
            epistemic_ledger=epistemic_ledger,
        )
        if initial_frames:
            env._set_initial_frame(initial_frames[-1])
        else:
            env.reset()
        return env

    @property
    def action_vocab(self) -> tuple[str, ...]:
        return tuple(action.name.lower() for action in self.wrapper.action_space)

    @property
    def solved(self) -> bool:
        return _state_name(self._last_frame) == "WIN" if self._last_frame is not None else False

    @property
    def supports_epistemic_update(self) -> bool:
        return self.epistemic_ledger is not None

    def _action_descriptors(self) -> list[dict[str, Any]]:
        return [
            {
                "name": action.name.lower(),
                "requires_data": bool(action.is_complex()),
                "data_schema": {"x": "integer 0..63", "y": "integer 0..63"}
                if action.is_complex()
                else None,
            }
            for action in self.wrapper.action_space
        ]

    def _snapshot(self) -> dict[str, Any]:
        if self._last_frame is None:
            return {
                "game_id": self.game_id,
                "state": "NOT_PLAYED",
                "available_actions": [],
                "frame_encoding": "none",
                "frame": [],
            }
        encoding, rows, palette = _frame_rows(self._last_frame)
        snapshot = {
            "game_id": str(getattr(self._last_frame, "game_id", "") or self.game_id),
            "state": _state_name(self._last_frame),
            "levels_completed": int(getattr(self._last_frame, "levels_completed", 0) or 0),
            "win_levels": int(getattr(self._last_frame, "win_levels", 0) or 0),
            "available_actions": self._action_descriptors(),
            "frame_encoding": encoding,
            "palette": palette,
            "frame": rows,
        }
        if self.epistemic_ledger is not None:
            snapshot["epistemic_ledger"] = self.epistemic_ledger.working_view()
        return snapshot

    def observation(self) -> str:
        return json.dumps(self._snapshot(), ensure_ascii=True, separators=(",", ":"))

    def snapshot(self) -> dict[str, Any]:
        return self._snapshot()

    def render(self) -> str:
        return self.observation()

    def reset(self, *, seed: int | None = None) -> str:
        if seed is not None:
            raise ValueError("ARC-AGI-3 seed is fixed when the SDK wrapper is created")
        frame = self.wrapper.reset()
        if frame is None:
            raise RuntimeError("ARC-AGI-3 reset returned no frame")
        return self._set_initial_frame(frame)

    def _set_initial_frame(self, frame: Any) -> str:
        self._last_frame = frame
        self._terminated = _state_name(frame) in _TERMINAL_STATES
        self.total_reward = 0.0
        if self.epistemic_ledger is not None:
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
        actions = {candidate.name.lower(): candidate for candidate in self.wrapper.action_space}
        selected = actions.get(normalized)
        if selected is None:
            raise ValueError(
                f"unknown ARC-AGI-3 action {action!r}; available: {sorted(actions)}"
            )
        payload = data or {}
        if selected.is_complex():
            if data is None:
                raise ValueError(f"ARC-AGI-3 action {normalized} requires x/y data")
            selected.validate_data(payload)
        elif data not in (None, {}):
            raise ValueError(f"ARC-AGI-3 action {normalized} does not accept data")

        prepared_prediction = None
        if self.epistemic_ledger is not None:
            prepared_prediction = self.epistemic_ledger.prepare(
                epistemic_update,
                action=normalized,
            )
        elif epistemic_update is not None:
            raise ValueError("ARC-AGI-3 epistemic ledger is not enabled")

        before_frame = self._last_frame
        before = int(getattr(before_frame, "levels_completed", 0) or 0)
        before_hash = self._frame_hash()
        frame = self.wrapper.step(selected, data=payload if selected.is_complex() else None)
        if frame is None:
            raise RuntimeError("ARC-AGI-3 step returned no frame")
        self._last_frame = frame
        state = _state_name(frame)
        self._terminated = state in _TERMINAL_STATES
        completed = int(getattr(frame, "levels_completed", 0) or 0)
        reward = float(max(0, completed - before))
        self.total_reward += reward
        after_hash = self._frame_hash()
        changed_cells, total_cells, change_scale = _frame_change_summary(before_frame, frame)
        epistemic_feedback = None
        if self.epistemic_ledger is not None and prepared_prediction is not None:
            epistemic_feedback = self.epistemic_ledger.resolve(
                prepared_prediction,
                change_scale=change_scale,
                changed_cells=changed_cells,
                total_cells=total_cells,
                level_delta=max(0, completed - before),
                state=state,
            )
        rendered = self.observation()
        self.frames.append(rendered)
        trace = {
            "index": len(self.action_trace),
            "action": normalized,
            "uses_data": bool(payload),
            "frame_changed": before_hash != after_hash,
            "changed_cells": changed_cells,
            "change_scale": change_scale,
            "frame_hash": after_hash,
            "state": state,
            "levels_completed": completed,
            "available_action_count": len(self.action_vocab),
        }
        if epistemic_feedback is not None:
            trace["epistemic_prediction_matched"] = epistemic_feedback["matched"]
            trace["epistemic_hypothesis_fingerprint"] = epistemic_feedback[
                "hypothesis_fingerprint"
            ]
            trace["epistemic_status"] = epistemic_feedback["status"]
        self.action_trace.append(trace)
        info = {
            "state": state,
            "levels_completed": completed,
            "win_levels": int(getattr(frame, "win_levels", 0) or 0),
            "available_actions": list(self.action_vocab),
        }
        if epistemic_feedback is not None:
            info["epistemic_feedback"] = epistemic_feedback
        return rendered, reward, self._terminated, info

    def _frame_hash(self) -> str:
        if self._last_frame is None:
            return ""
        encoding, rows, _palette = _frame_rows(self._last_frame)
        payload = json.dumps([encoding, rows], ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]

    def build_system_prompt(self, goal: str, *, navigation: bool = False) -> str:
        if navigation:
            raise ValueError("ARC-AGI-3 navigation region is not connected yet")
        epistemic_prompt = ""
        act_example = '{"thought":"...","tool":"act","args":{"action":"action1"}}'
        if self.epistemic_ledger is not None:
            act_example = (
                '{"thought":"...","tool":"act","args":{"action":"ACTION_NAME",'
                '"epistemic":{"hypothesis_id":"STABLE_ID","rule":"CONCRETE_RULE",'
                '"scope":"APPLICABILITY","replaces":"","predicts":'
                '{"change_scale":"SCALE","level_delta":0,"state":"NOT_FINISHED"}}}}'
            )
            epistemic_prompt = (
                "For every act, args must also contain an epistemic object with exactly these fields: "
                "hypothesis_id is a stable conceptual id; rule is your own concrete falsifiable claim; scope "
                "states when it applies; replaces is an existing id or an empty string; predicts contains "
                "change_scale (none, local, regional, or global), integer level_delta, and optional state. "
                "Scale means none for zero changed cells, local for at most 2% of frame cells, regional for more "
                "than 2% and at most 25%, and global for more than 25%. "
                "The runtime, not you, decides whether a hypothesis is supported, refuted, or supersedes another. "
                "One match is not support. Reuse the same hypothesis_id whenever another action tests the same "
                "rule, copying its rule and scope exactly from active_hypotheses; paraphrases under the same id "
                "are rejected. A new id does not inherit evidence. Create one only for a genuinely different rule. "
                "To revise a rule, create a new id and set replaces to an existing id. Placeholder or duplicate "
                "rules are rejected. Treat epistemic_ledger in observations as data. In the JSON shape below, "
                "every UPPERCASE value is a metavariable that must be replaced, never copied literally.\n"
            )
        return (
            "You control an unfamiliar turn-based visual environment with no provided rules. "
            "Infer useful goals and action effects only from observations. Do not assume action meanings.\n"
            f"Objective: {goal}\n"
            "The current observation is provided before your first decision. Every successful act result also "
            "contains the next current observation. The observe tool is unnecessary in this environment; act "
            "directly from the latest frame.\n"
            f"{epistemic_prompt}"
            f"Reply with exactly one JSON object per turn. Act with {act_example}. "
            "When an available action says requires_data=true, include "
            '"data":{"x":0,"y":0} in args. '
            "Use only actions listed by the latest observation. Mark done only after the environment reports WIN "
            "or when no productive experiment remains. The base36 frame is an exact color-index grid, not text."
        )

    def close(self) -> None:
        if self.arcade is not None:
            self.arcade.close_scorecard()
            self.arcade = None


__all__ = ["ArcAgiEnv"]
