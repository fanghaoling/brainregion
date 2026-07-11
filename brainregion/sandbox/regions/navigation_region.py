"""Deterministic navigation executor used to test control delegation.

This region deliberately contains no LLM. It maintains a proper DFS path stack
and chooses one primitive action at a time from locally explored cells. The
runtime remains the only component allowed to call ``env.step``; the region is a
policy, not an environment owner.

The implementation may read GridWorld's explored cells and walls, so it is
experimental scaffolding rather than evidence that a learned region can
navigate. Its purpose is to isolate the value of delegating execution from the
value of merely returning navigation advice to the main model.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..envs._actions import ABS_DELTA


class NavigationRegion:
    """Locally grounded DFS policy for absolute-direction GridWorld actions."""

    uses_model = False

    def __init__(self, start: tuple[int, int] = (0, 0)) -> None:
        start = tuple(start)
        self.path: list[tuple[int, int]] = [start]
        self.visited: set[tuple[int, int]] = {start}
        self.attempted: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)

    def next_action(self, env: Any) -> str | None:
        """Return the next DFS action, or ``None`` when no known route remains."""
        if bool(getattr(env, "ego_actions", False)):
            raise ValueError("NavigationRegion v1 only supports absolute actions")

        current = tuple(env._agent)
        self._sync_path(current)
        self.visited.add(current)
        explored = set(getattr(env, "_explored", set()))
        walls = set(getattr(env, "walls", set()))

        candidates: list[tuple[str, tuple[int, int]]] = []
        for direction in ("right", "down", "left", "up"):
            dx, dy = ABS_DELTA[direction]
            target = (current[0] + dx, current[1] + dy)
            if not (0 <= target[0] < env.size and 0 <= target[1] < env.size):
                continue
            if target not in explored or target in walls:
                continue
            if target in self.attempted[current] or target in self.visited:
                continue
            candidates.append((direction, target))

        if candidates:
            direction, target = candidates[0]
            self.attempted[current].add(target)
            return direction

        if len(self.path) < 2:
            return None
        parent = self.path[-2]
        direction = _direction_between(current, parent)
        if direction is None:
            raise RuntimeError(f"navigation path is not adjacent: {current!r} -> {parent!r}")
        self.attempted[current].add(parent)
        return direction

    def snapshot(self) -> dict[str, Any]:
        return {
            "path": [list(p) for p in self.path],
            "visited_cells": len(self.visited),
            "uses_model": False,
            "policy": "local_dfs",
        }

    def observe_position(self, position: tuple[int, int]) -> None:
        """Commit the runtime-observed position after an executed action."""
        current = tuple(position)
        self._sync_path(current)
        self.visited.add(current)

    def _sync_path(self, current: tuple[int, int]) -> None:
        if not self.path:
            self.path = [current]
            return
        if current == self.path[-1]:
            return
        if len(self.path) >= 2 and current == self.path[-2]:
            self.path.pop()
            return
        if current in self.path:
            self.path = self.path[: self.path.index(current) + 1]
            return
        self.path.append(current)


def _direction_between(origin: tuple[int, int], target: tuple[int, int]) -> str | None:
    delta = (target[0] - origin[0], target[1] - origin[1])
    for direction, candidate in ABS_DELTA.items():
        if candidate == delta:
            return direction
    return None


__all__ = ["NavigationRegion"]
