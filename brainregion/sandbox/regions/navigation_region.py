"""Deterministic navigation executors used to test control delegation.

This region deliberately contains no LLM. It maintains a proper DFS path stack
and chooses one primitive action at a time from locally explored cells. The
runtime remains the only component allowed to call ``env.step``; the region is a
policy, not an environment owner.

The implementation may read GridWorld's explored cells and walls, so it is
experimental scaffolding rather than evidence that a learned region can
navigate. Its purpose is to isolate the value of delegating execution from the
value of merely returning navigation advice to the main model.

``GroundedNavigationRegion`` is the perception-limited counterpart. It never
receives an environment object; it builds a known map solely from textual
observations and transition outcomes supplied by the runtime.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..envs._actions import ABS_DELTA


class NavigationRegion:
    """Privileged oracle DFS policy retained as an explicit control arm."""

    uses_model = False
    name = "navigation"
    access_mode = "oracle"

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
            "access_mode": self.access_mode,
            "confidence": 1.0,
        }

    def observe_position(self, position: tuple[int, int]) -> None:
        """Commit the runtime-observed position after an executed action."""
        current = tuple(position)
        self._sync_path(current)
        self.visited.add(current)

    def observe_transition(self, *, action: str, observation: Any, status: str) -> None:
        """OptionRegion adapter:oracle observation is the privileged env object."""
        del action, status
        self.observe_position(tuple(observation._agent))

    def option_boundary(self, observation: Any, *, actions_executed: int) -> str | None:
        """Oracle control runs to goal/budget;it is the privileged upper bound."""
        del observation, actions_executed
        return None

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


class GroundedNavigationRegion:
    """DFS/BFS policy grounded only in the text observation stream.

    Public methods intentionally accept strings and scalar transition data,
    never ``GridWorld``. Coordinates are inferred from the visible ``@`` marker
    in the observation; cells rendered as ``?`` remain absent from ``known``.
    """

    uses_model = False
    name = "navigation"
    access_mode = "grounded"

    def __init__(self) -> None:
        self.known: dict[tuple[int, int], str] = {}
        self.path: list[tuple[int, int]] = []
        self.visited: set[tuple[int, int]] = set()
        self.attempted: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
        self.current: tuple[int, int] | None = None
        self.last_decision = "uninitialized"
        self.confidence = 0.0

    def next_action(self, observation: str) -> str | None:
        """Choose one absolute action from accumulated visible evidence."""
        current = self._ingest(observation)

        goal = next((cell for cell, mark in self.known.items() if mark == "G"), None)
        if goal is not None:
            route = self._known_route(current, goal)
            if route:
                self.last_decision = "known_goal_route"
                self.confidence = 0.95
                return route[0]

        candidates: list[tuple[str, tuple[int, int]]] = []
        for direction in ("right", "down", "left", "up"):
            target = _move(current, direction)
            if not self._known_passable(target):
                continue
            if target in self.attempted[current] or target in self.visited:
                continue
            candidates.append((direction, target))
        if candidates:
            direction, target = candidates[0]
            self.attempted[current].add(target)
            self.last_decision = "visible_frontier"
            self.confidence = 0.75
            return direction

        if len(self.path) >= 2:
            parent = self.path[-2]
            direction = _direction_between(current, parent)
            if direction is None:
                raise RuntimeError(f"grounded path is not adjacent: {current!r} -> {parent!r}")
            self.attempted[current].add(parent)
            self.last_decision = "backtrack"
            self.confidence = 0.6
            return direction

        self.last_decision = "no_known_route"
        self.confidence = 0.0
        return None

    def observe_transition(self, *, action: str, observation: str, status: str) -> None:
        """Consume only runtime-visible transition facts after an action."""
        del action  # attempted edge was recorded when the action was selected
        current = self._ingest(observation)
        if status == "blocked":
            self.confidence = min(self.confidence, 0.5)
        self.current = current

    def option_boundary(self, observation: str, *, actions_executed: int) -> str | None:
        """Yield at a meaningful decision boundary after making progress."""
        current = self._ingest(observation)
        if actions_executed <= 0:
            return None
        if any(mark == "G" for mark in self.known.values()):
            return None  # keep following a known route;goal termination is checked by runtime
        frontiers = 0
        for direction in ABS_DELTA:
            target = _move(current, direction)
            if (
                self._known_passable(target)
                and target not in self.visited
                and target not in self.attempted[current]
            ):
                frontiers += 1
        if frontiers >= 2:
            self.last_decision = "junction_boundary"
            self.confidence = 0.5
            return "junction"
        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "path": [list(p) for p in self.path],
            "visited_cells": len(self.visited),
            "known_cells": len(self.known),
            "uses_model": False,
            "policy": "observation_dfs_bfs",
            "access_mode": self.access_mode,
            "last_decision": self.last_decision,
            "confidence": self.confidence,
        }

    def _ingest(self, observation: str) -> tuple[int, int]:
        rows = _parse_observation(observation)
        agents: list[tuple[int, int]] = []
        for y, row in enumerate(rows):
            for x, mark in enumerate(row):
                if mark == "?":
                    continue
                cell = (x, y)
                self.known[cell] = mark
                if mark == "@":
                    agents.append(cell)
        if len(agents) != 1:
            raise ValueError(f"observation must contain exactly one @, got {len(agents)}")
        current = agents[0]
        self.known[current] = "."
        self._sync_path(current)
        self.visited.add(current)
        self.current = current
        return current

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

    def _known_passable(self, cell: tuple[int, int]) -> bool:
        return self.known.get(cell) in {".", "G", "@"}

    def _known_route(self, start: tuple[int, int], goal: tuple[int, int]) -> list[str]:
        if start == goal:
            return []
        queue = [start]
        parent: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
        seen = {start}
        for cell in queue:
            for direction in ("right", "down", "left", "up"):
                nxt = _move(cell, direction)
                if nxt in seen or not self._known_passable(nxt):
                    continue
                seen.add(nxt)
                parent[nxt] = (cell, direction)
                if nxt == goal:
                    actions: list[str] = []
                    cursor = goal
                    while cursor != start:
                        prev, action = parent[cursor]
                        actions.append(action)
                        cursor = prev
                    actions.reverse()
                    return actions
                queue.append(nxt)
        return []


def _direction_between(origin: tuple[int, int], target: tuple[int, int]) -> str | None:
    delta = (target[0] - origin[0], target[1] - origin[1])
    for direction, candidate in ABS_DELTA.items():
        if candidate == delta:
            return direction
    return None


def _move(origin: tuple[int, int], direction: str) -> tuple[int, int]:
    dx, dy = ABS_DELTA[direction]
    return origin[0] + dx, origin[1] + dy


def _parse_observation(observation: str) -> list[str]:
    if not isinstance(observation, str):
        raise TypeError("observation must be text")
    rows = observation.splitlines()
    if not rows or not rows[0]:
        raise ValueError("observation is empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("observation rows must have equal width")
    allowed = {"?", "#", ".", "@", "G"}
    invalid = sorted({mark for row in rows for mark in row if mark not in allowed})
    if invalid:
        raise ValueError(f"observation contains invalid marks: {invalid}")
    return rows


__all__ = ["NavigationRegion", "GroundedNavigationRegion"]
