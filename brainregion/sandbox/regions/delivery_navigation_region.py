"""城区配送的 grounded 运动执行脑区。

脑区只读取 ``UrbanDeliveryEnv.observation()`` 文本，不接收环境对象、隐藏车辆位置或 oracle。
它负责执行当前交互点之间的移动；pickup、deliver 和 done 始终由主脑决定。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from ..envs._actions import ABS_DELTA

_PASSABLE = {"=", "S", "@", *"12345678"}
_MAP_MARKS = _PASSABLE | {"#", "V"}


@dataclass(frozen=True)
class DeliveryObservation:
    status: str
    rows: tuple[str, ...]
    agent: tuple[int, int]
    task: str
    target_unit: str | None


class DeliveryNavigationRegion:
    """从公开配送 observation 规划并执行最短可知路线。"""

    uses_model = False
    name = "navigation"
    access_mode = "grounded"

    def __init__(self) -> None:
        self.roads: set[tuple[int, int]] = set()
        self.blocked: set[tuple[int, int]] = set()
        self.vehicles: set[tuple[int, int]] = set()
        self.units: dict[str, tuple[int, int]] = {}
        self.shop: tuple[int, int] | None = None
        self.current: tuple[int, int] | None = None
        self.current_task = "uninitialized"
        self.last_decision = "uninitialized"
        self.confidence = 0.0
        self.replans = 0
        self.observed_transitions = 0

    def next_action(self, observation: str) -> str | None:
        state = self._ingest(observation)
        target = self._target_for(state)
        if target is None:
            self.last_decision = "await_main_interaction"
            self.confidence = 1.0
            return None
        if state.agent == target:
            self.last_decision = "destination_reached"
            self.confidence = 1.0
            return None
        route = self._known_route(state.agent, target)
        if not route:
            self.last_decision = "no_known_route"
            self.confidence = 0.0
            return None
        self.last_decision = "route_to_unit" if state.task == "deliver" else "route_to_shop"
        self.confidence = 0.9
        return route[0]

    def observe_transition(self, *, action: str, observation: str, status: str) -> None:
        del action
        self.observed_transitions += 1
        self._ingest(observation)
        if status == "blocked":
            self.last_decision = "blocked_replan"
            self.confidence = 0.7

    def option_boundary(self, observation: str, *, actions_executed: int) -> str | None:
        if actions_executed <= 0:
            return None
        state = self._ingest(observation)
        target = self._target_for(state)
        if target is not None and state.agent == target:
            self.last_decision = "destination_reached"
            self.confidence = 1.0
            return "destination_reached"
        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "uses_model": False,
            "policy": "public_map_bfs",
            "access_mode": self.access_mode,
            "current_task": self.current_task,
            "known_roads": len(self.roads),
            "known_blocked": len(self.blocked),
            "known_vehicles": len(self.vehicles),
            "known_units": sorted(self.units),
            "shop_known": self.shop is not None,
            "replans": self.replans,
            "observed_transitions": self.observed_transitions,
            "last_decision": self.last_decision,
            "confidence": self.confidence,
        }

    def _ingest(self, observation: str) -> DeliveryObservation:
        state = parse_delivery_observation(observation)
        self.current = state.agent
        self.current_task = state.task
        vehicles_before = len(self.vehicles)
        for y, row in enumerate(state.rows):
            for x, mark in enumerate(row):
                cell = (x, y)
                if mark in _PASSABLE:
                    self.roads.add(cell)
                    self.blocked.discard(cell)
                elif mark in {"#", "V"}:
                    self.blocked.add(cell)
                    if mark == "V":
                        self.vehicles.add(cell)
                if mark == "S":
                    self.shop = cell
                elif mark in "12345678":
                    self.units[f"U{mark}"] = cell
        if state.task == "deliver" and self.shop is None:
            # 首次激活发生在主脑 pickup 后，此时 @ 覆盖 S；当前位置可安全确定为商铺。
            self.shop = state.agent
        self.replans += len(self.vehicles) - vehicles_before
        return state

    def _target_for(self, state: DeliveryObservation) -> tuple[int, int] | None:
        if state.task == "deliver" and state.target_unit is not None:
            return self.units.get(state.target_unit)
        if state.task == "return":
            return self.shop
        return None

    def _known_route(self, start: tuple[int, int], goal: tuple[int, int]) -> list[str]:
        queue: deque[tuple[int, int]] = deque([start])
        parent: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
        seen = {start}
        while queue:
            cell = queue.popleft()
            for direction in ("right", "down", "left", "up"):
                dx, dy = ABS_DELTA[direction]
                nxt = (cell[0] + dx, cell[1] + dy)
                if nxt in seen or nxt not in self.roads or nxt in self.blocked:
                    continue
                seen.add(nxt)
                parent[nxt] = (cell, direction)
                if nxt == goal:
                    actions: list[str] = []
                    cursor = goal
                    while cursor != start:
                        previous, action = parent[cursor]
                        actions.append(action)
                        cursor = previous
                    actions.reverse()
                    return actions
                queue.append(nxt)
        return []


def parse_delivery_observation(observation: str) -> DeliveryObservation:
    if not isinstance(observation, str):
        raise TypeError("observation must be text")
    lines = observation.splitlines()
    if len(lines) < 2 or not lines[0].startswith("time="):
        raise ValueError("delivery observation must contain one status line and a map")
    status = lines[0]
    rows = tuple(lines[1:])
    width = len(rows[0])
    if width < 1 or any(len(row) != width for row in rows):
        raise ValueError("delivery map rows must have equal non-zero width")
    invalid = sorted({mark for row in rows for mark in row if mark not in _MAP_MARKS})
    if invalid:
        raise ValueError(f"delivery map contains invalid marks: {invalid}")
    agents = [(x, y) for y, row in enumerate(rows) for x, mark in enumerate(row) if mark == "@"]
    if len(agents) != 1:
        raise ValueError(f"delivery observation must contain exactly one @, got {len(agents)}")

    target_unit = None
    task = "idle"
    for token in status.split():
        if token.startswith("target=U"):
            target_unit = token.split("=", 1)[1]
        if token.startswith("carrying="):
            task = "deliver"
        elif token.startswith("return_to_shop_after="):
            task = "return"
        elif token.startswith("ready_for_pickup="):
            task = "pickup"
        elif token == "all_orders_delivered":
            task = "done"
    return DeliveryObservation(
        status=status,
        rows=rows,
        agent=agents[0],
        task=task,
        target_unit=target_unit,
    )


__all__ = ["DeliveryNavigationRegion", "DeliveryObservation", "parse_delivery_observation"]
