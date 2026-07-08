"""GridWorld:全可见网格寻路 env(Phase A,文本渲染)。

最简客观 grounding env —— agent 经 observe(看网格)/ act(移动)工具交互(工具接在 sandbox
loop 的 dispatch_tool,见 loop.py,不另起 driver)。到达 goal → reward=1.0(terminated)。
0/1 稀疏 reward(验证 agent loop,非 RL,无 shaping hacking)。

撞墙/越界不崩(agent 原地,info 标记 blocked);非法动作不崩(info invalid);terminal 后再 act
不动/不重计(already_done)。frames 记录每步渲染供 replay/调试窗。

确定性:布局由构造参数(walls/start/goal)决定,无 RNG → reset(seed) 对任意 seed 行为一致
(seed 保留供未来随机墙生成;显式布局下 no-op)。review 双强(2026-07-08):显式定义 None 行为。
"""
from __future__ import annotations

_WALL = "#"
_FLOOR = "."
_GOAL = "G"
_AGENT = "@"

_ACTION_DELTA: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


class GridWorld:
    """全可见 size×size 网格;agent(@)从 start 走到 goal(G),墙(#)挡路。

    坐标 (x=列, y=行),原点左上 (0,0)。render 返回逐行文本(无外框;越界=撞隐形墙,原地)。
    agent 与 goal 不同格(构造校验);agent 抵达 goal 后 terminated,render 仍标 @。
    """

    action_vocab = tuple(_ACTION_DELTA.keys())

    def __init__(
        self,
        *,
        size: int = 5,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] | None = None,
        walls: tuple[tuple[int, int], ...] = (),
    ) -> None:
        if not (2 <= size <= 50):
            raise ValueError(f"size 须在 2..50,got {size}")
        goal = tuple(goal) if goal is not None else (size - 1, size - 1)
        self.size = size
        self.start = tuple(start)
        self.goal = goal
        self.walls = frozenset(tuple(w) for w in walls)
        for name, cell in (("start", self.start), ("goal", self.goal)):
            if not self._in_grid(cell):
                raise ValueError(f"{name} {cell} 越界(size={size})")
        if self.start in self.walls:
            raise ValueError(f"start {self.start} 落在墙上")
        if self.goal in self.walls:
            raise ValueError(f"goal {self.goal} 落在墙上")
        if self.goal == self.start:
            raise ValueError(f"goal {goal} 不能等于 start {self.start}")

        self._agent = self.start
        self._terminated = False
        self.total_reward = 0.0
        self.frames: list[str] = [self.render()]

    def _in_grid(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.size and 0 <= y < self.size

    def reset(self, *, seed: int | None = None) -> str:
        """重置 agent 到 start,清空 frames。

        seed 保留(no-op):布局由构造参数决定,无 RNG,故 reset 对任意/None seed 行为确定一致。
        """
        self._agent = self.start
        self._terminated = False
        self.total_reward = 0.0
        self.frames = [self.render()]
        return self.frames[0]

    @property
    def solved(self) -> bool:
        return self._agent == self.goal

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """执行动作 → (obs, reward, terminated, info)。

        - terminal 后 → 不动 / reward 0 / terminated True / info already_done。
        - 非法动作(不在 vocab)→ 原地 / reward 0 / not terminated / info invalid。
        - 撞墙/越界 → 原地 / reward 0 / not terminated / info blocked。
        - 到达 goal → reward 1.0 / terminated / info goal,记 frame。
        - 其余合法移动 → reward 0 / not terminated,记 frame。

        无状态改变的动作(terminal/invalid/blocked)不追加 frame,replay 只记 distinct 态。
        """
        if self._terminated:
            return self.render(), 0.0, True, {"already_done": True}
        a = (action or "").strip().lower()
        if a not in _ACTION_DELTA:
            return self.render(), 0.0, False, {"invalid": action}
        dx, dy = _ACTION_DELTA[a]
        nx, ny = self._agent[0] + dx, self._agent[1] + dy
        if not self._in_grid((nx, ny)) or (nx, ny) in self.walls:
            return self.render(), 0.0, False, {"blocked": True}
        self._agent = (nx, ny)
        if self._agent == self.goal:
            self._terminated = True
            self.total_reward += 1.0
            self.frames.append(self.render())
            return self.frames[-1], 1.0, True, {"goal": True}
        self.frames.append(self.render())
        return self.frames[-1], 0.0, False, {}

    def render(self) -> str:
        rows: list[str] = []
        for y in range(self.size):
            chars = []
            for x in range(self.size):
                cell = (x, y)
                if cell == self._agent:
                    chars.append(_AGENT)
                elif cell == self.goal:
                    chars.append(_GOAL)
                elif cell in self.walls:
                    chars.append(_WALL)
                else:
                    chars.append(_FLOOR)
            rows.append("".join(chars))
        return "\n".join(rows)
