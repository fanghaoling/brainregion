"""GridWorld:网格寻路 env(Phase A 全可见 + Phase B fog 部分可观,文本渲染)。

最简客观 grounding env —— agent 经 observe(看网格)/ act(移动)工具交互(工具接在 sandbox
loop 的 dispatch_tool,见 loop.py,不另起 driver)。到达 goal → reward=1.0(terminated)。
0/1 稀疏 reward(验证 agent loop,非 RL,无 shaping hacking)。

**Phase B fog(部分可观)**:`visibility_radius` 非 None → agent 只看到 Chebyshev 半径内 cell,
未探索格渲染 `?`;env-backed `_explored` 累积(classic fog-of-war,探索过的不变)。逼出 视觉局部
感知 + 探索策略 + 记忆(transcript 累积心智地图;Phase C 再搬到 memory region)。radius=None = 全可见
(Phase A 现行,回归不变)。

撞墙/越界不崩(info blocked);非法动作不崩(info invalid);terminal 后再 act 不动/不重计(already_done)。
frames 记录每步渲染供 replay/调试窗。确定性:布局由构造参数决定;random_goal_seed 给定时 goal 由
seed 定(其余无 RNG)。review 双强(2026-07-08):radius 校验 + random_goal fallback。
"""
from __future__ import annotations

import random

_WALL = "#"
_FLOOR = "."
_GOAL = "G"
_AGENT = "@"
_FOG = "?"

_ACTION_DELTA: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


class GridWorld:
    """size×size 网格;agent(@)从 start 走到 goal(G),墙(#)挡路。

    坐标 (x=列, y=行),原点左上 (0,0)。render 返回逐行文本(无外框;越界=撞隐形墙,原地)。
    Phase B fog(visibility_radius 非 None):只渲染 _explored 内 cell,余 `?`;radius=None 全可见。

    Phase A 无墙布局保证 start→goal 可达(空网格);随机墙/可达性校验 = Phase C+(本 phase 不引入)。
    """

    action_vocab = tuple(_ACTION_DELTA.keys())

    def __init__(
        self,
        *,
        size: int = 5,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] | None = None,
        walls: tuple[tuple[int, int], ...] = (),
        visibility_radius: int | None = None,
        random_goal_seed: int | None = None,
        strict_obs: bool = False,
    ) -> None:
        if not (2 <= size <= 50):
            raise ValueError(f"size 须在 2..50,got {size}")
        # fog 半径校验(review gpt):None=全可见;非负 int=fog;负数/其他 → 拒。
        if visibility_radius is not None:
            if not isinstance(visibility_radius, int) or isinstance(visibility_radius, bool):
                raise ValueError(f"visibility_radius 须 int 或 None,got {type(visibility_radius).__name__}")
            if visibility_radius < 0:
                raise ValueError(f"visibility_radius 须 ≥0,got {visibility_radius}")

        self.size = size
        self.start = tuple(start)
        self.walls = frozenset(tuple(w) for w in walls)
        self.visibility_radius = visibility_radius
        self.strict_obs = strict_obs  # Phase C:True → observation() 只给当前视野(agent 须 recall_map 拿累积图)
        fog = visibility_radius is not None

        if not self._in_grid(self.start):
            raise ValueError(f"start {self.start} 越界(size={size})")
        if self.start in self.walls:
            raise ValueError(f"start {self.start} 落在墙上")

        # goal:random_goal_seed 给定 → seeded 随机;否则用显式 goal(默认远角)。
        if random_goal_seed is not None:
            self.goal = self._pick_random_goal(random_goal_seed, fog)
        else:
            self.goal = tuple(goal) if goal is not None else (size - 1, size - 1)
        if not self._in_grid(self.goal):
            raise ValueError(f"goal {self.goal} 越界(size={size})")
        if self.goal in self.walls:
            raise ValueError(f"goal {self.goal} 落在墙上")
        if self.goal == self.start:
            raise ValueError(f"goal {self.goal} 不能等于 start {self.start}")

        self._agent = self.start
        self._terminated = False
        self.total_reward = 0.0
        # fog:初始已知 = start 可见域;全可见时 explored 不用(空集,render 走全可见分支)。
        self._explored: set[tuple[int, int]] = set(self._visible_cells(self.start)) if fog else set()
        self.frames: list[str] = [self.render()]

    def _in_grid(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.size and 0 <= y < self.size

    def _visible_cells(self, pos: tuple[int, int]) -> list[tuple[int, int]]:
        """Chebyshev 半径 visibility_radius 内 + in-grid 的 cell(fog 用;radius=None → [])。"""
        r = self.visibility_radius
        if r is None:
            return []
        px, py = pos
        out = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                c = (px + dx, py + dy)
                if self._in_grid(c):
                    out.append(c)
        return out

    def _pick_random_goal(self, seed: int, fog: bool) -> tuple[int, int]:
        """seeded 随机 goal:非 start/非墙;fog 下优先 ∉ start 可见域(藏起来逼探索),
        无合法位(start 可见域覆盖全网格)→ fallback 任意非 start/非墙(review consensus)。"""
        rng = random.Random(seed)
        all_cells = [(x, y) for y in range(self.size) for x in range(self.size)]
        candidates = [c for c in all_cells if c != self.start and c not in self.walls]
        if not candidates:
            raise ValueError("无合法 goal 位置(网格全墙或仅 start)")
        if fog:
            start_vis = set(self._visible_cells(self.start))
            hidden = [c for c in candidates if c not in start_vis]
            pool = hidden if hidden else candidates  # fallback:小网格/大半径
        else:
            pool = candidates
        return rng.choice(pool)

    def reset(self, *, seed: int | None = None) -> str:
        """重置 agent 到 start,清空 frames + explored(fog 重探)。

        seed 保留(no-op):布局由构造参数决定(random_goal_seed 已固化 goal),无运行时 RNG。
        """
        self._agent = self.start
        self._terminated = False
        self.total_reward = 0.0
        if self.visibility_radius is not None:
            self._explored = set(self._visible_cells(self.start))
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

        fog 下成功移动后扩 _explored(新可见格);无状态改变的动作不扩/不追加 frame。
        """
        if self._terminated:
            return self.observation(), 0.0, True, {"already_done": True}
        a = (action or "").strip().lower()
        if a not in _ACTION_DELTA:
            return self.observation(), 0.0, False, {"invalid": action}
        dx, dy = _ACTION_DELTA[a]
        nx, ny = self._agent[0] + dx, self._agent[1] + dy
        if not self._in_grid((nx, ny)) or (nx, ny) in self.walls:
            return self.observation(), 0.0, False, {"blocked": True}
        self._agent = (nx, ny)
        if self.visibility_radius is not None:  # fog:扩探索域(在 render 前,新格本帧可见)
            self._explored |= set(self._visible_cells(self._agent))
        if self._agent == self.goal:
            self._terminated = True
            self.total_reward += 1.0
            self.frames.append(self.render())  # replay:累积图
            return self.observation(), 1.0, True, {"goal": True}  # agent:当前视野(strict)/累积
        self.frames.append(self.render())
        return self.observation(), 0.0, False, {}

    def render_visible(self) -> str:
        """当前 Chebyshev 视野 only(不含历史探索);无 fog(radius None)时 == render()。Phase C 严格观察用。"""
        r = self.visibility_radius
        if r is None:
            return self.render()  # 无 fog → 全可见,与 render 等价
        rows: list[str] = []
        for y in range(self.size):
            chars = []
            for x in range(self.size):
                cell = (x, y)
                if abs(x - self._agent[0]) > r or abs(y - self._agent[1]) > r:
                    chars.append(_FOG)  # 视野外 → `?`(不论是否探索过)
                    continue
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

    def observation(self) -> str:
        """agent 看到的观测:strict_obs → 当前视野(render_visible);否则累积图(render)。
        Phase C 统一接口 —— dispatch observe/act 走此(防 strict 模式 observe 泄漏累积图,gpt high)。"""
        return self.render_visible() if self.strict_obs else self.render()

    def render(self) -> str:
        rows: list[str] = []
        fog = self.visibility_radius is not None
        for y in range(self.size):
            chars = []
            for x in range(self.size):
                cell = (x, y)
                if fog and cell not in self._explored:
                    chars.append(_FOG)
                    continue
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
