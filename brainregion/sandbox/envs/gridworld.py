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
        wall_density: float = 0.0,
        random_walls_seed: int | None = None,
        maze_seed: int | None = None,
        maze_braid: float = 0.0,
    ) -> None:
        if not (2 <= size <= 50):
            raise ValueError(f"size 须在 2..50,got {size}")
        # fog 半径校验(review gpt):None=全可见;非负 int=fog;负数/其他 → 拒。
        if visibility_radius is not None:
            if not isinstance(visibility_radius, int) or isinstance(visibility_radius, bool):
                raise ValueError(f"visibility_radius 须 int 或 None,got {type(visibility_radius).__name__}")
            if visibility_radius < 0:
                raise ValueError(f"visibility_radius 须 ≥0,got {visibility_radius}")
        if not (0.0 <= wall_density <= 0.6):
            raise ValueError(f"wall_density 须在 0..0.6,got {wall_density}")
        if not (0.0 <= maze_braid <= 1.0):
            raise ValueError(f"maze_braid 须在 0..1,got {maze_braid}")

        self.size = size
        self.start = tuple(start)
        self.visibility_radius = visibility_radius
        self.strict_obs = strict_obs  # Phase C:True → observation() 只给当前视野(agent 须 recall_map 拿累积图)
        fog = visibility_radius is not None

        if not self._in_grid(self.start):
            raise ValueError(f"start {self.start} 越界(size={size})")

        # 地形:maze_seed → Prim's 迷宫(先生成定 floor,再从 floor 选 goal);外围一圈墙(地牢 enclosure,
        # 用户:方便辨认边缘)。maze 在 inner [1..size-2] 雕刻,start=(1,1);perimeter 恒墙。覆盖 random_walls。
        if maze_seed is not None:
            if size < 5:
                raise ValueError(f"maze 模式 size 须 ≥5(inner 雕刻区 + 墙边界),got {size}")
            self.start = (1, 1)   # maze:内角(外围墙);忽略传入 start
            self.walls, floors = self._gen_maze(maze_seed, maze_braid)
            if random_goal_seed is not None:
                self.goal = self._pick_goal_from_floors(random_goal_seed, floors, fog)
            elif goal is not None:
                self.goal = tuple(goal)
                if self.goal not in floors:
                    raise ValueError(f"maze 模式显式 goal {self.goal} 不在迷宫 floor(不可达)")
            else:
                self.goal = self._pick_goal_from_floors(maze_seed, floors, fog)
        else:
            # goal 先选(不避墙 —— 墙稍后生成时会避开 goal)。random_goal_seed → seeded 随机;否则显式/远角。
            if random_goal_seed is not None:
                self.goal = self._pick_random_goal(random_goal_seed, fog)
            else:
                self.goal = tuple(goal) if goal is not None else (size - 1, size - 1)
            # 墙:random_walls_seed → 生成(避开 start/goal + BFS 可达性保证);否则显式 walls。
            if random_walls_seed is not None:
                self.walls = self._gen_walls_reachable(random_walls_seed, wall_density)
            else:
                self.walls = frozenset(tuple(w) for w in walls)

        if not self._in_grid(self.goal):
            raise ValueError(f"goal {self.goal} 越界(size={size})")
        if self.goal == self.start:
            raise ValueError(f"goal {self.goal} 不能等于 start {self.start}")
        if self.start in self.walls:
            raise ValueError(f"start {self.start} 落在墙上")
        if self.goal in self.walls:
            raise ValueError(f"goal {self.goal} 落在墙上")
        if self.walls and not self._reachable(self.start, self.goal):
            raise ValueError(f"goal {self.goal} 不可达(墙挡死:start→goal 无通路)")

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
        """seeded 随机 goal:非 start;fog 下优先 ∉ start 可见域(藏起来逼探索),
        无合法位(start 可见域覆盖全网格)→ fallback 任意非 start(review consensus)。
        注:不滤墙 —— 墙在 goal 之后生成,会避开 goal(goal 永不在墙上)。"""
        rng = random.Random(seed)
        all_cells = [(x, y) for y in range(self.size) for x in range(self.size)]
        candidates = [c for c in all_cells if c != self.start]
        if not candidates:
            raise ValueError("无合法 goal 位置(仅 start)")
        if fog:
            start_vis = set(self._visible_cells(self.start))
            hidden = [c for c in candidates if c not in start_vis]
            pool = hidden if hidden else candidates  # fallback:小网格/大半径
        else:
            pool = candidates
        return rng.choice(pool)

    def _reachable(self, start: tuple[int, int], goal: tuple[int, int], walls: frozenset | None = None) -> bool:
        """BFS:start→goal 是否存在不穿墙的 4-邻接通路(可达性保证,review 早标的关键)。"""
        w = self.walls if walls is None else walls
        if start == goal:
            return True
        from collections import deque
        seen = {start}
        q: deque[tuple[int, int]] = deque([start])
        while q:
            cx, cy = q.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nxt = (cx + dx, cy + dy)
                if nxt in seen or not self._in_grid(nxt) or nxt in w:
                    continue
                if nxt == goal:
                    return True
                seen.add(nxt)
                q.append(nxt)
        return False

    def _gen_walls_reachable(self, seed: int, density: float) -> frozenset[tuple[int, int]]:
        """seeded 随机墙(避开 start/goal,密度=density × 可放格)+ BFS 可达性保证:
        多次重采(seed+attempt)直到 start→goal 可达;密度过高全失败 → 降密度重试;仍不行 → 空墙兜底。"""
        all_cells = [(x, y) for y in range(self.size) for x in range(self.size)]
        placeable = [c for c in all_cells if c != self.start and c != self.goal]
        d = max(0.0, min(0.6, density))
        # 原密度多 seed 重采
        for attempt in range(30):
            rng = random.Random(seed + attempt)
            n = int(len(placeable) * d)
            walls = frozenset(rng.sample(placeable, n)) if n > 0 else frozenset()
            if self._reachable(self.start, self.goal, walls):
                return walls
        # 降密度兜底
        for fallback_d in (d * 0.5, d * 0.25, 0.0):
            rng = random.Random(seed)
            n = int(len(placeable) * fallback_d)
            walls = frozenset(rng.sample(placeable, n)) if n > 0 else frozenset()
            if self._reachable(self.start, self.goal, walls):
                return walls
        return frozenset()  # 最终兜底:无墙(保证可构造)

    def _gen_maze(self, seed: int, braid: float) -> tuple[frozenset[tuple[int, int]], set[tuple[int, int]]]:
        """Phase 4.5 Prim's 迷宫(odd/odd junction 在 inner [1..size-2];走廊宽1 墙宽1;外围一圈墙)。

        全格起墙;从 start=(1,1) junction 起,frontier = 相邻未访 junction(距离2,均在 inner)的中间墙;
        随机取 frontier 墙 carve + 其后 junction。Prim's(随机 frontier)产 **bushy** 迷宫(多岔路多死胡同,
        记忆决策点多)。完美迷宫 = spanning tree(连通无环,可达)。braid>0 去死胡同加环。返 ``(walls, floors)``。
        carving 仅 inner [1..size-2] → perimeter (row/col 0 与 size-1) 恒墙(地牢 enclosure,方便辨认边缘)。
        """
        rng = random.Random(seed)
        S = self.size
        floors: set[tuple[int, int]] = {self.start}
        # junction-to-junction delta (距离 2) + 中间墙 delta (距离 1)
        steps = [((0, -2), (0, -1)), ((0, 2), (0, 1)), ((-2, 0), (-1, 0)), ((2, 0), (1, 0))]
        frontier: list[tuple[tuple[int, int], tuple[int, int]]] = []  # (wall_cell, beyond_junction)

        def _add_frontier(j: tuple[int, int]) -> None:
            for (dx, dy), (bx, by) in steps:
                nx, ny = j[0] + dx, j[1] + dy
                beyond = (nx, ny)
                # 仅 inner [1..S-2] carve → perimeter 不动(恒墙)
                if 1 <= nx <= S - 2 and 1 <= ny <= S - 2 and beyond not in floors:
                    frontier.append(((j[0] + bx, j[1] + by), beyond))

        _add_frontier(self.start)
        while frontier:
            idx = rng.randint(0, len(frontier) - 1)
            wall_cell, beyond = frontier.pop(idx)
            if beyond in floors:
                continue  # 已被别路 carve(lazy 去重)
            floors.add(wall_cell)
            floors.add(beyond)
            _add_frontier(beyond)
        walls: set[tuple[int, int]] = {(x, y) for y in range(S) for x in range(S)} - floors
        if braid > 0.0:
            self._braid_maze(walls, floors, braid, rng)
        return frozenset(walls), floors

    def _braid_maze(self, walls: set, floors: set, braid: float, rng: random.Random) -> None:
        """去死胡同加环(地牢感 + 略易):floor 邻居数=1 的 cell = 死胡同;取 braid 比例,各打通一个
        墙邻居(其对侧是 floor → 造环,连通性不减)。原地改 walls/floors。"""
        deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        def _floor_neighbors(c: tuple[int, int]) -> int:
            return sum(1 for dx, dy in deltas if (c[0] + dx, c[1] + dy) in floors)

        dead_ends = [c for c in floors if _floor_neighbors(c) == 1]
        rng.shuffle(dead_ends)
        n = int(len(dead_ends) * max(0.0, min(1.0, braid)))
        for c in dead_ends[:n]:
            opts = []
            for dx, dy in deltas:
                wall_cell = (c[0] + dx, c[1] + dy)        # 候选墙
                beyond = (c[0] + 2 * dx, c[1] + 2 * dy)   # 墙对侧
                if wall_cell in walls and beyond in floors and beyond != c:
                    opts.append(wall_cell)
            if opts:
                carve = rng.choice(opts)
                walls.discard(carve)
                floors.add(carve)

    def _pick_goal_from_floors(self, seed: int, floors: set[tuple[int, int]], fog: bool) -> tuple[int, int]:
        """maze 模式 goal:从 floor cell 选(非 start;fog 下优先 ∉ start 可见域藏起来;偏远)。
        fallback:hidden 空 → 任意 floor;floor 不足 → 任意非 start(防御,正常不触发)。"""
        rng = random.Random(seed)
        candidates = [c for c in floors if c != self.start]
        if not candidates:
            all_cells = [(x, y) for y in range(self.size) for x in range(self.size)]
            candidates = [c for c in all_cells if c != self.start]
        if fog:
            start_vis = set(self._visible_cells(self.start))
            hidden = [c for c in candidates if c not in start_vis]
            pool = hidden if hidden else candidates
        else:
            pool = candidates
        # 偏远:取 Manhattan 距 start 远的 top 半,再随机选(逼真探索,非近邻 trivial)
        pool = sorted(pool, key=lambda c: -(abs(c[0] - self.start[0]) + abs(c[1] - self.start[1])))
        far = pool[: max(1, len(pool) // 2)]
        return rng.choice(far)

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

    def relative_view(self) -> str:
        """agent-centered (2r+1)×(2r+1) 视野 patch(**无绝对坐标**;出界/视野外 `?`)。Phase D.2 region 用:
        逼 dead-reckon(不泄漏 abs 位置 → region 的 pose 是唯一位置源,忠实「惯性导航」+ 实验干净)。
        radius=None(全可见)时返 render()(此情况无 fog,abs 无意义;region 模式恒有 radius)。"""
        r = self.visibility_radius
        if r is None:
            return self.render()
        rows: list[str] = []
        for dy in range(-r, r + 1):
            chars = []
            for dx in range(-r, r + 1):
                cell = (self._agent[0] + dx, self._agent[1] + dy)
                if not self._in_grid(cell):
                    chars.append(_FOG)  # 出界 → ?
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
