"""Phase 4.6 拓扑记忆脑区(代码维护,无 LLM)—— 把 env 原始图「解读」成可推理的拓扑动作状态。

动机(Phase 4.5 诊断 + 文献):oracle 给 deepseek 完美像素图它也用不上(trace:能找 goal 但不会
走廊寻路到可见 goal)。文献([Do LLMs Build Spatial World Models? 2026] LLM 不建稳健空间世界模型;
[FTM ECCV2024] 拓扑图+frontier;[Coherent Spatial Memory 2025] LLM 自建图不自洽→代码维护)。
瓶颈在**表征**:agent 不会把像素图解读成「未探索出口 / 死胡同 / 回溯方向」这种可执行拓扑状态。

本 region = **解读器**:leech env 同源数据(walls/_explored/agent,与 oracle 同)+ 维护 trail →
解读成 Trémaux 可用的动作状态(frontier_directions / is_dead_end / backtrack_direction)。

⚠️ 诚实 scope(review 自折叠 #1/#5/#6):这是**表征/程序 scaffolding 实验**,非「脑区架构价值」实验。
region leech env 状态(非自主感知)+ 确定性代码(非 LLM 推理)→ 测「给解读后的可执行状态 + Trémaux
程序,deepseek 能否走迷宫/回溯」= 测 deepseek 迷宫失败是「表征/算法缺陷(scaffolding 可补)」还是
「硬上限」。结论措辞须收窄到「表征杠杆」,非「多脑区架构帮」。
"""
from __future__ import annotations

from typing import Any

_DELTA: dict[str, tuple[int, int]] = {
    "up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0),
}


class TopologicalRegion:
    """代码拓扑记忆:维护 agent 实际 trail(踩过的位置);``state(env)`` 解读 env 同源数据成
    Trémaux 动作状态。无 LLM、确定性(leech 准确 env → 隔离「解读」表征假设,不被感知误差污染)。

    - ``update(position)``:run_agent 每步 append actual env._agent(去重)。
    - ``state(env)``:读 env.walls / env._explored,用 trail 算 frontier_directions(当前 cell 的
      seen-floor 未踩邻)/ is_dead_end(无 frontier 且 ≤1 出口)/ backtrack_direction(→ trail[-2])。
    """

    def __init__(self, start: tuple[int, int] = (0, 0)) -> None:
        self.trail: list[tuple[int, int]] = [tuple(start)]

    def update(self, position) -> None:
        """run_agent 每步调:append actual position(去重 —— 原地/撞墙不重复 append)。"""
        pos = tuple(position)
        if not self.trail or self.trail[-1] != pos:
            self.trail.append(pos)

    def state(self, env: Any) -> dict:
        """解读 env 同源数据 → Trémaux 动作状态(无 LLM)。

        - frontier_directions:当前 cell 的邻中 = seen-floor(env._explored 且非墙)且 ∉ trail(未踩过)
          → 可探索出口(Trémaux:优先去这)。
        - is_dead_end:无 frontier 且非墙出口 ≤1(只剩来路)→ 原路退回。
        - backtrack_direction:当前 → trail[-2] 的方向(原路退回方向);trail<2 时 None。
        - has_frontier_in_trail:trail 上是否还有未探索出口的 cell(回溯目标存在与否)。
        """
        cur = tuple(env._agent)
        walls = env.walls
        explored = getattr(env, "_explored", set())
        trail_set = set(self.trail)

        frontier_dirs: list[str] = []
        exit_count = 0  # 非墙邻数(出口)
        for d, (dx, dy) in _DELTA.items():
            n = (cur[0] + dx, cur[1] + dy)
            if n in walls:
                continue
            # 非墙:是出口(已知 floor 或未见到的潜在通路)
            in_grid = (0 <= n[0] < env.size and 0 <= n[1] < env.size)
            if not in_grid:
                continue
            exit_count += 1
            if n in explored and n not in trail_set:
                frontier_dirs.append(d)  # seen-floor 未踩 = 可探索

        # 回溯方向:当前 → trail[-2]
        backtrack = None
        if len(self.trail) >= 2:
            prev = self.trail[-2]
            ddx, ddy = prev[0] - cur[0], prev[1] - cur[1]
            for d, (dx, dy) in _DELTA.items():
                if (dx, dy) == (ddx, ddy):
                    backtrack = d
                    break

        is_dead_end = (len(frontier_dirs) == 0) and exit_count <= 1
        should_backtrack = len(frontier_dirs) == 0  # 无未探索出口 → 回溯(死胡同或岔路全探过)

        return {
            "current": list(cur),
            "frontier_directions": frontier_dirs,
            "is_dead_end": is_dead_end,
            "should_backtrack": should_backtrack,
            "backtrack_direction": backtrack,
            "trail_length": len(self.trail),
            "suggestion": _suggest(frontier_dirs, backtrack, should_backtrack),
        }


def _suggest(frontier_dirs: list[str], backtrack: str | None, should_backtrack: bool) -> str:
    """中文可执行建议(给主脑依 Trémaux 程序用);非动作指令,是状态解读。"""
    if frontier_dirs:
        return f"有未探索出口 {frontier_dirs};依 Trémaux 选一个 act 去探"
    if should_backtrack and backtrack:
        return f"无可探索出口;原路回溯,act 向 {backtrack}"
    return "无可探索出口也无回溯方向(起点或异常)"


__all__ = ["TopologicalRegion"]
