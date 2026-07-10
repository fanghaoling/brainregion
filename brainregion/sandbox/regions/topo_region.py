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

from ..envs._actions import ABS_DELTA, relative_direction


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

        **ego 模式**(``env.ego_actions`` True,GPT#1 判模式):frontier/backtrack 转相对 heading
        (forward/left/right/back),suggestion 吐可执行 ego 配方(turn_left 后 forward);abs 模式
        仍吐 abs 词(零回归)。
        """
        cur = tuple(env._agent)
        walls = env.walls
        explored = getattr(env, "_explored", set())
        trail_set = set(self.trail)
        ego = bool(getattr(env, "ego_actions", False))
        heading = getattr(env, "_heading", None)

        frontier_abs: list[str] = []
        exit_count = 0  # 非墙邻数(出口)
        for d, (dx, dy) in ABS_DELTA.items():
            n = (cur[0] + dx, cur[1] + dy)
            if n in walls:
                continue
            # 非墙:是出口(已知 floor 或未见到的潜在通路)
            in_grid = (0 <= n[0] < env.size and 0 <= n[1] < env.size)
            if not in_grid:
                continue
            exit_count += 1
            if n in explored and n not in trail_set:
                frontier_abs.append(d)  # seen-floor 未踩 = 可探索

        # 回溯方向:当前 → trail[-2]
        backtrack_abs = None
        if len(self.trail) >= 2:
            prev = self.trail[-2]
            ddx, ddy = prev[0] - cur[0], prev[1] - cur[1]
            for d, (dx, dy) in ABS_DELTA.items():
                if (dx, dy) == (ddx, ddy):
                    backtrack_abs = d
                    break

        is_dead_end = (len(frontier_abs) == 0) and exit_count <= 1
        should_backtrack = len(frontier_abs) == 0  # 无未探索出口 → 回溯(死胡同或岔路全探过)

        if ego and heading:  # ego:转相对 heading + 可执行配方(无 abs 词泄漏,review opus-3/gpt-3)
            order = {"forward": 0, "left": 1, "right": 2, "back": 3}
            frontier_rel = sorted((relative_direction(d, heading) for d in frontier_abs), key=lambda x: order.get(x, 9))
            backtrack_rel = relative_direction(backtrack_abs, heading) if backtrack_abs else None
            return {
                "current": list(cur),
                "heading": heading,
                "frontier_directions": frontier_rel,
                "is_dead_end": is_dead_end,
                "should_backtrack": should_backtrack,
                "backtrack_direction": backtrack_rel,
                "trail_length": len(self.trail),
                "suggestion": _suggest_ego(frontier_rel, backtrack_rel, should_backtrack),
            }

        return {
            "current": list(cur),
            "frontier_directions": frontier_abs,
            "is_dead_end": is_dead_end,
            "should_backtrack": should_backtrack,
            "backtrack_direction": backtrack_abs,
            "trail_length": len(self.trail),
            "suggestion": _suggest(frontier_abs, backtrack_abs, should_backtrack),
        }


def _suggest(frontier_dirs: list[str], backtrack: str | None, should_backtrack: bool) -> str:
    """中文可执行建议(abs 模式,给主脑依 Trémaux 程序用);非动作指令,是状态解读。"""
    if frontier_dirs:
        return f"有未探索出口 {frontier_dirs};依 Trémaux 选一个 act 去探"
    if should_backtrack and backtrack:
        return f"无可探索出口;原路回溯,act 向 {backtrack}"
    return "无可探索出口也无回溯方向(起点或异常)"


def _suggest_ego(frontier_rel: list[str], backtrack_rel: str | None, should_backtrack: bool) -> str:
    """ego 可执行配方(相对 heading;无 abs 词,无 turn_180 → back 用 turn_left×2)。"""
    if frontier_rel:
        if "forward" in frontier_rel:
            return "有未探索出口在前方;act forward 去探"
        return f"有未探索出口在你{frontier_rel}侧;转向后 forward(如 turn_left 后 forward)"
    if should_backtrack and backtrack_rel:
        if backtrack_rel == "forward":
            return "回溯在前方;act forward 原路退回"
        if backtrack_rel == "back":
            return "回溯在身后;turn_left turn_left(掉头)后 forward 原路退回"
        return f"回溯在你{backtrack_rel}侧;turn_{backtrack_rel} 后 forward 原路退回"
    return "无可探索出口也无回溯方向(起点或异常)"


__all__ = ["TopologicalRegion"]
