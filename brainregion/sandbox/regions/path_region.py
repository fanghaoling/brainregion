"""Phase 4.7 路径轨迹记忆脑区(代码维护,无 LLM)—— 把 agent 走过的连续路径**标在图上**给主脑看。

动机(用户洞察 + Phase 4.6 后):之前所有「图记忆」都不帮 agent ——
- oracle(env.render() 完美像素图)sonnet 都用不好(咨询 9× 反而解更少);
- MemoryRegion 只给**定性文字** rough_map + 单个 pose(无空间路径);
- TopologicalRegion 维护 trail 但只返**解读状态**(frontier/backtrack),不把路径本身给主脑。
用户指出:从没实现过「每次运动留下对应记忆场景位置的痕迹、连成连续路径」。文字 rough_map 是弱空间
表征,可能正是不帮的根因。本 region = 补这个缺口:**leech env 同源图 + 维护 trail → 渲染成「图上标了
我走过的路径(·)」的场景**,主脑 recall_path 拿这个空间接地的路径记忆。

⚠️ 诚实 scope:与 topo 同属**表征 scaffolding 实验**(非「脑区架构价值」实验)。leech env 准确状态
+ 代码(非 LLM)→ 测「路径标注表征是否比裸图/文字更可用」。结论收窄到「表征杠杆」。
关键 contrast:**path_trace(图+路径标)vs path_oracle(裸图)** —— 同数据,只差路径标注。
"""
from __future__ import annotations

from typing import Any

_WALL = "#"
_FLOOR = "."
_PATH = "·"   # 走过的格子(trail)—— 连续路径标记
_GOAL = "G"
_AGENT = "@"
_FOG = "?"


class PathTraceRegion:
    """代码路径轨迹记忆:维护 agent 实际 trail(踩过的位置序列);``state(env)`` 渲染 env 同源图为
    「图上标了走过的路径(`·`)」。无 LLM、确定性(leech 准确 env → 隔离「路径标注」表征假设)。

    - ``update(position)``:run_agent 每步 append actual env._agent(去重 —— 原地/撞墙不重复)。
    - ``state(env)``:读 env.walls/_explored/_agent/goal + trail → 渲染 path_map(trail cell 标 `·`,
      seen-未踩 标 `.`,墙 `#`,未探索 `?`,agent `@`,goal `G`)。返 ``{path_map, trail_length}``。
    """

    def __init__(self, start: tuple[int, int] = (0, 0)) -> None:
        self.trail: list[tuple[int, int]] = [tuple(start)]

    def update(self, position) -> None:
        """run_agent 每步调:append actual position(去重 —— 原地/撞墙不重复 append)。"""
        pos = tuple(position)
        if not self.trail or self.trail[-1] != pos:
            self.trail.append(pos)

    def render_with_path(self, env: Any) -> str:
        """env 同源图(explored/walls/agent/goal)+ trail 标 `·`(走过的连续路径)。fog 下未探索显 `?`。

        与 oracle(env.render())唯一差:trail cell 标 `·`(走过)vs `.`(看到没踩)→ 路径可视化。
        """
        fog = env.visibility_radius is not None
        trail_set = set(self.trail)
        rows: list[str] = []
        for y in range(env.size):
            chars = []
            for x in range(env.size):
                cell = (x, y)
                if fog and cell not in getattr(env, "_explored", set()):
                    chars.append(_FOG)
                elif cell == env._agent:
                    chars.append(_AGENT)
                elif cell == env.goal:
                    chars.append(_GOAL)
                elif cell in env.walls:
                    chars.append(_WALL)
                elif cell in trail_set:
                    chars.append(_PATH)   # 走过 → 路径标记
                else:
                    chars.append(_FLOOR)  # 看到没踩
            rows.append("".join(chars))
        return "\n".join(rows)

    def state(self, env: Any) -> dict:
        """渲染路径图 + trail 长度(无 LLM)。"""
        return {
            "path_map": self.render_with_path(env),
            "trail_length": len(self.trail),
        }


__all__ = ["PathTraceRegion"]
