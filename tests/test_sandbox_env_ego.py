"""Phase 4.8 ego-relative action 测试 —— ego vocab / heading / turn / forward / dead-reckon /
topo recipe / replay / golden 等价 / prompt。

abs 零回归在各自现有测试覆盖(test_sandbox_envs/region/eval/loop/strategy,897 全绿);本文件测 ego 新行为
+ **desync 探测硬线**(memory.heading==env._heading 每步,catch dead-reckon 失步级联)+ ActionModel golden
等价(新表==合并前 4 份 _ACTION_DELTA,review opus-2)。
"""
from __future__ import annotations

import json

import pytest

from brainregion.sandbox.envs._actions import (
    ABS, EGO, ABS_DELTA, ABS_DIR_HEADING, INITIAL_HEADING, relative_direction,
)
from brainregion.sandbox.envs import build_env_system_prompt
from brainregion.sandbox.envs.gridworld import GridWorld
from brainregion.sandbox.regions.memory_region import MemoryRegion
from brainregion.sandbox.regions.topo_region import TopologicalRegion
from brainregion.sandbox.loop import ToolCall, dispatch_tool, scoped_env
from brainregion.sandbox.env_eval import _positions_from_traj


# ---------- traj mock(避依赖 eval 测试的 _T/_S helper)----------
class _Step:
    def __init__(self, tool: str, action: str) -> None:
        self.tool = tool
        self.args = {"action": action}


class _Traj:
    def __init__(self, steps: list[_Step]) -> None:
        self.steps = steps


def _tc(tool: str, args: dict | None = None) -> ToolCall:
    return ToolCall(thought="", tool=tool, args=args or {}, done=False, answer="")


# ==================== ActionModel golden 等价(review opus-2)====================
def test_actionmodel_abs_delta_matches_legacy():
    """新 ActionModel.ABS delta == 合并前 gridworld/memory/env_eval 三份 _ACTION_DELTA(显式等价)。"""
    legacy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
    for a, d in legacy.items():
        assert ABS.delta(a, "E") == d          # abs 忽略 heading
    assert ABS_DELTA == legacy
    assert tuple(ABS_DELTA.keys()) == ABS.vocab


def test_actionmodel_ego_vocab_no_turn180():
    assert EGO.vocab == ("forward", "turn_left", "turn_right")   # turn_180 defer


def test_heading_after_cycles():
    h = "E"
    seq = []
    for _ in range(4):
        h = EGO.heading_after("turn_left", h)
        seq.append(h)
    assert seq == ["N", "W", "S", "E"]           # 左转 4 次回原
    h = "E"
    for _ in range(4):
        h = EGO.heading_after("turn_right", h)
    assert h == "E"


def test_heading_illegal_raises():
    with pytest.raises(ValueError):
        EGO.heading_after("turn_left", "X")
    with pytest.raises(ValueError):
        EGO.delta("forward", "Q")


def test_relative_direction_four_facings():
    # 面东(E):right=forward / up=left / down=right / left=back
    assert relative_direction("right", "E") == "forward"
    assert relative_direction("up", "E") == "left"
    assert relative_direction("down", "E") == "right"
    assert relative_direction("left", "E") == "back"
    # 面北:up=forward
    assert relative_direction("up", "N") == "forward"
    # 面西:up=right(面西时北在右)
    assert relative_direction("up", "W") == "right"
    assert ABS_DIR_HEADING == {"up": "N", "down": "S", "left": "W", "right": "E"}


# ==================== GridWorld ego ====================
def test_ego_vocab_and_initial_heading_fixed():
    g = GridWorld(size=5, ego_actions=True)
    assert g.action_vocab == ("forward", "turn_left", "turn_right")
    assert g._heading == INITIAL_HEADING == "E"   # 固定 East(避 maze 布局混杂,D3)
    assert g.ego_actions is True


def test_abs_default_unchanged():
    g = GridWorld(size=5)
    assert g.ego_actions is False
    assert g.action_vocab == ("up", "down", "left", "right")
    assert g._heading is None                     # abs 无 heading


def test_ego_turn_does_not_move_returns_turned():
    g = GridWorld(size=5, ego_actions=True)
    before = g._agent
    _o, _r, _t, info = g.step("turn_left")
    assert g._agent == before                     # 位置不变(钉死 status bug,review Plan agent)
    assert info == {"turned": True}
    assert g._heading == "N"


def test_ego_turn_left_cycles_heading():
    g = GridWorld(size=5, ego_actions=True)
    seq = []
    for _ in range(4):
        g.step("turn_left")
        seq.append(g._heading)
    assert seq == ["N", "W", "S", "E"]


def test_ego_forward_moves_along_heading():
    g = GridWorld(size=5, start=(2, 2), ego_actions=True)   # heading E
    g.step("forward")                                       # E → (3,2)
    assert g._agent == (3, 2)
    g.step("turn_right")                                    # E→S
    g.step("forward")                                       # S → (3,3)
    assert g._agent == (3, 3)


def test_ego_forward_blocked_keeps_pos_and_heading():
    g = GridWorld(size=5, start=(0, 0), walls=((1, 0),), ego_actions=True)  # E → (1,0) 墙
    _o, _r, _t, info = g.step("forward")
    assert g._agent == (0, 0)
    assert g._heading == "E"
    assert info == {"blocked": True}


def test_ego_already_done_terminal():
    g = GridWorld(size=5, start=(0, 0), goal=(1, 0), ego_actions=True)
    g.step("forward")                            # 到 goal
    assert g.solved
    _o, _r, _t, info = g.step("forward")         # 已 done
    assert info == {"already_done": True}


def test_ego_reset_restores_heading():
    """reset 恢复初始 heading(跨 episode 不继承上局航向;review gpt-0/gpt-5 high)。"""
    g = GridWorld(size=5, start=(2, 2), ego_actions=True)
    g.step("turn_left")                          # E→N
    assert g._heading == "N"
    g.reset()
    assert g._heading == INITIAL_HEADING
    assert g._agent == g.start


def test_ego_invalid_action():
    g = GridWorld(size=5, ego_actions=True)
    _o, _r, _t, info = g.step("up")              # abs 词,ego vocab 不含
    assert info == {"invalid": "up"}


# ==================== MemoryRegion heading dead-reckon(守 D.2)+ desync 探测 ====================
def test_memory_ego_turn_rotates_heading_not_pose():
    m = MemoryRegion(start=(0, 0), action_model=EGO, heading="E")
    m.update("turn_left", "turned", "@.")
    assert m.heading == "N"
    assert m.pose == (0, 0)


def test_memory_ego_forward_advances_along_heading():
    m = MemoryRegion(start=(0, 0), action_model=EGO, heading="E")
    m.update("forward", "moved", "@.")
    assert m.pose == (1, 0)
    assert m.heading == "E"


def test_memory_ego_forward_blocked_no_drift():
    m = MemoryRegion(start=(0, 0), action_model=EGO, heading="E")
    m.update("forward", "blocked", "@#")
    assert m.pose == (0, 0)
    assert m.heading == "E"


def test_memory_abs_default_zero_regression():
    m = MemoryRegion(start=(0, 0))               # 默认 ABS
    m.update("right", "moved", "@.")
    assert m.pose == (1, 0)


def test_memory_desync_detector():
    """desync 探测硬线:scripted turn+forward 过 env + 镜像 memory.update,**每步** pose/heading 一致。

    catch heading dead-reckon 失步级联(heading 无纠偏信号,失步→之后 forward pose 全错;review opus-9/Risk#2)。
    """
    g = GridWorld(size=7, start=(3, 3), ego_actions=True)
    m = MemoryRegion(start=g.start, action_model=EGO, heading=g._initial_heading)
    assert m.pose == g._agent and m.heading == g._heading
    seq = [("turn_left", "turned"), ("forward", "moved"), ("turn_right", "turned"),
           ("forward", "moved"), ("forward", "moved")]
    for action, status in seq:
        g.step(action)
        m.update(action, status, g.relative_view())
        assert m.pose == g._agent, f"pose desync after {action}: mem {m.pose} env {g._agent}"
        assert m.heading == g._heading, f"heading desync after {action}: mem {m.heading} env {g._heading}"


# ==================== Topo ego recipe ====================
def test_topo_ego_relative_directions_no_abs_leak():
    g = GridWorld(size=5, ego_actions=True, visibility_radius=2)
    topo = TopologicalRegion(start=g.start)
    topo.update(g._agent)
    st = topo.state(g)
    assert st["heading"] == "E"
    for d in st["frontier_directions"]:
        assert d in ("forward", "left", "right", "back")   # 相对方位,无 abs 词(review opus-3/gpt-3)
    assert "up" not in st["suggestion"] and "down" not in st["suggestion"]


def test_topo_abs_default_unchanged():
    g = GridWorld(size=5, visibility_radius=2)
    topo = TopologicalRegion(start=g.start)
    topo.update(g._agent)
    st = topo.state(g)
    for d in st["frontier_directions"]:
        assert d in ABS_DELTA                      # abs 词(零回归)


# ==================== Replay _positions_from_traj 追 heading(查墙,opus-0)====================
def test_replay_ego_tracks_heading_final_matches_env():
    """ego 重放:turn→旋转,forward→delta 查墙;末位置==env(一致性,consensus C1 + plan F)。"""
    g = GridWorld(size=7, start=(3, 3), ego_actions=True)
    actions = ["turn_left", "forward", "turn_right", "forward", "forward"]
    traj = _Traj([_Step("act", a) for a in actions])
    for a in actions:
        g.step(a)
    g_replay = GridWorld(size=7, start=(3, 3), ego_actions=True)   # 未 step 的同配置 env
    positions = _positions_from_traj(traj, g_replay)
    assert positions[-1] == g._agent              # 重放末位置 == 真实 env 末位置


def test_replay_ego_forward_into_wall_no_move():
    """重放 forward 撞墙不位移(否则一致性断言失败;review opus-0)。"""
    g = GridWorld(size=5, start=(0, 0), walls=((1, 0),), ego_actions=True)
    traj = _Traj([_Step("act", "forward")])       # E → (1,0) 墙
    positions = _positions_from_traj(traj, g)
    assert positions == [(0, 0), (0, 0)]          # 撞墙不位移


def test_replay_abs_unchanged():
    g = GridWorld(size=5, start=(0, 0))
    traj = _Traj([_Step("act", "right"), _Step("act", "down")])
    positions = _positions_from_traj(traj, g)
    assert positions == [(0, 0), (1, 0), (1, 1)]  # abs 零回归


# ==================== dispatch observe/act 带 heading metadata(GPT#2)====================
def test_dispatch_observe_ego_returns_heading():
    g = GridWorld(size=5, ego_actions=True, visibility_radius=2, strict_obs=True)
    with scoped_env(g):
        result_str, err = dispatch_tool(_tc("observe"))
    assert err is None
    obj = json.loads(result_str)
    assert obj["heading"] == "E"                   # heading metadata(不进 render)


def test_dispatch_act_ego_turn_returns_turned_info():
    g = GridWorld(size=5, ego_actions=True)
    with scoped_env(g):
        result_str, err = dispatch_tool(_tc("act", {"action": "turn_left"}))
    assert err is None
    obj = json.loads(result_str)
    assert obj["info"] == {"turned": True}
    assert obj["heading"] == "N"


def test_dispatch_observe_abs_no_heading():
    g = GridWorld(size=5, visibility_radius=2, strict_obs=True)   # abs
    with scoped_env(g):
        result_str, err = dispatch_tool(_tc("observe"))
    obj = json.loads(result_str)
    assert "heading" not in obj                    # abs 无 heading(零回归)


# ==================== prompt ego(review opus-3/gpt-3)====================
def test_prompt_ego_no_abs_action_words():
    g = GridWorld(size=5, ego_actions=True, visibility_radius=2)
    p = build_env_system_prompt(g, "到 G", memory=True)
    assert "forward" in p and "turn_left" in p     # ego 动作词
    assert "动作词表:forward" in p
    # ego prompt 不应含 abs 动作词(作动作词;legend 的 @/#/. 不算)
    for w in ("动作词表:up", "up, down", "向 right"):
        assert w not in p


def test_prompt_abs_unchanged():
    g = GridWorld(size=5, visibility_radius=2)
    p = build_env_system_prompt(g, "到 G", memory=True)
    assert "动作词表:up, down, left, right" in p
    assert "ego-relative" not in p
