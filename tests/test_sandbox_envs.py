"""GridWorld(Phase A env)单测。

覆盖 plan A.1 + review 双强硬化(2026-07-08):确定性 / step 语义 / terminal guard /
非法 action / 撞墙越界 / render / frames 有界+reset 清空 / 构造边界校验 / reset(None) 行为定义。
"""
from __future__ import annotations

import pytest

from brainregion.sandbox.envs import GridWorld, render_replay_html, write_replay_html


# ---------- 构造 + 边界校验 ----------


def test_default_5x5_start_corner_goal_far_corner():
    env = GridWorld()
    assert env.size == 5
    assert env.start == (0, 0)
    assert env.goal == (4, 4)
    assert env.solved is False
    assert env.total_reward == 0.0
    assert env.action_vocab == ("up", "down", "left", "right")


def test_size_out_of_range_raises():
    with pytest.raises(ValueError, match="size"):
        GridWorld(size=1)
    with pytest.raises(ValueError, match="size"):
        GridWorld(size=51)


def test_goal_start_on_wall_or_oob_or_equal_raises():
    with pytest.raises(ValueError, match="越界"):
        GridWorld(size=3, goal=(5, 5))
    with pytest.raises(ValueError, match="start.*落在墙"):
        GridWorld(size=3, start=(1, 1), walls=((1, 1),))
    with pytest.raises(ValueError, match="goal.*落在墙"):
        GridWorld(size=3, goal=(2, 2), walls=((2, 2),))
    with pytest.raises(ValueError, match="不能等于"):
        GridWorld(size=3, start=(1, 1), goal=(1, 1))


# ---------- render ----------


def test_render_5x5_agent_and_goal_and_walls():
    env = GridWorld(size=5, start=(0, 0), goal=(4, 4), walls=((2, 2),))
    rendered = env.render()
    rows = rendered.split("\n")
    assert len(rows) == 5
    assert all(len(r) == 5 for r in rows)
    assert rows[0][0] == "@"  # agent at start
    assert rows[4][4] == "G"  # goal
    assert rows[2][2] == "#"  # wall


# ---------- step 语义 ----------


def test_step_valid_move_no_reward_not_terminated():
    env = GridWorld(size=5, start=(0, 0), goal=(4, 4))
    obs, reward, terminated, info = env.step("right")
    assert env._agent == (1, 0)
    assert reward == 0.0
    assert terminated is False
    assert info == {}
    assert env.solved is False


def test_step_reach_goal_reward_one_terminated():
    env = GridWorld(size=3, start=(0, 0), goal=(1, 0))  # 一步即达
    obs, reward, terminated, info = env.step("right")
    assert reward == 1.0
    assert terminated is True
    assert info == {"goal": True}
    assert env.solved is True
    assert env.total_reward == 1.0


def test_step_action_case_and_whitespace_normalized():
    env = GridWorld(size=3, start=(0, 0), goal=(1, 0))
    obs, reward, terminated, _ = env.step("  RIGHT  ")
    assert terminated is True and reward == 1.0


def test_step_invalid_action_no_move_no_crash():
    env = GridWorld(size=5, start=(2, 2), goal=(4, 4))
    before = env._agent
    obs, reward, terminated, info = env.step("fly")
    assert env._agent == before
    assert reward == 0.0
    assert terminated is False
    assert info == {"invalid": "fly"}


def test_step_blocked_by_wall_no_move():
    env = GridWorld(size=3, start=(0, 0), goal=(2, 2), walls=((1, 0),))
    obs, reward, terminated, info = env.step("right")
    assert env._agent == (0, 0)
    assert reward == 0.0
    assert terminated is False
    assert info == {"blocked": True}


def test_step_blocked_out_of_bounds_no_move():
    env = GridWorld(size=3, start=(0, 0), goal=(2, 2))
    obs, reward, terminated, info = env.step("left")  # 越界
    assert env._agent == (0, 0)
    assert info == {"blocked": True}


# ---------- terminal guard ----------


def test_step_after_terminated_already_done_no_double_reward():
    env = GridWorld(size=3, start=(0, 0), goal=(1, 0))
    env.step("right")  # 抵达 goal,terminated
    assert env.total_reward == 1.0
    obs, reward, terminated, info = env.step("right")  # terminal 后再 act
    assert env._agent == (1, 0)  # 不动
    assert reward == 0.0
    assert terminated is True
    assert info == {"already_done": True}
    assert env.total_reward == 1.0  # 不重计


# ---------- 确定性 + reset ----------


def test_reset_clears_state_and_frames():
    env = GridWorld(size=5, start=(0, 0), goal=(4, 4))
    env.step("right")
    env.step("down")
    assert len(env.frames) == 3  # 初始 + 2 move
    obs = env.reset()
    assert env._agent == (0, 0)
    assert env.solved is False
    assert env.total_reward == 0.0
    assert env._terminated is False
    assert env.frames == [env.render()]  # 清空回初始单帧


def test_reset_seed_none_deterministic_defined_behavior():
    """review 硬化:reset(seed=None) 行为必须定义且确定(显式布局下与任意 seed 一致)。"""
    env = GridWorld(size=4, start=(0, 0), goal=(3, 3), walls=((1, 1), (2, 2)))
    obs_none = env.reset(seed=None)
    obs_7 = env.reset(seed=7)
    obs_99 = env.reset(seed=99)
    assert obs_none == obs_7 == obs_99  # 显式布局 → 无 RNG → 任意 seed 一致


def test_deterministic_same_layout_same_trajectory():
    """同显式布局,同动作序列 → 同 obs 序列(确定性 grounding 前提)。"""
    actions = ["right", "down", "right", "down"]

    def run():
        env = GridWorld(size=5, start=(0, 0), goal=(4, 4), walls=((2, 0),))
        return [env.step(a)[0] for a in actions]

    assert run() == run()


# ---------- frames 有界 ----------


def test_frames_one_per_state_change_bounded_by_steps():
    """无效/撞墙/terminal 不追加 frame;frames 只记 distinct 态,随步数有界(无膨胀)。"""
    env = GridWorld(size=5, start=(0, 0), goal=(3, 0))
    env.step("fly")          # invalid → 无 frame
    env.step("left")         # blocked(oob)→ 无 frame
    env.step("right")        # move → +1 frame
    env.step("up")           # blocked(oob)→ 无 frame
    env.step("right")        # move → +1 frame
    env.step("right")        # reach goal → +1 frame(terminated)
    env.step("right")        # already_done → 无 frame
    assert len(env.frames) == 4  # 初始 + 3 state-changing steps


# ---------- replay HTML(XSS 硬化 gpt #19 + utf-8 opus #6)----------


def test_render_replay_html_contains_frames_and_meta():
    html = render_replay_html(["@.G", ".@G"], {"goal": "到达 G", "solved": True, "n_steps": 2})
    assert "<html" in html and "回放" in html          # 结构 + 中文标题
    assert "@.G" in html and ".@G" in html             # 帧进 JSON(无 <,原样)
    assert "到达 G" in html                            # meta 进 JSON
    assert "textContent" in html                       # DOM 用 textContent(非 innerHTML)


def test_render_replay_html_escapes_script_injection():
    """review gpt #19:frame 含 </script><script>alert(1)</script> 不得破出/执行。"""
    payload = "</script><script>alert(1)</script>"
    html = render_replay_html([payload], {"note": payload})
    assert "<script>alert" not in html                 # 注入的 <script> 被转义(<script)
    assert "\\u003c" in html                           # < 确实被转义
    assert "alert(1)" in html                          # 内容保留(在 JSON 串里,非执行位)


def test_render_replay_html_escapes_img_onerror():
    """review gpt #19:frame 含 <img onerror=...> 不得注入。"""
    html = render_replay_html(["<img onerror=alert(1)>"], {})
    assert "<img onerror" not in html                  # < 被转义 → 非标签
    assert "\\u003cimg" in html


def test_write_replay_html_explicit_utf8(tmp_path):
    """review opus #6:写文件显式 utf-8(不靠 PYTHONIOENCODING,避 cp936 乱码)。"""
    env = GridWorld(size=3, start=(0, 0), goal=(2, 2))
    out = write_replay_html(tmp_path / "replay.html", env.frames, {"goal": "到达目标"})
    content = out.read_text(encoding="utf-8")
    assert "回放" in content and "到达目标" in content

