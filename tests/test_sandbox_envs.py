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


# ---------- Phase B fog(部分可观)----------


def test_fog_render_hides_unexplored():
    """fog:未探索格显 `?`;start 邻域(Chebyshev 半径内)可见。"""
    env = GridWorld(size=6, start=(0, 0), goal=(5, 5), visibility_radius=1)
    rows = env.render().split("\n")
    # start (0,0) 邻域半径1 = (0,0)(1,0)(0,1)(1,1);其余 `?`
    assert rows[0][0] == "@"  # agent 可见
    assert rows[5][5] == "?"  # goal 远,未探索
    assert "?" in env.render()  # 有雾


def test_fog_explored_accumulates_and_persists():
    """移动后新格进 explored;离开的旧格仍可见(env-backed classic fog)。"""
    env = GridWorld(size=6, start=(0, 0), goal=(5, 5), visibility_radius=1)
    env.step("right")  # (0,0)->(1,0):新可见 (2,0)(1,1)(2,1)
    assert (2, 0) in env._explored and (1, 1) in env._explored
    env.step("left")  # 回 (0,0):(1,0) 仍 explored(持久)
    assert (1, 0) in env._explored


def test_fog_goal_hidden_until_within_radius():
    """goal 在半径外时显 `?`;agent 走近后显 G。"""
    env = GridWorld(size=6, start=(0, 0), goal=(5, 5), visibility_radius=1)
    assert "G" not in env.render()  # 初始 goal 远,隐藏
    for _ in range(4):
        env.step("right")  # -> (4,0)
    for _ in range(4):
        env.step("down")  # -> (4,4):dist to (5,5)=1 ≤ radius → 可见
    assert "G" in env.render()


def test_fog_radius_0_only_agent_cell():
    """radius=0:只 agent 格可见,余 `?`。"""
    env = GridWorld(size=3, start=(1, 1), goal=(0, 0), visibility_radius=0)
    rendered = env.render().replace("\n", "")
    assert rendered.count("@") == 1
    assert rendered.count("?") == 8  # 3x3 - 1 agent


def test_fog_reset_clears_explored():
    """reset 清 explored 回 start 可见域(re-explore)。"""
    env = GridWorld(size=6, start=(0, 0), goal=(5, 5), visibility_radius=1)
    env.step("right")
    env.step("right")
    assert len(env._explored) > 1
    env.reset()
    assert env._explored == set(env._visible_cells(env.start))


def test_no_fog_full_visible_regression():
    """radius=None(Phase A):无 `?`,全可见(回归不变)。"""
    env = GridWorld(size=4, start=(0, 0), goal=(3, 3))  # visibility_radius=None
    assert "?" not in env.render()
    assert env._explored == set()


def test_visibility_radius_validation():
    """review gpt:radius 负数 → ValueError;非 int → ValueError。"""
    with pytest.raises(ValueError, match="visibility_radius"):
        GridWorld(size=4, visibility_radius=-1)
    with pytest.raises(ValueError, match="visibility_radius"):
        GridWorld(size=4, visibility_radius=1.5)


def test_random_goal_fog_hidden_outside_start_visibility():
    """random_goal_seed + fog + 大网格:goal 落在 start 可见域外(逼探索)。"""
    env = GridWorld(size=8, start=(0, 0), visibility_radius=2, random_goal_seed=42)
    start_vis = set(env._visible_cells(env.start))
    assert env.goal not in start_vis  # 藏起来
    assert env.goal != env.start


def test_random_goal_fallback_when_start_visibility_covers_grid():
    """review consensus:小网格+大半径 → start 可见域覆盖全网格 → fallback 任意非 start(不崩)。"""
    env = GridWorld(size=3, start=(0, 0), visibility_radius=5, random_goal_seed=1)
    assert env.goal != env.start  # fallback 仍合法
    assert env.goal in [(x, y) for y in range(3) for x in range(3)]


def test_random_goal_deterministic_same_seed():
    """同 seed → 同 goal(可复现)。"""
    a = GridWorld(size=7, start=(0, 0), visibility_radius=2, random_goal_seed=7)
    b = GridWorld(size=7, start=(0, 0), visibility_radius=2, random_goal_seed=7)
    assert a.goal == b.goal


# ---------- Phase C:strict_obs + render_visible + observation() ----------


def test_render_visible_only_current_view():
    """render_visible 只画当前 Chebyshev 视野(视野外 `?`,即使探索过)。"""
    env = GridWorld(size=6, start=(0, 0), goal=(5, 5), visibility_radius=1)
    env.step("right")  # 探索过 (0,0) 邻域;agent 现 (1,0)
    visible = env.render_visible().split("\n")
    # agent (1,0) 视野半径1 = (0,0)(1,0)(2,0)(0,1)(1,1)(2,1);(0,0) 探索过但在视野外 → `?`
    rendered_flat = "".join(visible)
    assert visible[0][1] == "@"  # agent 可见
    # (0,0) 探索过但距 agent(1,0) Chebyshev=1 → 在视野内,应显(非 ?);(5,5) 视野外 → ?
    assert visible[5][5] == "?"
    assert "?" in rendered_flat  # 严格视野有雾(探索过但视野外的也算 ?)


def test_observation_respects_strict_obs():
    """observation():strict → 当前视野(探索过但视野外显 ?);loose → 累积图(探索过显 .)。
    需移动 2+ 步让旧格出当前视野(1 步时 explored 恰好 == 当前视野看不出差异)。"""
    env_strict = GridWorld(size=5, start=(0, 0), goal=(4, 4), visibility_radius=1, strict_obs=True)
    env_loose = GridWorld(size=5, start=(0, 0), goal=(4, 4), visibility_radius=1, strict_obs=False)
    for env in (env_strict, env_loose):
        env.step("right")
        env.step("right")  # agent (2,0);(0,0) 探索过但在 (2,0) 视野(Chebyshev 2)外
    strict_rows = env_strict.observation().split("\n")
    loose_rows = env_loose.observation().split("\n")
    assert strict_rows[0][0] == "?"  # (0,0) 探索过但出视野 → strict 隐藏
    assert loose_rows[0][0] == "."   # (0,0) 探索过 → loose 累积图显地


def test_observation_loose_equals_render_regression():
    """strict_obs=False:observation() == render()(Phase A/B 零回归)。"""
    env = GridWorld(size=5, start=(0, 0), goal=(4, 4), visibility_radius=2, strict_obs=False)
    env.step("right")
    env.step("down")
    assert env.observation() == env.render()


def test_step_returns_strict_observation_when_strict():
    """strict_obs=True:step 返当前视野(observation);frames 仍累积图(render)。
    移动 2 步让旧格出视野 → observation != render(累积)。"""
    env = GridWorld(size=5, start=(0, 0), goal=(4, 4), visibility_radius=1, strict_obs=True)
    env.step("right")  # (0,0)->(1,0)
    obs, _, _, _ = env.step("right")  # (1,0)->(2,0);(0,0) 探索过但出 (2,0) 视野
    assert obs == env.render_visible()  # 当前视野
    assert env.frames[-1] == env.render()  # replay 用累积图
    assert obs != env.render()  # 严格视野 != 累积图(累积含 (0,0),视野不含)

