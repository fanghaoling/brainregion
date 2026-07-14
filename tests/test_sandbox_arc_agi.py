from __future__ import annotations

import json
import asyncio
from dataclasses import dataclass

import pytest

from brainregion.sandbox.envs.arc_agi import ArcAgiEnv
from brainregion.sandbox.isolation import cleanup_run_dir, make_run_dir
from brainregion.sandbox.loop import ToolCall, dispatch_tool, run_agent, scoped_env
from brainregion.sandbox.task import SandboxTask
from brainregion.cli import build_parser


@dataclass(frozen=True)
class _State:
    value: str


class _Action:
    def __init__(self, name: str, *, complex_action: bool = False) -> None:
        self.name = name
        self._complex = complex_action
        self.validated: list[dict] = []

    def is_complex(self) -> bool:
        return self._complex

    def validate_data(self, data: dict) -> bool:
        if set(data) != {"x", "y"}:
            raise ValueError("x/y required")
        self.validated.append(dict(data))
        return True


class _Frame:
    def __init__(
        self,
        *,
        state: str = "NOT_FINISHED",
        completed: int = 0,
        actions: list[int] | None = None,
        grid: list[list[int]] | None = None,
    ) -> None:
        self.game_id = "fake-1234"
        self.state = _State(state)
        self.levels_completed = completed
        self.win_levels = 2
        self.available_actions = actions or []
        self.frame = [grid or [[0, 1], [2, 35]]]


class _Wrapper:
    def __init__(self, initial: _Frame, actions: list[_Action], responses: list[_Frame]) -> None:
        self.initial = initial
        self.action_space = actions
        self.responses = list(responses)
        self.calls: list[tuple[_Action, dict | None]] = []

    def reset(self):
        return self.initial

    def step(self, action, data=None):
        self.calls.append((action, data))
        return self.responses.pop(0)


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content
        self.error = None
        self.usage = {}
        self.cost_usd = 0.0
        self.cost_source = None

    @property
    def ok(self) -> bool:
        return True


class _Backend:
    def __init__(self) -> None:
        self.turn = 0

    async def complete_messages(self, messages, **kwargs):
        del messages, kwargs
        self.turn += 1
        if self.turn == 1:
            return _Response(
                json.dumps(
                    {"thought": "try an available action", "tool": "act", "args": {"action": "action1"}}
                )
            )
        return _Response(json.dumps({"thought": "environment won", "done": True, "answer": "done"}))


def test_arc_adapter_encodes_frame_and_dynamic_action_contract_without_sdk_import():
    wrapper = _Wrapper(_Frame(), [_Action("ACTION1"), _Action("ACTION6", complex_action=True)], [])
    env = ArcAgiEnv(wrapper=wrapper, game_id="fake")

    env.reset()
    snapshot = json.loads(env.observation())

    assert env.action_vocab == ("action1", "action6")
    assert snapshot["frame_encoding"] == "base36_grid"
    assert snapshot["frame"] == ["01", "2z"]
    assert snapshot["palette"] == [0, 1, 2, 35]
    assert snapshot["available_actions"] == [
        {"name": "action1", "requires_data": False, "data_schema": None},
        {
            "name": "action6",
            "requires_data": True,
            "data_schema": {"x": "integer 0..63", "y": "integer 0..63"},
        },
    ]


def test_arc_adapter_validates_complex_data_and_tracks_level_reward():
    complex_action = _Action("ACTION6", complex_action=True)
    wrapper = _Wrapper(
        _Frame(),
        [complex_action],
        [_Frame(state="WIN", completed=2, actions=[6])],
    )
    env = ArcAgiEnv(wrapper=wrapper, game_id="fake")
    env.reset()

    with pytest.raises(ValueError, match="requires x/y data"):
        env.step("action6")
    observation, reward, terminated, info = env.step("action6", data={"x": 12, "y": 34})

    assert json.loads(observation)["state"] == "WIN"
    assert reward == 2.0
    assert terminated is True
    assert env.solved is True
    assert env.total_reward == 2.0
    assert info["levels_completed"] == 2
    assert complex_action.validated == [{"x": 12, "y": 34}]
    assert wrapper.calls == [(complex_action, {"x": 12, "y": 34})]
    assert env.action_trace == [
        {
            "index": 0,
            "action": "action6",
            "uses_data": True,
            "frame_changed": False,
            "frame_hash": env.action_trace[0]["frame_hash"],
            "state": "WIN",
            "levels_completed": 2,
            "available_action_count": 1,
        }
    ]


def test_dispatch_tool_passes_structured_action_data_only_to_capable_env():
    complex_action = _Action("ACTION6", complex_action=True)
    wrapper = _Wrapper(_Frame(), [complex_action], [_Frame(completed=1)])
    env = ArcAgiEnv(wrapper=wrapper, game_id="fake")
    env.reset()
    call = ToolCall(
        thought="probe a coordinate",
        tool="act",
        args={"action": "action6", "data": {"x": 3, "y": 4}},
        done=False,
        answer="",
    )

    with scoped_env(env):
        result, error = dispatch_tool(call)

    assert error is None
    assert json.loads(result)["reward"] == 1.0
    assert wrapper.calls == [(complex_action, {"x": 3, "y": 4})]


def test_arc_adapter_rejects_data_for_simple_action():
    simple = _Action("ACTION1")
    wrapper = _Wrapper(_Frame(), [simple], [_Frame()])
    env = ArcAgiEnv(wrapper=wrapper, game_id="fake")
    env.reset()

    with pytest.raises(ValueError, match="does not accept data"):
        env.step("action1", data={"x": 1, "y": 2})


def test_arc_env_cli_is_explicit_and_bounded():
    args = build_parser().parse_args(
        [
            "sandbox",
            "arc-env",
            "--game",
            "ls20",
            "--main-brain",
            "buzz_anthropic/claude-sonnet-5",
            "--max-steps",
            "6",
            "--max-cost-usd",
            "0.05",
        ]
    )

    assert args.sandbox_command == "arc-env"
    assert args.game == "ls20"
    assert args.max_steps == 6
    assert args.max_cost_usd == 0.05
    assert args.thinking == "off"


def test_run_agent_does_not_require_gridworld_private_position():
    simple = _Action("ACTION1")
    wrapper = _Wrapper(_Frame(), [simple], [_Frame(state="WIN", completed=2)])
    env = ArcAgiEnv(wrapper=wrapper, game_id="fake")
    env.reset()
    task = SandboxTask(id="arc-fake", goal="discover and finish")
    run_dir = make_run_dir(prefix="brainregion-arc-fake-")
    try:
        with scoped_env(env):
            trajectory = asyncio.run(
                run_agent(
                    _Backend(),
                    "mock",
                    task,
                    run_dir=run_dir,
                    max_steps=2,
                    system_prompt=env.build_system_prompt(task.goal),
                    verify_fn=lambda *_args, **_kwargs: {
                        "tests_green": env.solved,
                        "solve_status": "solved" if env.solved else "tests_fail",
                        "pytest": None,
                        "gold_diff": "",
                    },
                )
            )
    finally:
        cleanup_run_dir(run_dir)

    assert trajectory.done is True
    assert trajectory.tests_green is True
    assert trajectory.termination_reason == "done"
    assert wrapper.calls == [(simple, None)]
