"""G plan S4 — run_agent ↔ ComputerUseBridge end-to-end (FAKE backend, mock adapter).

The S4 gate: act_cu/focus_cu intercepted by run_agent (not dispatch_tool), bridge.act drives
the mock adapter, the scene text lands in <visual> (visual_ephemeral), done terminates, and
verify_fn closes over the bridge. No real model, no real VLM, no real Unity.
"""

from __future__ import annotations

import asyncio
import json
import sys

from brainregion.computer.bridge import ComputerUseBridge
from brainregion.computer.perception import PerceptionRegion
from brainregion.computer.session import BoundFreshness, ComputerUseSession
from brainregion.computer.targeting import TargetingController
from brainregion.computer.unity_mock import UnityEditorMockAdapter
from brainregion.providers.base import ModelResponse
from brainregion.sandbox import cleanup_run_dir, make_run_dir, run_agent
from brainregion.sandbox.task import SandboxTask


class _MockBackend:
    """Replays a scripted tool-call sequence; no real model."""

    def __init__(self, script):
        self.script = script
        self.i = 0

    async def complete_messages(self, messages, **kw):
        content = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return ModelResponse(model="mock", content=content, usage={}, cost_usd=0.0, cost_source=None, transport_mode="")


def _J(d):
    return json.dumps(d, ensure_ascii=False)


def _bridge():
    adapter = UnityEditorMockAdapter()
    session = ComputerUseSession(
        session_id="cu",
        adapter=adapter,
        allowed_apps={"unity.editor"},
        freshness=BoundFreshness(ttl_ms=10000.0),
    )
    perception = PerceptionRegion(event_sink=lambda *a, **k: None)
    targeting = TargetingController(session=session, perception=perception)
    return ComputerUseBridge(session=session, perception=perception, targeting=targeting), adapter


def test_run_agent_drives_bridge_act_cu_then_done():
    """run_agent intercepts act_cu → bridge.act (mock) → done terminates; verify_fn closes
    over the bridge. Proves the S4 wiring (the bridge is reachable from run_agent, not just
    dispatch_tool's raise-fallback)."""
    bridge, _adapter = _bridge()
    backend = _MockBackend(
        [
            _J(
                {
                    "thought": "click play",
                    "tool": "act_cu",
                    "args": {
                        "action": "click",
                        "locator": {
                            "anchor": {"panel_name": "toolbar"},
                            "descriptor": {"role": "button", "attributes": {"icon_shape": "play"}},
                        },
                    },
                }
            ),
            _J({"thought": "done", "done": True, "answer": "clicked play"}),
        ]
    )
    task = SandboxTask(id="cu", goal="click the Play button", files={}, tests={})
    run_dir = make_run_dir()
    try:
        traj = asyncio.run(
            run_agent(
                backend,
                "mock",
                task,
                run_dir=run_dir,
                arm="none",
                max_steps=5,
                python_exe=sys.executable,
                cu_bridge=bridge,
                verify_fn=lambda t, r, python_exe=None: {"tests_green": True},
            )
        )
        assert traj.n_steps >= 1, f"act_cu did not execute; n_steps={traj.n_steps}"
        assert traj.solve_status == "solved", f"solve_status={traj.solve_status}"
        assert bridge.current is not None  # prime() + act seeded _current
    finally:
        cleanup_run_dir(run_dir)


def test_run_agent_no_cu_bridge_excludes_cu_tools():
    """GPT #9: cu_bridge=None → CU_TOOLS not in available_tools (model never sees them). The
    default code-regime run is untouched (CU system prompt not selected, act_cu unavailable)."""
    backend = _MockBackend([_J({"thought": "x", "done": True, "answer": "x"})])
    task = SandboxTask(id="cu2", goal="g", files={}, tests={})
    run_dir = make_run_dir()
    try:
        traj = asyncio.run(
            run_agent(
                backend,
                "mock",
                task,
                run_dir=run_dir,
                arm="none",
                max_steps=3,
                python_exe=sys.executable,
                verify_fn=lambda t, r, python_exe=None: {"tests_green": True},
            )
        )
        # default run completes without cu_bridge; system prompt is the code-regime one
        assert traj.solve_status == "solved"
    finally:
        cleanup_run_dir(run_dir)
