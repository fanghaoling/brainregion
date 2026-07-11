from __future__ import annotations

import asyncio
import json
from pathlib import Path

from brainregion.providers.base import ModelResponse
from brainregion.sandbox import cleanup_run_dir, make_run_dir, materialize_fixture, run_agent
from brainregion.sandbox.fixtures import get_fixture
from brainregion.sandbox.regions import VerificationOptionRegion
from brainregion.workspace import read_text
from brainregion.workspace.files import scoped_workspace_root


def _j(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False)


class ScriptBackend:
    def __init__(self, script: list[str]):
        self.script = script
        self.index = 0
        self.last_messages: list[dict] = []

    async def complete_messages(self, messages, **kwargs):
        self.last_messages = [dict(item) for item in messages]
        content = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        return ModelResponse(
            model=kwargs.get("model", "mock"), content=content, usage={}, cost_usd=0.0,
        )


def _materialized_fixture():
    task = get_fixture("off_by_one")
    run_dir = make_run_dir()
    materialize_fixture(task, Path(run_dir))
    with scoped_workspace_root(run_dir):
        sha = read_text("ranges.py")["sha256"]
    return task, run_dir, sha


def _patch_call(sha: str, *, replacement: str, dry_run: bool = False) -> str:
    return _j({
        "thought": "修补边界",
        "tool": "apply_text_patch",
        "args": {
            "path": "ranges.py",
            "expected_sha256": sha,
            "replacements": [{"old_text": "range(start, end)", "new_text": replacement}],
            "dry_run": dry_run,
        },
    })


def test_verification_option_runs_after_real_patch_and_replaces_manual_test_turn():
    task, run_dir, sha = _materialized_fixture()
    backend = ScriptBackend([
        _patch_call(sha, replacement="range(start, end + 1)"),
        _j({"thought": "自动验证已通过", "done": True, "answer": "fixed"}),
    ])
    try:
        traj = asyncio.run(run_agent(
            backend, "mock", task, run_dir=run_dir, max_steps=3,
            option_region=VerificationOptionRegion(), option_continuous=True,
        ))
    finally:
        cleanup_run_dir(run_dir)

    assert traj.tests_green is True and traj.n_steps == 2
    assert traj.workspace_effects == 1 and traj.verification_runs == 1
    assert traj.last_verification_passed is True
    assert traj.automatic_region_activations == 1
    assert traj.option_activations[0]["trigger"] == "after_main_effect"
    assert traj.option_activations[0]["region"] == "verification"
    assert traj.option_activations[0]["actions"] == ["run_check"]
    injected = "\n".join(m["content"] for m in backend.last_messages if m["role"] == "user")
    assert 'actor="verification_region"' in injected
    assert '"stop_reason": "decision_boundary:verification_complete"' in injected
    assert '"solved": true' in injected


def test_verification_option_returns_failed_test_evidence():
    task, run_dir, sha = _materialized_fixture()
    backend = ScriptBackend([
        _patch_call(sha, replacement="range(start, end - 1)"),
        _j({"thought": "看到失败证据", "done": True, "answer": "not fixed"}),
    ])
    try:
        traj = asyncio.run(run_agent(
            backend, "mock", task, run_dir=run_dir, max_steps=3,
            option_region=VerificationOptionRegion(), option_continuous=True,
        ))
    finally:
        cleanup_run_dir(run_dir)

    assert traj.tests_green is False
    assert traj.verification_runs == 1 and traj.last_verification_passed is False
    assert traj.option_activations[0]["solved"] is False
    injected = "\n".join(m["content"] for m in backend.last_messages if m["role"] == "user")
    assert '"status": "failed"' in injected
    assert '"exit_code": 1' in injected


def test_verification_option_ignores_dry_run_patch():
    task, run_dir, sha = _materialized_fixture()
    backend = ScriptBackend([
        _patch_call(sha, replacement="range(start, end + 1)", dry_run=True),
        _j({"thought": "dry run only", "done": True, "answer": "unchanged"}),
    ])
    try:
        traj = asyncio.run(run_agent(
            backend, "mock", task, run_dir=run_dir, max_steps=2,
            option_region=VerificationOptionRegion(), option_continuous=True,
        ))
    finally:
        cleanup_run_dir(run_dir)
    assert traj.workspace_effects == 0 and traj.verification_runs == 0


def test_verification_option_ignores_noop_patch():
    task, run_dir, sha = _materialized_fixture()
    backend = ScriptBackend([
        _patch_call(sha, replacement="range(start, end)"),
        _j({"thought": "no change", "done": True, "answer": "unchanged"}),
    ])
    try:
        traj = asyncio.run(run_agent(
            backend, "mock", task, run_dir=run_dir, max_steps=2,
            option_region=VerificationOptionRegion(), option_continuous=True,
        ))
    finally:
        cleanup_run_dir(run_dir)
    assert traj.workspace_effects == 0 and traj.verification_runs == 0


def test_verification_option_ignores_failed_patch():
    task, run_dir, _sha = _materialized_fixture()
    backend = ScriptBackend([
        _patch_call("0" * 64, replacement="range(start, end + 1)"),
        _j({"thought": "patch failed", "done": True, "answer": "unchanged"}),
    ])
    try:
        traj = asyncio.run(run_agent(
            backend, "mock", task, run_dir=run_dir, max_steps=2,
            option_region=VerificationOptionRegion(), option_continuous=True,
        ))
    finally:
        cleanup_run_dir(run_dir)
    assert traj.workspace_effects == 0 and traj.verification_runs == 0


def test_verification_region_deduplicates_same_effect():
    region = VerificationOptionRegion()
    effect = {"effect_id": "1:abc"}
    assert region.next_action(effect) == "run_check"
    region.observe_transition(
        action="run_check",
        observation={"ok": True, "status": "passed", "kind": "python -m pytest"},
        status="passed",
    )
    assert region.next_action(effect) is None


def test_verification_region_cli_flag():
    from brainregion.cli import build_parser

    args = build_parser().parse_args([
        "sandbox", "run", "--main-brain", "deepseek-v4-flash", "--verification-region",
    ])
    assert args.verification_region is True
