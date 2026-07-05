from __future__ import annotations

from brainregion.core.wake import wake_gate


def test_project_understanding_memory_task_wakes_memory_region():
    out = wake_gate(
        problem="整理 BrainRegion 项目理解、记忆卡片和上下文压缩经验。",
        gold_regions=["memory"],
        top_k=5,
    )

    assert "memory" in out["activated_regions"]["woken"]
    assert out["wake_metrics"]["hit"] == ["memory"]
    tools = [action["tool"] for action in out["suggested_actions"]]
    assert tools[:3] == ["recall_experiences", "inspect", "record_experience"]
