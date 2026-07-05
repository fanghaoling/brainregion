from __future__ import annotations

import pytest

from brainregion import server
from brainregion.memory import store as memory_store


def test_mcp_record_experience_normalizes_registered_region():
    rec = server.record_experience(summary="seed", triggers=["k"], region="Unity-ECS")

    assert rec["ok"] is True
    event = next(e for e in memory_store.list_experiences() if e.id == rec["id"])
    assert event.region == "unity_ecs"


def test_mcp_record_experience_keeps_global_region():
    rec = server.record_experience(summary="global", triggers=["k"], region="  ")

    assert rec["ok"] is True
    event = next(e for e in memory_store.list_experiences() if e.id == rec["id"])
    assert event.region == ""


def test_mcp_record_experience_rejects_unknown_region():
    with pytest.raises(ValueError, match="unknown experience region"):
        server.record_experience(summary="orphan", triggers=["k"], region="tooling")

    assert memory_store.list_experiences() == []
