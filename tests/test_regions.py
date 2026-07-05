from __future__ import annotations

import pytest

from brainregion.core.regions import RegionDefinition, list_regions, load_region, route_regions


def test_list_and_load_builtin_regions():
    names = list_regions()
    assert "planning" in names
    assert "performance" in names
    assert "unity_ecs" in names

    region = load_region("planning")
    assert region.id == "planning"
    assert "plan" in region.triggers


def test_route_regions_selects_planning_and_review():
    result = route_regions(
        goal="Create a roadmap and task plan, then review the design before implementation.",
        top_k=3,
    )
    selected = [item["id"] for item in result["selected"]]
    assert selected[0] == "planning"
    assert "review" in selected
    assert result["trace"]["strategy"] == "deterministic_keyword_v1"
    assert result["trace"]["input"]["file_contents_used"] is False


def test_route_regions_supports_chinese_triggers():
    result = route_regions(problem="这个功能性能很慢，需要优化内存分配和延迟。", top_k=2)
    selected = [item["id"] for item in result["selected"]]
    assert selected[0] == "performance"
    assert result["selected"][0]["matched_triggers"]


def test_route_regions_empty_input_returns_no_selection():
    result = route_regions()
    assert result["selected"] == []
    assert result["candidates"] == []
    assert result["trace"]["no_match_reason"] == "empty_input"


def test_route_regions_uses_file_paths_as_weak_signal_only():
    result = route_regions(
        files={
            "Assets/Scripts/FlowFieldSystem.cs": "this content mentions security but must be ignored",
        },
        top_k=3,
    )
    selected = [item["id"] for item in result["selected"]]
    assert "unity_ecs" in selected
    assert "security" not in selected
    assert result["trace"]["input"]["file_paths"] == 1
    assert result["trace"]["input"]["file_contents_used"] is False


def test_memory_allocation_routes_to_performance_not_memory_region():
    result = route_regions(problem="Optimize memory allocations and latency in the hot path.", top_k=3)
    selected = [item["id"] for item in result["selected"]]
    assert "performance" in selected
    assert "memory" not in selected


def test_project_understanding_routes_to_memory_region():
    result = route_regions(
        problem="整理 BrainRegion 项目理解、记忆卡片、seed memory 和上下文压缩经验。",
        top_k=3,
    )
    selected = [item["id"] for item in result["selected"]]
    assert selected[0] == "memory"
    assert "performance" not in selected
    assert result["selected"][0]["score"] >= 6


def test_route_regions_negative_triggers_can_filter_matches():
    regions = [
        RegionDefinition(
            id="research",
            name="Research",
            triggers=["research", "docs"],
            negative_triggers=["do not browse"],
        )
    ]
    result = route_regions(goal="Research docs but do not browse the web.", regions=regions, min_score=2)
    assert result["selected"] == []
    assert result["candidates"][0]["score"] < 2
    assert result["candidates"][0]["negative_triggers"]
    assert result["trace"]["no_match_reason"] == "below_min_score"


def test_route_regions_validates_limits():
    with pytest.raises(ValueError, match="top_k"):
        route_regions(goal="plan", top_k=0)
    with pytest.raises(ValueError, match="min_score"):
        route_regions(goal="plan", min_score=-1)


def test_server_route_regions_tool():
    from brainregion.server import list_regions as server_list_regions
    from brainregion.server import route_regions as server_route_regions

    listed = server_list_regions()
    assert any(region["id"] == "debugging" for region in listed["regions"])

    result = server_route_regions(problem="tests are failing with an exception", top_k=1)
    assert result["selected"][0]["id"] == "debugging"
