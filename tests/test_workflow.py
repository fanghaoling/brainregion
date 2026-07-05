from __future__ import annotations

from brainregion.core.workflow import suggest_workflow


def _tools(result: dict) -> list[str]:
    return [action["tool"] for action in result["next_actions"]]


def test_suggest_workflow_recommends_planner_for_planning_region():
    result = suggest_workflow(
        goal="Create a roadmap and task plan with milestones and acceptance criteria.",
        top_k=3,
    )

    assert result["selected_regions"][0]["id"] == "planning"
    assert result["next_actions"][0]["tool"] == "plan_task"
    assert result["next_actions"][0]["requires_user_approval"] is True
    assert result["next_actions"][0]["suggested_args"]["goal"].startswith("Create a roadmap")
    assert result["trace"]["auto_execute"] is False
    assert result["trace"]["models_called"] is False


def test_suggest_workflow_recommends_consult_modes_for_debugging_and_performance():
    result = suggest_workflow(
        problem="Tests are failing with an exception and the hot path has slow allocation latency.",
        top_k=4,
    )

    consult_actions = [action for action in result["next_actions"] if action["tool"] == "consult_problem"]
    modes = {action["suggested_args"].get("mode") for action in consult_actions}
    assert "debugging" in modes
    assert "performance" in modes
    assert all(action["requires_user_approval"] is True for action in consult_actions)


def test_suggest_workflow_recommends_review_code_when_files_are_present():
    result = suggest_workflow(
        goal="Please review this code change before merge.",
        files={"src/service.py": "print('hello')"},
        top_k=3,
    )

    assert "review_code" in _tools(result)
    action = next(action for action in result["next_actions"] if action["tool"] == "review_code")
    assert action["suggested_args"]["files"] == {"src/service.py": "print('hello')"}
    assert "extra_context" in action["suggested_args"]


def test_suggest_workflow_recommends_review_document_without_files():
    result = suggest_workflow(goal="Review this architecture decision record.", top_k=3)

    assert "review_document" in _tools(result)
    action = next(action for action in result["next_actions"] if action["tool"] == "review_document")
    assert action["suggested_args"]["document_type"] == "markdown"
    assert "architecture decision record" in action["suggested_args"]["content"]


def test_suggest_workflow_recommends_memory_actions():
    result = suggest_workflow(
        problem="整理 BrainRegion 项目理解、记忆卡片和上下文压缩经验。",
        goal="决定下一步是否先召回已有项目理解，再写入新的 seed memory",
        top_k=4,
    )

    selected = [region["id"] for region in result["selected_regions"]]
    assert selected[0] == "memory"

    tools = _tools(result)
    assert tools[:3] == ["recall_experiences", "inspect", "record_experience"]
    recall = result["next_actions"][0]
    assert recall["suggested_args"]["top_k"] == 5
    assert "项目理解" in recall["suggested_args"]["text"]
    inspect = result["next_actions"][1]
    assert inspect["suggested_args"] == {"view": "memory", "memory_preview_k": 8}
    record = result["next_actions"][2]
    assert record["suggested_args"]["region"] == "memory"
    assert record["suggested_args"]["status"] == "pending"
    assert not any(action["source_regions"] == ["performance"] for action in result["next_actions"])
    assert all(action["requires_user_approval"] is True for action in result["next_actions"])


def test_suggest_workflow_recommends_unity_ecs_consultant():
    result = suggest_workflow(
        problem="Unity ECS FlowField system architecture needs DOTS guidance.",
        top_k=3,
    )

    action = next(action for action in result["next_actions"] if action["tool"] == "consult_problem")
    assert action["suggested_args"]["consultants"] == ["unity_ecs"]
    assert action["source_regions"] == ["unity_ecs"]


def test_suggest_workflow_keeps_unity_ecs_consultant_for_performance_problem():
    result = suggest_workflow(
        problem="Unity ECS FlowField has slow allocation latency in the Burst job system.",
        top_k=4,
    )

    action = next(
        action
        for action in result["next_actions"]
        if action["tool"] == "consult_problem" and action["suggested_args"].get("mode") == "performance"
    )
    assert action["suggested_args"]["consultants"] == ["performance", "unity_ecs", "critic"]
    assert action["source_regions"] == ["performance", "unity_ecs"]


def test_suggest_workflow_empty_input_has_no_actions():
    result = suggest_workflow()

    assert result["selected_regions"] == []
    assert result["next_actions"] == []
    assert result["trace"]["routing"]["no_match_reason"] == "empty_input"


def test_server_suggest_workflow_tool():
    from brainregion.server import suggest_workflow as server_suggest_workflow

    result = server_suggest_workflow(problem="debug a flaky failure", top_k=2)
    assert result["next_actions"][0]["tool"] == "consult_problem"
    assert result["next_actions"][0]["suggested_args"]["mode"] == "debugging"
