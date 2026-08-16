from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from open_vocab_grasping.agent.deepseek_client import DeepSeekPlanner, PlannerResponse
from open_vocab_grasping.agent.local_planners import MockDeepSeekPlanner
from open_vocab_grasping.agent.runtime import run_agent_instruction
from open_vocab_grasping.agent.schemas import CANONICAL_PICK_STEPS, validate_grasp_plan
from open_vocab_grasping.config import load_config


def valid_plan(target: str = "mug") -> dict[str, Any]:
    return {
        "action": "pick",
        "target": target,
        "steps": list(CANONICAL_PICK_STEPS),
        "execution_mode": "open-vocab-simple",
        "explanation": "Pick the requested object.",
    }


def test_plan_schema_accepts_only_canonical_safe_plan() -> None:
    assert validate_grasp_plan(valid_plan(), ["mug", "bottle"])["target"] == "mug"
    unsafe = {**valid_plan(), "code": "import os; os.system('whoami')"}
    with pytest.raises(ValueError, match="schema"):
        validate_grasp_plan(unsafe, ["mug"])
    wrong_steps = {**valid_plan(), "steps": ["execute"]}
    with pytest.raises(ValueError, match="schema"):
        validate_grasp_plan(wrong_steps, ["mug"])
    with pytest.raises(ValueError, match="scene whitelist"):
        validate_grasp_plan(valid_plan("laptop"), ["mug"])


def test_mock_planner_is_explicit_and_supports_chinese() -> None:
    response = MockDeepSeekPlanner().plan("请帮我抓取桌面上的杯子", ["mug", "bottle"])
    assert response.provider == "mock"
    assert response.plan["target"] == "mug"
    assert response.response["mock"] is True


def test_deepseek_planner_validates_json_and_never_logs_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-never-log")
    captured: dict[str, Any] = {}

    def requester(
        url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        captured.update({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
        return {
            "model": "deepseek-v4-flash",
            "choices": [{"message": {"content": json.dumps(valid_plan())}}],
            "usage": {"total_tokens": 42},
        }

    response = DeepSeekPlanner({"model": "deepseek-v4-flash"}, requester).plan(
        "抓取杯子", ["mug", "bottle"]
    )
    assert response.plan["target"] == "mug"
    assert response.usage["total_tokens"] == 42
    assert "secret-never-log" not in json.dumps(response.request)
    assert captured["headers"]["Authorization"] == "Bearer secret-never-log"
    assert captured["payload"]["response_format"] == {"type": "json_object"}


def test_deepseek_prompt_uses_configured_graspnet_execution_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-never-log")
    captured: dict[str, Any] = {}
    plan = {**valid_plan(), "execution_mode": "open-vocab-graspnet"}

    def requester(
        url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        captured["payload"] = payload
        return {
            "model": "deepseek-v4-flash",
            "choices": [{"message": {"content": json.dumps(plan)}}],
        }

    response = DeepSeekPlanner(
        {"execution_mode": "open-vocab-graspnet"}, requester
    ).plan("抓取杯子", ["mug"])

    assert response.plan["execution_mode"] == "open-vocab-graspnet"
    assert "execution_mode must be open-vocab-graspnet" in captured["payload"]["messages"][0]["content"]


def test_deepseek_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        DeepSeekPlanner({}).plan("抓取杯子", ["mug"])


def test_agent_runtime_writes_redacted_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    robot_result = {"success": True, "failure_reason": None, "state_history": ["DONE"]}
    def fake_robot_pipeline(
        config: dict[str, Any], target: str, seed: int, *, stage_controller: Any
    ) -> tuple[Path, dict[str, Any]]:
        for step in CANONICAL_PICK_STEPS:
            stage_controller.begin_stage(step)
            stage_controller.complete_stage(step)
        return output, robot_result

    monkeypatch.setattr(
        "open_vocab_grasping.agent.runtime.run_open_vocab_grasp", fake_robot_pipeline
    )

    class FakePlanner:
        def plan(self, instruction: str, available_targets: list[str]) -> PlannerResponse:
            plan = validate_grasp_plan(valid_plan(), available_targets)
            return PlannerResponse(
                "deepseek",
                "deepseek-v4-flash",
                plan,
                {"model": "deepseek-v4-flash"},
                {"choices": []},
                0.25,
                {"total_tokens": 10},
            )

    config = {
        "scene": {"object_names": ["mug", "bottle"]},
        "grasp": {"generator": "graspnet"},
    }
    execution = run_agent_instruction(config, "请抓取杯子", 0, planner=FakePlanner())
    assert execution.success
    assert execution.plan["execution_mode"] == "open-vocab-graspnet"
    assert (output / "agent_request.json").is_file()
    assert (output / "agent_response.json").is_file()
    assert (output / "agent_plan.json").is_file()
    assert (output / "agent_result.json").is_file()
    assert (output / "agent_generated_plan.py").is_file()
    assert (output / "agent_generated_plan_trace.json").is_file()
    assert execution.robot_result["generated_plan_execution"] == {
        "python_executed": True,
        "controller": "SafeRobotController",
        "pipeline_stage_gating": True,
        "completed_steps": list(CANONICAL_PICK_STEPS),
    }
    trace = json.loads((output / "agent_generated_plan_trace.json").read_text())
    assert [entry["step"] for entry in trace] == list(CANONICAL_PICK_STEPS)
    assert all(entry["status"] == "completed" for entry in trace)
    audit_text = "".join(path.read_text() for path in output.glob("agent_*.json"))
    assert "Authorization" not in audit_text
    assert "DEEPSEEK_API_KEY" not in audit_text


def test_agent_gui_config_keeps_agent_and_live_visualization() -> None:
    config = load_config("configs/agent_gui.yaml")
    assert config["simulation"]["gui"] is True
    assert config["simulation"]["realtime_pacing"] is True
    assert config["simulation"]["hold_after_run_s"] == 10.0
    assert config["agent"]["model"] == "deepseek-v4-flash"


def test_agent_graspnet_config_requests_matching_execution_mode() -> None:
    config = load_config("configs/agent_graspnet_gui.yaml")
    assert config["grasp"]["generator"] == "graspnet"
    assert config["agent"]["execution_mode"] == "open-vocab-graspnet"
