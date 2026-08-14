from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol

from open_vocab_grasping.agent.deepseek_client import DeepSeekPlanner, PlannerResponse
from open_vocab_grasping.agent.codegen import compile_and_execute_plan
from open_vocab_grasping.agent.local_planners import DeterministicPlanner, MockDeepSeekPlanner
from open_vocab_grasping.pipeline import run_open_vocab_grasp


AgentEventCallback = Callable[[str, dict[str, Any]], None]


class Planner(Protocol):
    def plan(self, instruction: str, available_targets: list[str]) -> PlannerResponse: ...


@dataclass(frozen=True)
class AgentExecution:
    success: bool
    output: Path
    plan: dict[str, Any]
    robot_result: dict[str, Any]
    provider: str
    model: str
    llm_latency_s: float
    usage: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": str(self.output),
            "plan": self.plan,
            "robot_result": self.robot_result,
            "provider": self.provider,
            "model": self.model,
            "llm_latency_s": self.llm_latency_s,
            "usage": self.usage,
        }


def build_planner(mode: str, config: dict[str, Any]) -> Planner:
    if mode == "deepseek":
        return DeepSeekPlanner(config.get("agent", {}))
    if mode == "mock":
        return MockDeepSeekPlanner()
    if mode == "deterministic":
        return DeterministicPlanner()
    raise ValueError(f"Unsupported agent mode {mode!r}")


def _emit(callback: AgentEventCallback | None, event: str, **fields: Any) -> None:
    if callback is not None:
        callback(event, fields)


def _write_audit(
    output: Path,
    instruction: str,
    response: PlannerResponse,
    execution: AgentExecution,
    generated_source: str,
    generated_trace: list[dict[str, Any]],
) -> None:
    # Authorization headers and the API key are never passed to this function.
    files = {
        "agent_request.json": {
            "instruction": instruction,
            "provider": response.provider,
            "request": response.request,
        },
        "agent_response.json": response.response,
        "agent_plan.json": response.plan,
        "agent_result.json": execution.to_dict(),
        "agent_generated_plan_trace.json": generated_trace,
    }
    for filename, payload in files.items():
        (output / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output / "agent_generated_plan.py").write_text(generated_source, encoding="utf-8")


def run_agent_instruction(
    config: dict[str, Any],
    instruction: str,
    seed: int,
    mode: str = "deepseek",
    event_callback: AgentEventCallback | None = None,
    planner: Planner | None = None,
) -> AgentExecution:
    available_targets = [str(target) for target in config["scene"]["object_names"]]
    selected_planner = planner or build_planner(mode, config)
    _emit(event_callback, "planning_started", mode=mode, instruction=instruction)
    response = selected_planner.plan(instruction, available_targets)
    configured_mode = (
        "open-vocab-graspnet"
        if str(config.get("grasp", {}).get("generator", "geometric_baseline")) == "graspnet"
        else "open-vocab-simple"
    )
    if response.plan["execution_mode"] != configured_mode:
        response = replace(response, plan={**response.plan, "execution_mode": configured_mode})
    generated_source, generated_trace = compile_and_execute_plan(response.plan)
    _emit(
        event_callback,
        "plan_validated",
        provider=response.provider,
        model=response.model,
        target=response.plan["target"],
        steps=response.plan["steps"],
        generated_python_validated=True,
    )
    _emit(event_callback, "robot_pipeline_started", target=response.plan["target"], seed=seed)
    output, robot_result = run_open_vocab_grasp(config, str(response.plan["target"]), seed)
    execution = AgentExecution(
        success=bool(robot_result["success"]),
        output=output,
        plan=response.plan,
        robot_result=robot_result,
        provider=response.provider,
        model=response.model,
        llm_latency_s=response.latency_s,
        usage=response.usage,
    )
    _write_audit(output, instruction, response, execution, generated_source, generated_trace)
    _emit(
        event_callback,
        "robot_pipeline_finished",
        success=execution.success,
        failure_reason=robot_result.get("failure_reason"),
        output=str(output),
        video=str(output / "demo.mp4"),
    )
    return execution
