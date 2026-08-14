from __future__ import annotations

import re
from typing import Any

from open_vocab_grasping.agent.deepseek_client import PlannerResponse
from open_vocab_grasping.agent.schemas import CANONICAL_PICK_STEPS, validate_grasp_plan
from open_vocab_grasping.nlp import parse_command


def _plan(target: str, explanation: str, available_targets: list[str]) -> dict[str, Any]:
    return validate_grasp_plan(
        {
            "action": "pick",
            "target": target,
            "steps": list(CANONICAL_PICK_STEPS),
            "execution_mode": "open-vocab-simple",
            "explanation": explanation,
        },
        available_targets,
    )


class DeterministicPlanner:
    def plan(self, instruction: str, available_targets: list[str]) -> PlannerResponse:
        parsed = parse_command(instruction)
        plan = _plan(str(parsed["target"]), "Parsed by deterministic English grammar.", available_targets)
        return PlannerResponse("deterministic", "none", plan, {}, {}, 0.0, {})


class MockDeepSeekPlanner:
    """Offline test double. Its results must never be reported as real API output."""

    _ALIASES = {
        "杯子": "mug",
        "马克杯": "mug",
        "瓶子": "bottle",
        "碗": "bowl",
        "盒子": "box",
        "mug": "mug",
        "bottle": "bottle",
        "bowl": "bowl",
        "box": "box",
    }

    def plan(self, instruction: str, available_targets: list[str]) -> PlannerResponse:
        lowered = instruction.lower()
        matches = [target for alias, target in self._ALIASES.items() if alias in lowered]
        unique = sorted(set(matches))
        if len(unique) != 1 or not re.search(r"抓|拿|pick|grasp", lowered):
            raise ValueError("Mock planner supports one explicit pick target only")
        plan = _plan(unique[0], "Offline mock planner response; no API was called.", available_targets)
        response = {"mock": True, "model": "mock-deepseek", "plan": plan}
        return PlannerResponse("mock", "mock-deepseek", plan, {}, response, 0.0, {})
