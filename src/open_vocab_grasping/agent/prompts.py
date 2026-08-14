from __future__ import annotations

import json

from open_vocab_grasping.agent.schemas import CANONICAL_PICK_STEPS


def build_system_prompt(
    available_targets: list[str], execution_mode: str = "open-vocab-simple"
) -> str:
    if execution_mode not in {"open-vocab-simple", "open-vocab-graspnet"}:
        raise ValueError(f"Unsupported agent execution mode: {execution_mode}")
    targets = ", ".join(sorted(str(target) for target in available_targets))
    example = {
        "action": "pick",
        "target": "mug",
        "steps": list(CANONICAL_PICK_STEPS),
        "execution_mode": execution_mode,
        "explanation": "The user asked to pick the mug.",
    }
    return (
        "You are the high-level task planner for a PyBullet robot grasping system. "
        "Interpret the user's Chinese or English command. Return exactly one JSON object and no "
        "markdown. Only a pick action is permitted. The target must be one exact English name "
        f"from this scene whitelist: [{targets}]. The steps must be exactly "
        f"{json.dumps(list(CANONICAL_PICK_STEPS))}. The execution_mode must be "
        f"{execution_mode}. Never output Python, shell commands, imports, file operations, or "
        "additional fields. JSON example: "
        f"{json.dumps(example, ensure_ascii=False)}"
    )
