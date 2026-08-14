from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator


CANONICAL_PICK_STEPS = (
    "observe",
    "detect",
    "generate_grasps",
    "select_grasp",
    "execute",
    "evaluate",
)

GRASP_PLAN_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ValidatedGraspPlan",
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "target", "steps", "execution_mode", "explanation"],
    "properties": {
        "action": {"type": "string", "const": "pick"},
        "target": {
            "type": "string",
            "minLength": 1,
            "maxLength": 40,
            "pattern": r"^[a-z][a-z0-9 _-]*$",
        },
        "steps": {
            "type": "array",
            "const": list(CANONICAL_PICK_STEPS),
        },
        "execution_mode": {
            "type": "string",
            "enum": ["open-vocab-simple", "open-vocab-graspnet"],
        },
        "explanation": {"type": "string", "minLength": 1, "maxLength": 300},
    },
}

_VALIDATOR = Draft202012Validator(GRASP_PLAN_SCHEMA)


def validate_grasp_plan(plan: dict[str, Any], available_targets: list[str]) -> dict[str, Any]:
    """Reject unapproved actions, fields, steps and targets before dispatch."""
    errors = sorted(_VALIDATOR.iter_errors(plan), key=lambda error: list(error.path))
    if errors:
        details = "; ".join(error.message for error in errors)
        raise ValueError(f"DeepSeek plan failed schema validation: {details}")
    normalized_targets = {str(target).strip().lower() for target in available_targets}
    target = str(plan["target"]).strip().lower()
    if target not in normalized_targets:
        raise ValueError(
            f"DeepSeek target {target!r} is not in the current scene whitelist: "
            f"{sorted(normalized_targets)}"
        )
    return {
        "action": "pick",
        "target": target,
        "steps": list(CANONICAL_PICK_STEPS),
        "execution_mode": str(plan["execution_mode"]),
        "explanation": str(plan["explanation"]).strip(),
    }
