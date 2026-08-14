from __future__ import annotations

import re
from typing import Any

from jsonschema import Draft202012Validator


ALLOWED_ACTIONS = frozenset({"pick"})
COMMAND_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "OpenVocabularyGraspCommand",
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "target", "destination"],
    "properties": {
        "action": {"type": "string", "enum": sorted(ALLOWED_ACTIONS)},
        "target": {
            "type": "string",
            "minLength": 1,
            "maxLength": 80,
            "pattern": r"^[^\u0000-\u001f\u007f]+$",
        },
        # The first release supports grasping only; placement is deliberately
        # unavailable until a safe destination planner is implemented.
        "destination": {"type": "null"},
    },
}
_VALIDATOR = Draft202012Validator(COMMAND_SCHEMA)


def validate_command(payload: dict[str, Any]) -> dict[str, str | None]:
    """Validate an external command before it can reach the action dispatcher."""
    errors = sorted(_VALIDATOR.iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        details = "; ".join(error.message for error in errors)
        raise ValueError(f"Command does not match the action schema: {details}")
    action = str(payload["action"])
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Action {action!r} is not whitelisted")
    return {
        "action": action,
        "target": str(payload["target"]),
        "destination": None,
    }


def parse_command(command: str) -> dict[str, str | None]:
    """Parse a deterministic English pick instruction into a validated payload."""
    match = re.fullmatch(r"\s*(pick|grasp)\s+(?:up\s+)?(?:the\s+)?(.+?)\s*", command, re.IGNORECASE)
    if not match:
        raise ValueError("Only whitelisted pick/grasp commands are supported")
    target = match.group(2).strip().lower()
    return validate_command({"action": "pick", "target": target, "destination": None})
