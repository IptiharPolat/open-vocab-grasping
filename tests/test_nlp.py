from __future__ import annotations

import pytest

from open_vocab_grasping.cli import main
from open_vocab_grasping.nlp import parse_command, validate_command


@pytest.mark.parametrize(
    ("text", "target"),
    [
        ("pick the red mug", "red mug"),
        ("pick up the bowl", "bowl"),
        ("GRASP bottle", "bottle"),
    ],
)
def test_parse_whitelisted_pick_commands(text: str, target: str) -> None:
    assert parse_command(text) == {
        "action": "pick",
        "target": target,
        "destination": None,
    }


@pytest.mark.parametrize("text", ["drop the mug", "", "pick "])
def test_parse_rejects_non_whitelisted_commands(text: str) -> None:
    with pytest.raises(ValueError):
        parse_command(text)


def test_schema_rejects_extra_fields_and_unapproved_actions() -> None:
    with pytest.raises(ValueError, match="schema"):
        validate_command(
            {"action": "execute_python", "target": "mug", "destination": None, "code": "x"}
        )


def test_cli_parse_outputs_valid_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["parse", "--instruction", "pick the red mug"]) == 0
    output = capsys.readouterr().out
    assert '"target": "red mug"' in output
