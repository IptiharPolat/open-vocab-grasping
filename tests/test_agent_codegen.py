import pytest

from open_vocab_grasping.agent.codegen import (
    compile_and_execute_plan,
    validate_plan_python,
)
from open_vocab_grasping.agent.schemas import CANONICAL_PICK_STEPS


def _plan() -> dict[str, object]:
    return {
        "action": "pick",
        "target": "mug",
        "steps": list(CANONICAL_PICK_STEPS),
        "execution_mode": "open-vocab-simple",
        "explanation": "Pick the mug.",
    }


def test_validated_plan_compiles_and_executes_exact_dsl_trace() -> None:
    source, trace = compile_and_execute_plan(_plan())
    assert "controller.detect('mug')" in source
    assert [entry["step"] for entry in trace] == list(CANONICAL_PICK_STEPS)
    assert trace[1]["target"] == "mug"


@pytest.mark.parametrize(
    "unsafe",
    [
        "import os\n",
        "def execute_plan(controller):\n    __import__('os').system('id')\n",
        (
            "def execute_plan(controller):\n"
            "    controller.observe()\n"
            "    controller.detect('bottle')\n"
            "    controller.generate_grasps()\n"
            "    controller.select_grasp()\n"
            "    controller.execute()\n"
            "    controller.evaluate()\n"
        ),
    ],
)
def test_generated_plan_ast_rejects_imports_calls_and_target_changes(unsafe: str) -> None:
    with pytest.raises(ValueError):
        validate_plan_python(unsafe, "mug")
