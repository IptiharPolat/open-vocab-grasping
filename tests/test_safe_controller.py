from __future__ import annotations

from open_vocab_grasping.agent.codegen import execute_plan_python, render_plan_python
from open_vocab_grasping.agent.safe_controller import SafeRobotController
from open_vocab_grasping.agent.schemas import CANONICAL_PICK_STEPS


def _plan() -> dict[str, object]:
    return {
        "action": "pick",
        "target": "bottle",
        "steps": list(CANONICAL_PICK_STEPS),
        "execution_mode": "open-vocab-graspnet",
        "explanation": "Pick the bottle.",
    }


def test_generated_python_synchronously_gates_real_pipeline_stages() -> None:
    source = render_plan_python(_plan())
    controller = SafeRobotController("bottle")
    controller.start(lambda: execute_plan_python(source, "bottle", controller))

    actual_stages: list[str] = []
    for step in CANONICAL_PICK_STEPS:
        controller.begin_stage(step)
        actual_stages.append(step)
        controller.complete_stage(step)

    trace = controller.finish()
    assert actual_stages == list(CANONICAL_PICK_STEPS)
    assert [entry["step"] for entry in trace] == actual_stages
    assert all(entry["status"] == "completed" for entry in trace)
    assert trace[1]["target"] == "bottle"
