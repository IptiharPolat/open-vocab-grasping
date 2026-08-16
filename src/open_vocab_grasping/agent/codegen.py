"""Compile validated LLM plans to a tiny auditable Python planning DSL."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Protocol

from open_vocab_grasping.agent.schemas import CANONICAL_PICK_STEPS


_METHODS = (
    "observe",
    "detect",
    "generate_grasps",
    "select_grasp",
    "execute",
    "evaluate",
)


class PlanController(Protocol):
    def observe(self) -> None: ...
    def detect(self, target: str) -> None: ...
    def generate_grasps(self) -> None: ...
    def select_grasp(self) -> None: ...
    def execute(self) -> None: ...
    def evaluate(self) -> None: ...


def render_plan_python(plan: dict[str, Any]) -> str:
    """Render only a previously schema-validated canonical grasp plan."""
    if list(plan.get("steps", [])) != list(CANONICAL_PICK_STEPS):
        raise ValueError("Only the canonical grasp steps can be compiled")
    target = str(plan["target"])
    return (
        "# Generated from a schema-validated LLM plan; no arbitrary model code.\n"
        "def execute_plan(controller):\n"
        "    controller.observe()\n"
        f"    controller.detect({target!r})\n"
        "    controller.generate_grasps()\n"
        "    controller.select_grasp()\n"
        "    controller.execute()\n"
        "    controller.evaluate()\n"
    )


def validate_plan_python(source: str, expected_target: str) -> ast.Module:
    """Require one exact function and six allowlisted controller calls."""
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Generated plan Python is invalid: {exc}") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Generated plan must contain exactly one function")
    function = tree.body[0]
    if function.name != "execute_plan" or function.decorator_list:
        raise ValueError("Generated plan function must be undecorated execute_plan")
    arguments = function.args
    if (
        [argument.arg for argument in arguments.args] != ["controller"]
        or arguments.vararg is not None
        or arguments.kwarg is not None
        or arguments.kwonlyargs
        or arguments.defaults
    ):
        raise ValueError("execute_plan must accept only one controller argument")
    if len(function.body) != len(_METHODS):
        raise ValueError("Generated plan must contain exactly six controller calls")

    for index, (statement, method) in enumerate(zip(function.body, _METHODS)):
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise ValueError(f"Plan statement {index} is not an allowlisted call")
        call = statement.value
        if call.keywords or not isinstance(call.func, ast.Attribute):
            raise ValueError(f"Plan statement {index} has unsupported call syntax")
        if (
            not isinstance(call.func.value, ast.Name)
            or call.func.value.id != "controller"
            or call.func.attr != method
        ):
            raise ValueError(f"Plan statement {index} must call controller.{method}")
        if method == "detect":
            if (
                len(call.args) != 1
                or not isinstance(call.args[0], ast.Constant)
                or call.args[0].value != expected_target
            ):
                raise ValueError("controller.detect target does not match the validated plan")
        elif call.args:
            raise ValueError(f"controller.{method} does not accept arguments")
    return tree


def execute_plan_python(source: str, expected_target: str, controller: PlanController) -> None:
    """Execute validated generated Python only against an explicit safe controller."""
    tree = validate_plan_python(source, expected_target)
    namespace: dict[str, Any] = {}
    exec(compile(tree, "<validated-grasp-plan>", "exec"), {"__builtins__": {}}, namespace)
    namespace["execute_plan"](controller)


@dataclass
class _PlanRecorder:
    expected_target: str
    trace: list[dict[str, Any]] = field(default_factory=list)

    def _record(self, step: str, **fields: Any) -> None:
        self.trace.append({"step": step, **fields})

    def observe(self) -> None:
        self._record("observe")

    def detect(self, target: str) -> None:
        if target != self.expected_target:
            raise ValueError("Generated-code target changed after validation")
        self._record("detect", target=target)

    def generate_grasps(self) -> None:
        self._record("generate_grasps")

    def select_grasp(self) -> None:
        self._record("select_grasp")

    def execute(self) -> None:
        self._record("execute")

    def evaluate(self) -> None:
        self._record("evaluate")


def compile_and_execute_plan(plan: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Compile, AST-validate and execute the tiny DSL with no Python builtins."""
    source = render_plan_python(plan)
    recorder = _PlanRecorder(str(plan["target"]))
    execute_plan_python(source, str(plan["target"]), recorder)
    executed_steps = [str(entry["step"]) for entry in recorder.trace]
    if executed_steps != list(CANONICAL_PICK_STEPS):
        raise RuntimeError("Generated Python execution trace differs from the validated plan")
    return source, recorder.trace
