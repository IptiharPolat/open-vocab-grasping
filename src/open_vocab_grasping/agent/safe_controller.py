"""Synchronize validated generated Python with real robot pipeline stages."""

from __future__ import annotations

import threading
from collections.abc import Callable
from time import perf_counter
from typing import Any

from open_vocab_grasping.agent.schemas import CANONICAL_PICK_STEPS


class SafeRobotController:
    """One-shot capability gate used by the generated grasp-plan Python.

    The generated program runs in a restricted worker thread. Each allowlisted
    controller call blocks until the real pipeline begins and completes the same
    stage. Consequently the pipeline cannot advance unless the generated Python
    authorizes the exact target and canonical stage order.
    """

    def __init__(self, expected_target: str) -> None:
        self.expected_target = expected_target
        self._condition = threading.Condition()
        self._requested_index = 0
        self._pipeline_index = 0
        self._pending_step: str | None = None
        self._active_step: str | None = None
        self._records: dict[str, dict[str, Any]] = {}
        self._trace: list[dict[str, Any]] = []
        self._program_finished = False
        self._program_error: Exception | None = None
        self._abort_reason: str | None = None
        self._thread: threading.Thread | None = None
        self._started_at = perf_counter()

    def start(self, program: Callable[[], None]) -> None:
        """Start the already AST-validated generated program exactly once."""
        with self._condition:
            if self._thread is not None:
                raise RuntimeError("Generated grasp program has already been started")

        def run() -> None:
            try:
                program()
            except Exception as exc:  # surfaced to the pipeline thread with context
                with self._condition:
                    self._program_error = exc
                    self._condition.notify_all()
            finally:
                with self._condition:
                    self._program_finished = True
                    self._condition.notify_all()

        self._thread = threading.Thread(target=run, name="validated-grasp-plan", daemon=True)
        self._thread.start()

    def _elapsed(self) -> float:
        return perf_counter() - self._started_at

    def _authorize(self, step: str, *, target: str | None = None) -> None:
        with self._condition:
            if self._abort_reason is not None:
                raise RuntimeError(f"Robot pipeline aborted: {self._abort_reason}")
            if self._requested_index >= len(CANONICAL_PICK_STEPS):
                raise RuntimeError(f"Generated program requested unexpected extra stage {step!r}")
            expected = CANONICAL_PICK_STEPS[self._requested_index]
            if step != expected:
                raise RuntimeError(
                    f"Generated program requested stage {step!r}; expected {expected!r}"
                )
            if step == "detect" and target != self.expected_target:
                raise ValueError(
                    f"Generated-code target {target!r} differs from validated target "
                    f"{self.expected_target!r}"
                )
            if self._pending_step is not None:
                raise RuntimeError(
                    f"Generated program requested {step!r} while {self._pending_step!r} is pending"
                )
            record: dict[str, Any] = {
                "step": step,
                "status": "authorized",
                "requested_at_s": self._elapsed(),
            }
            if target is not None:
                record["target"] = target
            self._records[step] = record
            self._pending_step = step
            self._requested_index += 1
            self._condition.notify_all()
            while (
                self._pending_step == step
                and self._abort_reason is None
                and self._program_error is None
            ):
                self._condition.wait()
            if self._abort_reason is not None:
                raise RuntimeError(f"Robot pipeline aborted: {self._abort_reason}")
            if self._program_error is not None:
                raise RuntimeError("Generated grasp program failed") from self._program_error

    def observe(self) -> None:
        self._authorize("observe")

    def detect(self, target: str) -> None:
        self._authorize("detect", target=target)

    def generate_grasps(self) -> None:
        self._authorize("generate_grasps")

    def select_grasp(self) -> None:
        self._authorize("select_grasp")

    def execute(self) -> None:
        self._authorize("execute")

    def evaluate(self) -> None:
        self._authorize("evaluate")

    def begin_stage(self, step: str) -> None:
        """Block the real pipeline until generated Python authorizes ``step``."""
        with self._condition:
            if self._pipeline_index >= len(CANONICAL_PICK_STEPS):
                raise RuntimeError(f"Robot pipeline attempted unexpected extra stage {step!r}")
            expected = CANONICAL_PICK_STEPS[self._pipeline_index]
            if step != expected:
                raise RuntimeError(f"Robot pipeline entered {step!r}; expected {expected!r}")
            while (
                self._pending_step is None
                and self._program_error is None
                and self._abort_reason is None
                and not self._program_finished
            ):
                self._condition.wait()
            if self._program_error is not None:
                raise RuntimeError("Generated grasp program failed before pipeline dispatch") from self._program_error
            if self._abort_reason is not None:
                raise RuntimeError(f"Robot pipeline aborted: {self._abort_reason}")
            if self._pending_step != step:
                raise RuntimeError(
                    f"Generated program authorized {self._pending_step!r}; pipeline requires {step!r}"
                )
            self._active_step = step
            self._records[step]["started_at_s"] = self._elapsed()
            self._records[step]["status"] = "running"

    def complete_stage(self, step: str) -> None:
        """Acknowledge that the real pipeline completed the authorized stage."""
        with self._condition:
            if self._active_step != step or self._pending_step != step:
                raise RuntimeError(f"Cannot complete inactive robot pipeline stage {step!r}")
            record = self._records[step]
            record["completed_at_s"] = self._elapsed()
            record["status"] = "completed"
            self._trace.append(dict(record))
            self._active_step = None
            self._pending_step = None
            self._pipeline_index += 1
            self._condition.notify_all()

    def abort(self, reason: str) -> None:
        with self._condition:
            self._abort_reason = reason
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def finish(self) -> list[dict[str, Any]]:
        """Verify program termination and exact stage completion, then return the audit trace."""
        if self._thread is None:
            raise RuntimeError("Generated grasp program was never started")
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            self.abort("generated_program_did_not_finish")
            raise RuntimeError("Generated grasp program did not finish after pipeline evaluation")
        if self._program_error is not None:
            raise RuntimeError("Generated grasp program failed") from self._program_error
        completed = [str(record["step"]) for record in self._trace]
        if completed != list(CANONICAL_PICK_STEPS):
            raise RuntimeError(
                f"Generated program completed stages {completed}; expected {list(CANONICAL_PICK_STEPS)}"
            )
        return [dict(record) for record in self._trace]
