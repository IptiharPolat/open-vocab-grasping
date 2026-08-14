from __future__ import annotations

import json
from pathlib import Path

import pytest

from open_vocab_grasping.dashboard import (
    ACTION_RUN,
    DashboardView,
    build_dashboard_view,
    execute_dashboard_action,
)


def test_dashboard_view_uses_actual_artifacts(tmp_path: Path) -> None:
    (tmp_path / "rgb.png").write_bytes(b"image")
    (tmp_path / "demo.mp4").write_bytes(b"video")
    result = {
        "success": True,
        "state_history": ["RESET", "DETECT", "DONE"],
        "candidate_generator": "geometric_scene_baseline_not_graspnet",
        "truth_used_for_semantic_selection": False,
    }
    (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (tmp_path / "candidates.json").write_text(
        json.dumps(
            [
                {
                    "center": [0.4, 0.1, 0.08],
                    "accepted": False,
                    "final_score": 0.2,
                    "width": 0.05,
                    "rejection_reasons": ["outside_detection_box"],
                }
            ]
        ),
        encoding="utf-8",
    )
    view = build_dashboard_view(
        tmp_path,
        ACTION_RUN,
        {"action": "pick", "target": "mug", "destination": None},
        result,
    )
    assert isinstance(view, DashboardView)
    assert "抓取成功" in view.status
    assert view.rgb_path == str((tmp_path / "rgb.png").resolve())
    assert view.video_path == str((tmp_path / "demo.mp4").resolve())
    assert view.candidate_rows[0][-1] == "outside_detection_box"
    assert view.metrics["truth_used_for_semantic_selection"] is False
    assert "`DETECT`" in view.timeline


def test_dashboard_rejects_unlisted_actions() -> None:
    with pytest.raises(ValueError, match="Unsupported dashboard action"):
        execute_dashboard_action("run arbitrary python", "pick the mug", 0, "missing.yaml")
