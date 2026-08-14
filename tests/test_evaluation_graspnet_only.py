from pathlib import Path

from open_vocab_grasping.evaluation.runner import _episode_row


def test_graspnet_only_detection_metrics_are_not_applicable() -> None:
    row = _episode_row(
        "graspnet-only",
        "bottle",
        0,
        Path("outputs/example"),
        {
            "success": False,
            "failure_reason": "object_not_lifted",
            "candidate_count": 10,
            "accepted_count": 1,
            "ik_reachable": True,
        },
    )
    assert row["detection_success"] is None
    assert row["target_selection_correct"] is None
