from open_vocab_grasping.evaluation.report import summarize_episodes, summary_markdown


def test_summary_rates_and_failures_are_computed_from_rows() -> None:
    rows = [
        {
            "mode": "open-vocab-simple", "target": "mug", "success": True,
            "failure_reason": None, "target_selection_correct": True,
            "detection_success": True, "candidate_generation_success": True,
            "ik_reachable": True, "grasp_success": True, "end_to_end_success": True,
            "detection_s": 0.1, "grasp_generation_s": 0.2, "planning_s": 0.3,
            "execution_s": 0.4, "total_s": 1.0,
        },
        {
            "mode": "open-vocab-simple", "target": "mug", "success": False,
            "failure_reason": "detection_failed", "target_selection_correct": False,
            "detection_success": False, "candidate_generation_success": True,
            "ik_reachable": False, "grasp_success": False, "end_to_end_success": False,
            "detection_s": 0.2, "grasp_generation_s": 0.2, "planning_s": 0.1,
            "execution_s": 0.0, "total_s": 0.5,
        },
    ]
    summary = summarize_episodes(rows, {"open-vocab-graspnet": "unavailable"})
    mode = summary["modes"]["open-vocab-simple"]
    assert mode["end_to_end_success_rate"] == 0.5
    assert mode["mean_total_s"] == 0.75
    assert mode["failure_distribution"] == {"detection_failed": 1}
    markdown = summary_markdown(summary)
    assert "50.0%" in markdown
    assert "open-vocab-graspnet" in markdown
