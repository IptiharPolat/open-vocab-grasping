from open_vocab_grasping.grasping.association import Detection
from open_vocab_grasping.perception.yolo_world import (
    filter_tabletop_fallback_geometry,
    prompt_consensus_detections,
)


def test_prompt_consensus_keeps_overlapping_distinct_prompt_boxes() -> None:
    detections = [
        Detection((10.0, 10.0, 30.0, 30.0), "box", 0.004),
        Detection((11.0, 10.0, 31.0, 30.0), "cube", 0.014),
        Detection((9.0, 11.0, 29.0, 31.0), "toy block", 0.007),
        Detection((100.0, 100.0, 150.0, 150.0), "cardboard box", 0.02),
    ]
    fused, votes = prompt_consensus_detections(
        detections, "box", minimum_prompt_votes=2, consensus_iou=0.55
    )
    assert len(fused) == 1
    assert fused[0].label == "box"
    expected = 3.0 * (0.004 * 0.014 * 0.007) ** (1.0 / 3.0)
    assert abs(fused[0].confidence - expected) < 1e-12
    assert votes == [3]


def test_prompt_consensus_does_not_count_duplicate_boxes_from_same_prompt() -> None:
    detections = [
        Detection((10.0, 10.0, 30.0, 30.0), "bowl", 0.01),
        Detection((11.0, 11.0, 31.0, 31.0), "bowl", 0.008),
    ]
    fused, votes = prompt_consensus_detections(
        detections, "bowl", minimum_prompt_votes=2, consensus_iou=0.55
    )
    assert fused == []
    assert votes == []


def test_tabletop_fallback_rejects_scene_sized_and_border_boxes() -> None:
    detections = [
        Detection((10.0, 10.0, 630.0, 470.0), "box", 0.02),
        Detection((280.0, 160.0, 335.0, 220.0), "box", 0.01),
        Detection((0.0, 400.0, 250.0, 480.0), "box", 0.005),
    ]
    kept, votes, rejected = filter_tabletop_fallback_geometry(
        detections, [3, 3, 2], (480, 640),
        maximum_area_fraction=0.08, reject_border_touching=True,
    )
    assert kept == [detections[1]]
    assert votes == [3]
    assert rejected == 2
