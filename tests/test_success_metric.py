import numpy as np

from open_vocab_grasping.evaluation.metrics import box_iou_xyxy, detection_metrics, grasp_success
from open_vocab_grasping.grasping.association import Detection


def test_success_requires_lift_and_tool_proximity() -> None:
    success, _ = grasp_success(np.array([0.5, 0.0, 0.03]), np.array([0.5, 0.0, 0.13]),
                               np.array([0.5, 0.0, 0.20]), 0.06)
    assert success
    success, _ = grasp_success(np.array([0.5, 0.0, 0.03]), np.array([0.5, 0.0, 0.04]),
                               np.array([0.5, 0.0, 0.20]), 0.06)
    assert not success


def test_detection_metrics_use_real_iou_threshold() -> None:
    truth = Detection((10.0, 10.0, 30.0, 30.0), "mug", 1.0)
    predictions = [
        Detection((12.0, 12.0, 28.0, 28.0), "mug", 0.7),
        Detection((50.0, 50.0, 60.0, 60.0), "mug", 0.9),
    ]
    assert box_iou_xyxy(truth.bbox_xyxy, predictions[0].bbox_xyxy) == 0.64
    metrics = detection_metrics(predictions, truth, 0.5)
    assert metrics["detection_success"]
    assert not metrics["target_selection_correct"]
