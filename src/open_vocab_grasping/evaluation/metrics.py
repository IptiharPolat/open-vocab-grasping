from __future__ import annotations

import numpy as np

from open_vocab_grasping.grasping.association import Detection


def box_iou_xyxy(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def detection_metrics(
    detections: list[Detection], truth: Detection, iou_threshold: float
) -> dict[str, float | int | bool | None]:
    overlaps = [box_iou_xyxy(detection.bbox_xyxy, truth.bbox_xyxy) for detection in detections]
    best_index = int(np.argmax(overlaps)) if overlaps else None
    best_iou = float(overlaps[best_index]) if best_index is not None else 0.0
    selected_index = int(np.argmax([detection.confidence for detection in detections])) if detections else None
    selected_iou = float(overlaps[selected_index]) if selected_index is not None else 0.0
    return {
        "prediction_count": len(detections),
        "detection_success": bool(best_iou >= iou_threshold),
        "best_iou": best_iou,
        "best_match_index": best_index,
        "selected_iou": selected_iou,
        "target_selection_correct": bool(selected_iou >= iou_threshold),
        "iou_threshold": float(iou_threshold),
    }


def grasp_success(
    initial_object_position: np.ndarray,
    final_object_position: np.ndarray,
    final_tool_position: np.ndarray,
    minimum_lift_m: float,
    maximum_tool_distance_m: float = 0.16,
) -> tuple[bool, dict[str, float | bool]]:
    initial = np.asarray(initial_object_position, dtype=np.float64)
    final = np.asarray(final_object_position, dtype=np.float64)
    tool = np.asarray(final_tool_position, dtype=np.float64)
    lift = float(final[2] - initial[2])
    distance = float(np.linalg.norm(final - tool))
    success = lift >= minimum_lift_m and distance <= maximum_tool_distance_m
    return success, {
        "lift_m": lift,
        "tool_object_distance_m": distance,
        "lift_threshold_met": lift >= minimum_lift_m,
        "object_near_tool": distance <= maximum_tool_distance_m,
    }
