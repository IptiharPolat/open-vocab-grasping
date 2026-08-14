from __future__ import annotations

import json
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from open_vocab_grasping.grasping.association import Detection


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def prompt_consensus_detections(
    detections: list[Detection],
    canonical_label: str,
    *,
    minimum_prompt_votes: int,
    consensus_iou: float,
) -> tuple[list[Detection], list[int]]:
    """Fuse low-confidence boxes only when distinct text prompts agree spatially."""
    groups: list[list[Detection]] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        destination: list[Detection] | None = None
        for group in groups:
            if _box_iou(detection.bbox_xyxy, group[0].bbox_xyxy) >= consensus_iou:
                destination = group
                break
        if destination is None:
            groups.append([detection])
        elif detection.label not in {item.label for item in destination}:
            destination.append(detection)

    fused: list[Detection] = []
    votes: list[int] = []
    for group in groups:
        if len(group) < minimum_prompt_votes:
            continue
        confidence = np.asarray([item.confidence for item in group], dtype=np.float64)
        boxes = np.asarray([item.bbox_xyxy for item in group], dtype=np.float64)
        weights = np.maximum(confidence, 1e-9)
        fused_box = np.average(boxes, axis=0, weights=weights)
        # Geometric mean rewards consistently supported boxes and suppresses a
        # group dominated by one anomalously high prompt score. Multiplying by
        # distinct prompt votes makes stronger semantic consensus rank first.
        consensus_confidence = float(
            np.clip(np.exp(np.mean(np.log(np.maximum(confidence, 1e-9)))) * len(group), 0.0, 1.0)
        )
        fused.append(
            Detection(
                tuple(float(value) for value in fused_box),
                canonical_label,
                consensus_confidence,
            )
        )
        votes.append(len(group))
    order = np.argsort([-item.confidence for item in fused])
    return [fused[index] for index in order], [votes[index] for index in order]


def filter_tabletop_fallback_geometry(
    detections: list[Detection],
    votes: list[int],
    image_shape: tuple[int, int],
    *,
    maximum_area_fraction: float,
    reject_border_touching: bool,
) -> tuple[list[Detection], list[int], int]:
    """Reject scene/table-sized fallback boxes using only image geometry."""
    height, width = image_shape
    image_area = float(height * width)
    kept_detections: list[Detection] = []
    kept_votes: list[int] = []
    for detection, vote_count in zip(detections, votes):
        x1, y1, x2, y2 = detection.bbox_xyxy
        area_fraction = max(0.0, x2 - x1) * max(0.0, y2 - y1) / image_area
        touches_border = x1 <= 1.0 or y1 <= 1.0 or x2 >= width - 1.0 or y2 >= height - 1.0
        if area_fraction > maximum_area_fraction or (reject_border_touching and touches_border):
            continue
        kept_detections.append(detection)
        kept_votes.append(vote_count)
    return kept_detections, kept_votes, len(detections) - len(kept_detections)


class YOLOWorldDetector:
    """Lazy real YOLO-World adapter; never falls back to oracle silently."""

    def __init__(
        self,
        model_path: str,
        confidence: float,
        iou: float,
        device: str | None = None,
        image_size: int = 640,
        max_detections: int = 50,
        fallback_confidence: float | None = None,
        fallback_min_prompt_votes: int = 2,
        fallback_consensus_iou: float = 0.55,
        fallback_maximum_area_fraction: float = 0.08,
        fallback_reject_border_touching: bool = True,
        force_prompt_consensus: bool = False,
    ):
        try:
            from ultralytics import YOLOWorld
            from ultralytics.utils.downloads import attempt_download_asset
        except ImportError as exc:
            raise RuntimeError(
                "YOLO-World dependencies are not installed. Install the `yolo` extra and download weights."
            ) from exc
        try:
            import clip  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Dynamic YOLO-World prompts require the Ultralytics CLIP fork. "
                "Install git+https://github.com/ultralytics/CLIP.git."
            ) from exc
        path = Path(model_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            downloaded = Path(attempt_download_asset(path.name))
            if not downloaded.exists():
                raise FileNotFoundError(f"Unable to download official YOLO-World weights: {path.name}")
            if downloaded.resolve() != path.resolve():
                shutil.move(str(downloaded), path)
        self.model_path = path.resolve()
        self.model = YOLOWorld(str(self.model_path))
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.image_size = image_size
        self.max_detections = max_detections
        self.fallback_confidence = fallback_confidence
        self.fallback_min_prompt_votes = fallback_min_prompt_votes
        self.fallback_consensus_iou = fallback_consensus_iou
        self.fallback_maximum_area_fraction = fallback_maximum_area_fraction
        self.fallback_reject_border_touching = fallback_reject_border_touching
        self.force_prompt_consensus = force_prompt_consensus
        self.last_inference_s = 0.0
        self.last_speed_ms: dict[str, float] = {}
        self.last_retry_used = False
        self.last_primary_count = 0
        self.last_fallback_raw_count = 0
        self.last_consensus_votes: list[int] = []
        self.last_fallback_geometry_rejected = 0

    def _predict(
        self, rgb: np.ndarray, classes: list[str], confidence_threshold: float
    ) -> tuple[list[Detection], float, dict[str, float]]:
        self.model.set_classes(classes)
        started = perf_counter()
        # A PIL source preserves RGB ordering. Ultralytics interprets ndarray
        # image sources as BGR, which would silently swap red and blue prompts.
        result = self.model.predict(
            source=Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB"),
            conf=confidence_threshold,
            iou=self.iou,
            imgsz=self.image_size,
            max_det=self.max_detections,
            device=self.device,
            verbose=False,
        )[0]
        wall_s = perf_counter() - started
        speed_ms = {key: float(value) for key, value in result.speed.items()}
        detections: list[Detection] = []
        if result.boxes is None:
            return detections, wall_s, speed_ms
        for xyxy, class_id, confidence in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int),
            result.boxes.conf.cpu().numpy(),
        ):
            detections.append(
                Detection(tuple(float(v) for v in xyxy), classes[class_id], float(confidence))
            )
        return detections, wall_s, speed_ms

    def detect(self, rgb: np.ndarray, classes: list[str]) -> list[Detection]:
        primary, wall_s, speed_ms = self._predict(rgb, classes, self.confidence)
        self.last_inference_s = wall_s
        self.last_speed_ms = speed_ms
        self.last_retry_used = False
        self.last_primary_count = len(primary)
        self.last_fallback_raw_count = 0
        self.last_consensus_votes = []
        self.last_fallback_geometry_rejected = 0
        if (primary and not self.force_prompt_consensus) or self.fallback_confidence is None:
            return primary
        if self.fallback_confidence >= self.confidence or len(classes) < self.fallback_min_prompt_votes:
            return primary

        self.last_retry_used = True
        fallback: list[Detection] = []
        aggregate_speed = dict(speed_ms)
        for prompt in classes:
            prompt_detections, prompt_wall_s, prompt_speed = self._predict(
                rgb, [prompt], self.fallback_confidence
            )
            fallback.extend(prompt_detections)
            self.last_inference_s += prompt_wall_s
            for key, value in prompt_speed.items():
                aggregate_speed[key] = aggregate_speed.get(key, 0.0) + value
        self.last_speed_ms = aggregate_speed
        self.last_fallback_raw_count = len(fallback)
        fused, votes = prompt_consensus_detections(
            fallback,
            classes[0],
            minimum_prompt_votes=self.fallback_min_prompt_votes,
            consensus_iou=self.fallback_consensus_iou,
        )
        fused, votes, rejected = filter_tabletop_fallback_geometry(
            fused,
            votes,
            np.asarray(rgb).shape[:2],
            maximum_area_fraction=self.fallback_maximum_area_fraction,
            reject_border_touching=self.fallback_reject_border_touching,
        )
        self.last_fallback_geometry_rejected = rejected
        self.last_consensus_votes = votes
        return fused


def save_detections(path: str | Path, detections: list[Detection]) -> None:
    Path(path).write_text(
        json.dumps(
            [
                {"bbox_xyxy": detection.bbox_xyxy, "label": detection.label, "confidence": detection.confidence}
                for detection in detections
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
