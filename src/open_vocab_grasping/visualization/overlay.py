from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from open_vocab_grasping.grasping.association import AssociationRecord, Detection


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(output)


def save_detection_overlay(path: str | Path, rgb: np.ndarray, detections: list[Detection]) -> None:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    for detection in detections:
        draw.rectangle(detection.bbox_xyxy, outline=(255, 30, 30), width=3)
        draw.text((detection.bbox_xyxy[0] + 3, detection.bbox_xyxy[1] + 3),
                  f"{detection.label} {detection.confidence:.2f}", fill=(255, 30, 30))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def save_association_overlay(
    path: str | Path,
    rgb: np.ndarray,
    detection: Detection,
    records: list[AssociationRecord],
) -> None:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(detection.bbox_xyxy, outline=(255, 30, 30), width=3)
    draw.text(
        (detection.bbox_xyxy[0] + 3, detection.bbox_xyxy[1] + 3),
        f"target: {detection.label} {detection.confidence:.2f}",
        fill=(255, 30, 30),
    )
    for record in records:
        if record.projected_uv is None:
            continue
        u, v = record.projected_uv
        color = (20, 190, 40) if record.accepted else (255, 150, 20)
        radius = 5
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=color, outline=(0, 0, 0))
        draw.text((u + 7, v - 7), str(record.candidate_index), fill=color, stroke_width=1,
                  stroke_fill=(0, 0, 0))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
