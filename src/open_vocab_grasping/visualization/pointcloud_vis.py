from __future__ import annotations

from pathlib import Path

import numpy as np


def save_topdown_preview(path: str | Path, points: np.ndarray, colors: np.ndarray, size: int = 640) -> None:
    from PIL import Image

    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    if len(points):
        x, y = points[:, 0], points[:, 1]
        u = np.clip(((x - 0.15) / 0.75 * (size - 1)).astype(int), 0, size - 1)
        v = np.clip(((0.5 - y) / 1.0 * (size - 1)).astype(int), 0, size - 1)
        order = np.argsort(points[:, 2])
        canvas[v[order], u[order]] = np.clip(colors[order] * 255, 0, 255).astype(np.uint8)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGB").save(output)


def save_candidate_topdown(
    path: str | Path,
    points_world: np.ndarray,
    colors: np.ndarray,
    candidate_centers_world: np.ndarray,
    accepted: np.ndarray,
    size: int = 640,
) -> None:
    from PIL import Image, ImageDraw

    temporary = Path(path).with_suffix(".base.png")
    save_topdown_preview(temporary, points_world, colors, size)
    image = Image.open(temporary).convert("RGB")
    temporary.unlink()
    draw = ImageDraw.Draw(image)
    for index, (center, keep) in enumerate(zip(candidate_centers_world, accepted)):
        u = int(np.clip((center[0] - 0.15) / 0.75 * (size - 1), 0, size - 1))
        v = int(np.clip((0.5 - center[1]) / 1.0 * (size - 1), 0, size - 1))
        color = (20, 190, 40) if keep else (255, 130, 20)
        draw.ellipse((u - 6, v - 6, u + 6, v + 6), fill=color, outline=(0, 0, 0))
        draw.text((u + 8, v - 8), str(index), fill=color, stroke_width=1, stroke_fill=(0, 0, 0))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
