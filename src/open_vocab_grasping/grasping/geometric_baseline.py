from __future__ import annotations

import numpy as np

from open_vocab_grasping.grasping.interface import GraspCandidate


def _circle_from_three(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    a, b, c = points
    matrix = 2.0 * np.array([b - a, c - a])
    if abs(np.linalg.det(matrix)) < 1e-8:
        return None
    rhs = np.array([np.dot(b, b) - np.dot(a, a), np.dot(c, c) - np.dot(a, a)])
    center = np.linalg.solve(matrix, rhs)
    return center, float(np.linalg.norm(center - a))


def _robust_cylindrical_center(points: np.ndarray) -> np.ndarray | None:
    """Fit an XY circle while rejecting handles and other protrusions."""
    z_low, z_high = np.quantile(points[:, 2], [0.25, 0.75])
    xy = points[(points[:, 2] >= z_low) & (points[:, 2] <= z_high), :2]
    if len(xy) < 30:
        return None
    rng = np.random.default_rng(0)
    best_center: np.ndarray | None = None
    best_count = 0
    for indices in rng.integers(0, len(xy), size=(256, 3)):
        circle = _circle_from_three(xy[indices])
        if circle is None:
            continue
        center, radius = circle
        if not 0.015 <= radius <= 0.055:
            continue
        residual = np.abs(np.linalg.norm(xy - center, axis=1) - radius)
        count = int(np.count_nonzero(residual < 0.004))
        if count > best_count:
            best_center, best_count = center, count
    if best_center is None or best_count < 0.20 * len(xy):
        return None
    return best_center


class GeometricTopDownGenerator:
    """Explicit CPU-only baseline; this is not GraspNet inference."""

    name = "geometric_baseline"

    def generate_from_points(self, points_world: np.ndarray) -> list[GraspCandidate]:
        points = np.asarray(points_world, dtype=np.float64)
        if len(points) < 10:
            return []
        low = np.quantile(points, 0.05, axis=0)
        high = np.quantile(points, 0.95, axis=0)
        # Estimate an unseen cylinder center from the visible arc while rejecting
        # protrusions such as a mug handle. Fall back to robust surface medians.
        center = np.median(points, axis=0)
        cylindrical_center = _robust_cylindrical_center(points)
        if cylindrical_center is not None:
            center[:2] = cylindrical_center
        center[2] = (low[2] + high[2]) / 2.0
        extents = high - low
        width = float(min(max(min(extents[0], extents[1]) + 0.012, 0.025), 0.078))
        # Tool z axis points down; tool x closes across the target. R is T_world_tool.
        rotations: list[np.ndarray] = []
        for yaw in (0.0, np.pi / 2.0, np.pi / 4.0, -np.pi / 4.0):
            x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            z = np.array([0.0, 0.0, -1.0])
            y = np.cross(z, x)
            rotations.append(np.column_stack((x, y, z)))
        return [
            GraspCandidate(
                center=center.copy(),
                rotation=rotation,
                width=width,
                depth=0.035,
                grasp_score=1.0 - 0.08 * index,
                frame="world",
                clearance=float(center[2]),
            )
            for index, rotation in enumerate(rotations)
        ]


def generate_scene_clusters(
    points_world: np.ndarray,
    table_z_m: float,
    eps_m: float = 0.018,
    min_points: int = 40,
    minimum_height_m: float = 0.008,
    maximum_height_m: float = 0.22,
) -> tuple[list[GraspCandidate], list[np.ndarray]]:
    """Generate scene proposals from geometry only; no simulation IDs are read."""
    points = np.asarray(points_world, dtype=np.float64)
    object_points = points[
        (points[:, 2] > table_z_m + minimum_height_m)
        & (points[:, 2] < table_z_m + maximum_height_m)
    ]
    if len(object_points) < min_points:
        return [], []
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(object_points)
    labels = np.asarray(
        cloud.cluster_dbscan(eps=float(eps_m), min_points=int(min_points), print_progress=False)
    )
    generator = GeometricTopDownGenerator()
    candidates: list[GraspCandidate] = []
    clusters: list[np.ndarray] = []
    for label in sorted(set(labels.tolist())):
        if label < 0:
            continue
        cluster = object_points[labels == label]
        if len(cluster) < min_points:
            continue
        extent = np.quantile(cluster, 0.95, axis=0) - np.quantile(cluster, 0.05, axis=0)
        # Reject broad robot/table fragments while retaining ordinary tabletop objects.
        if extent[0] > 0.16 or extent[1] > 0.16 or extent[2] > maximum_height_m:
            continue
        generated = generator.generate_from_points(cluster)
        if generated:
            cluster_index = len(clusters)
            for candidate in generated:
                candidate.source_cluster_id = cluster_index
            clusters.append(cluster)
            candidates.extend(generated)
    return candidates, clusters
