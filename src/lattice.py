from __future__ import annotations

import math
from typing import Literal

import numpy as np

from geom_np import polygon_radius, shift_poses_to_origin, transform_polygon
from scoring import polygons_intersect
from tree_data import TREE_POINTS

Pattern = Literal["hex", "square"]


def lattice_poses(
    n: int,
    *,
    pattern: Pattern = "hex",
    margin: float = 0.02,
    rotate_deg: float = 0.0,
) -> np.ndarray:
    points = np.array(TREE_POINTS, dtype=float)
    step, row_height = _compute_spacing(points, pattern, rotate_deg, margin)

    if n <= 0:
        return np.zeros((0, 3), dtype=float)

    cols = int(math.ceil(math.sqrt(n)))
    poses = np.zeros((n, 3), dtype=float)

    if pattern == "hex":
        for i in range(n):
            row = i // cols
            col = i % cols
            x = col * step + (step / 2.0 if row % 2 == 1 else 0.0)
            y = row * row_height
            poses[i] = (x, y, rotate_deg)
    else:
        for i in range(n):
            row = i // cols
            col = i % cols
            x = col * step
            y = row * row_height
            poses[i] = (x, y, rotate_deg)

    return shift_poses_to_origin(points, poses)


def _compute_spacing(
    points: np.ndarray,
    pattern: Pattern,
    rotate_deg: float,
    margin: float,
) -> tuple[float, float]:
    radius = polygon_radius(points)
    upper = 2.5 * radius

    def no_overlap(dx: float, dy: float) -> bool:
        base = transform_polygon(points, np.array([0.0, 0.0, rotate_deg], dtype=float))
        other = transform_polygon(points, np.array([dx, dy, rotate_deg], dtype=float))
        return not polygons_intersect(base, other)

    def binary_search_dx(low: float, high: float, dy: float) -> float:
        for _ in range(50):
            mid = (low + high) / 2.0
            if no_overlap(mid, dy):
                high = mid
            else:
                low = mid
        return high

    def binary_search_dy(low: float, high: float, dx: float) -> float:
        for _ in range(50):
            mid = (low + high) / 2.0
            if no_overlap(dx, mid):
                high = mid
            else:
                low = mid
        return high

    if pattern == "hex":
        # Hex lattice neighbors for a node include (±dx, 0) and (±dx/2, ±dy).
        # When rotate_deg != 0, symmetry on x/y axes is not guaranteed, so we must
        # check all sign combinations for the staggered row offsets.
        dx = binary_search_dx(0.0, upper, 0.0)

        def _hex_ok(dy: float) -> bool:
            half = dx / 2.0
            offsets = [(half, dy), (-half, dy), (half, -dy), (-half, -dy)]
            return all(no_overlap(ox, oy) for ox, oy in offsets)

        low = 0.0
        high = upper
        for _ in range(50):
            mid = (low + high) / 2.0
            if _hex_ok(mid):
                high = mid
            else:
                low = mid
        row_height = max(high, 1e-6)
    else:
        # Square lattice neighbors include (±dx, 0), (0, ±dy) and (±dx, ±dy).
        dx = binary_search_dx(0.0, upper, 0.0)
        dy = binary_search_dy(0.0, upper, 0.0)

        def _square_ok(scale: float) -> bool:
            sx = dx * scale
            sy = dy * scale
            offsets = [
                (sx, 0.0),
                (-sx, 0.0),
                (0.0, sy),
                (0.0, -sy),
                (sx, sy),
                (sx, -sy),
                (-sx, sy),
                (-sx, -sy),
            ]
            return all(no_overlap(ox, oy) for ox, oy in offsets)

        # Ensure diagonal neighbors do not overlap (can happen for rotated shapes).
        scale_low = 1.0
        scale_high = 3.0
        # If even the upper scale is not enough, fall back to something safe.
        if not _square_ok(scale_high):
            scale_high = 5.0
        for _ in range(40):
            mid = (scale_low + scale_high) / 2.0
            if _square_ok(mid):
                scale_high = mid
            else:
                scale_low = mid
        dx = dx * scale_high
        row_height = dy * scale_high

    step = dx * (1.0 + margin)
    row_height = row_height * (1.0 + margin)
    return step, row_height
