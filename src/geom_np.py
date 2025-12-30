from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def transform_polygon(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Rotate (deg) + translate points by pose (x, y, theta_deg)."""
    x, y, theta_deg = pose
    theta = math.radians(theta_deg)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return points @ rot.T + np.array([x, y], dtype=float)


def polygon_bbox(poly: np.ndarray) -> np.ndarray:
    """Return [min_x, min_y, max_x, max_y]."""
    min_xy = np.min(poly, axis=0)
    max_xy = np.max(poly, axis=0)
    return np.array([min_xy[0], min_xy[1], max_xy[0], max_xy[1]], dtype=float)


def packing_bbox(points: np.ndarray, poses: np.ndarray) -> np.ndarray:
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for pose in poses:
        poly = transform_polygon(points, pose)
        bbox = polygon_bbox(poly)
        min_x = min(min_x, bbox[0])
        min_y = min(min_y, bbox[1])
        max_x = max(max_x, bbox[2])
        max_y = max(max_y, bbox[3])
    return np.array([min_x, min_y, max_x, max_y], dtype=float)


def packing_score(points: np.ndarray, poses: np.ndarray) -> float:
    bbox = packing_bbox(points, poses)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return float(max(width, height))


def shift_poses_to_origin(points: np.ndarray, poses: np.ndarray) -> np.ndarray:
    bbox = packing_bbox(points, poses)
    shift_x = -bbox[0]
    shift_y = -bbox[1]
    shifted = np.array(poses, dtype=float, copy=True)
    shifted[:, 0] += shift_x
    shifted[:, 1] += shift_y
    return shifted


def polygon_radius(points: np.ndarray) -> float:
    norms = np.linalg.norm(points, axis=1)
    return float(np.max(norms))


def prefix_score(s_values: Iterable[float]) -> float:
    total = 0.0
    for idx, s in enumerate(s_values, start=1):
        total += (s * s) / idx
    return total
