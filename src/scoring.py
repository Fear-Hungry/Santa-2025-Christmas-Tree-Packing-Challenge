from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np

from geom_np import packing_score, prefix_score, transform_polygon
from tree_data import TREE_POINTS


def _parse_val(value: str) -> float:
    value = value.strip()
    if value.startswith("s") or value.startswith("S"):
        value = value[1:]
    return float(value)


def load_submission(csv_path: Path, *, nmax: int | None = None) -> dict[int, np.ndarray]:
    puzzles: dict[int, list[list[float]]] = defaultdict(list)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            puzzle_id = int(row["id"].split("_")[0])
            if nmax is not None and puzzle_id > nmax:
                continue
            puzzles[puzzle_id].append(
                [
                    _parse_val(row["x"]),
                    _parse_val(row["y"]),
                    _parse_val(row["deg"]),
                ]
            )
    return {pid: np.array(rows, dtype=float) for pid, rows in puzzles.items()}


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_intersect(p1, p2, p3, p4, eps: float = 1e-9) -> bool:
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)
    return (d1 > eps) != (d2 > eps) and (d3 > eps) != (d4 > eps)


def _point_in_polygon(point: np.ndarray, poly: np.ndarray, eps: float = 1e-9) -> bool:
    x, y = point
    inside = False
    n = poly.shape[0]
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        cond1 = (p1[1] > y) != (p2[1] > y)
        if not cond1:
            continue
        x_int = (p2[0] - p1[0]) * (y - p1[1]) / (p2[1] - p1[1] + eps) + p1[0]
        if x + eps < x_int:
            inside = not inside
    return inside


def _polygons_intersect(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    min1 = np.min(poly1, axis=0)
    max1 = np.max(poly1, axis=0)
    min2 = np.min(poly2, axis=0)
    max2 = np.max(poly2, axis=0)
    if not (np.all(max1 >= min2) and np.all(max2 >= min1)):
        return False

    n1 = poly1.shape[0]
    n2 = poly2.shape[0]
    for i in range(n1):
        p1 = poly1[i]
        p2 = poly1[(i + 1) % n1]
        for j in range(n2):
            p3 = poly2[j]
            p4 = poly2[(j + 1) % n2]
            if _segments_intersect(p1, p2, p3, p4):
                return True

    if _point_in_polygon(poly1[0], poly2):
        return True
    if _point_in_polygon(poly2[0], poly1):
        return True
    return False


def polygons_intersect(poly1: np.ndarray, poly2: np.ndarray) -> bool:
    return _polygons_intersect(poly1, poly2)


def _check_overlaps(points: np.ndarray, poses: np.ndarray) -> bool:
    polys: list[np.ndarray] = []
    for pose in poses:
        polys.append(transform_polygon(points, pose))
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if _polygons_intersect(polys[i], polys[j]):
                return True
    return False


@dataclass
class ScoreResult:
    nmax: int
    score: float
    s_max: float
    overlap_check: bool
    require_complete: bool
    per_n: list[dict]

    def to_json(self) -> dict:
        return {
            "nmax": self.nmax,
            "score": self.score,
            "s_max": self.s_max,
            "overlap_check": self.overlap_check,
            "require_complete": self.require_complete,
            "per_n": self.per_n,
        }


def score_submission(
    csv_path: Path,
    *,
    nmax: int | None = None,
    check_overlap: bool = True,
    require_complete: bool = True,
) -> ScoreResult:
    points = np.array(TREE_POINTS, dtype=float)
    puzzles = load_submission(csv_path, nmax=nmax)
    if not puzzles:
        return ScoreResult(0, 0.0, 0.0, check_overlap, require_complete, [])

    max_n = max(puzzles)
    if nmax is None:
        nmax = max_n
    if require_complete:
        missing = [n for n in range(1, nmax + 1) if n not in puzzles]
        if missing:
            raise ValueError(f"Missing puzzles: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    total = 0.0
    s_max = 0.0
    per_n: list[dict] = []
    for n in range(1, nmax + 1):
        poses = puzzles.get(n)
        if poses is None:
            continue
        if poses.shape[0] != n:
            raise ValueError(f"Puzzle {n} expected {n} trees, got {poses.shape[0]}")
        if check_overlap and _check_overlaps(points, poses):
            raise ValueError(f"Overlap detected in puzzle {n}")
        s = packing_score(points, poses)
        s_max = max(s_max, s)
        contrib = (s * s) / n
        total += contrib
        per_n.append({"puzzle": n, "s": s, "contrib": contrib})

    return ScoreResult(nmax, total, s_max, check_overlap, require_complete, per_n)


def score_prefix(s_values: Iterable[float]) -> float:
    return prefix_score(s_values)
