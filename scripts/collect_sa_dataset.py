#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from geom_np import packing_score, polygon_radius, shift_poses_to_origin, transform_polygon  # noqa: E402
from scoring import polygons_intersect  # noqa: E402
from tree_data import TREE_POINTS  # noqa: E402


def _grid_initial(n: int, spacing: float) -> np.ndarray:
    cols = int(np.ceil(np.sqrt(n)))
    poses = np.zeros((n, 3), dtype=float)
    for i in range(n):
        row = i // cols
        col = i % cols
        poses[i] = (col * spacing, row * spacing, 0.0)
    return poses


def _random_initial(n: int, spacing: float, rand_scale: float) -> np.ndarray:
    scale = spacing * math.sqrt(max(n, 1)) * rand_scale
    xy = np.random.uniform(-scale, scale, size=(n, 2))
    theta = np.random.uniform(0.0, 360.0, size=(n, 1))
    return np.concatenate([xy, theta], axis=1)


def _check_overlaps(points: np.ndarray, poses: np.ndarray) -> bool:
    polys = [transform_polygon(points, pose) for pose in poses]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_intersect(polys[i], polys[j]):
                return True
    return False


@dataclass
class RunRecord:
    poses: List[np.ndarray]
    idxs: List[int]
    deltas: List[np.ndarray]
    final_score: float


def _run_sa_collect(
    n: int,
    *,
    steps: int,
    t_start: float,
    t_end: float,
    trans_sigma: float,
    rot_sigma: float,
    init_mode: str,
    rand_scale: float,
    points: np.ndarray,
) -> RunRecord:
    radius = polygon_radius(points)
    spacing = 2.0 * radius * 1.2
    if init_mode == "grid":
        poses = _grid_initial(n, spacing)
    elif init_mode == "random":
        poses = _random_initial(n, spacing, rand_scale)
    else:
        poses = _grid_initial(n, spacing)

    poses = shift_poses_to_origin(points, poses)
    score = packing_score(points, poses)

    accepted_poses: List[np.ndarray] = []
    accepted_idxs: List[int] = []
    accepted_deltas: List[np.ndarray] = []

    for i in range(steps):
        frac = i / max(steps, 1)
        temp = t_start * (t_end / t_start) ** frac
        idx = np.random.randint(0, n)
        delta = np.random.normal(size=(3,))
        delta[0:2] *= trans_sigma * temp
        delta[2] *= rot_sigma * temp
        candidate = poses.copy()
        candidate[idx] = candidate[idx] + delta
        candidate[idx, 2] = np.mod(candidate[idx, 2], 360.0)

        if _check_overlaps(points, candidate):
            continue

        cand_score = packing_score(points, candidate)
        dscore = cand_score - score
        if dscore < 0 or np.random.rand() < math.exp(-dscore / max(temp, 1e-9)):
            accepted_poses.append(poses.copy())
            accepted_idxs.append(idx)
            accepted_deltas.append(delta.copy())
            poses = candidate
            score = cand_score

    return RunRecord(accepted_poses, accepted_idxs, accepted_deltas, score)


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect behavior cloning dataset from SA runs")
    ap.add_argument("--n-list", type=str, default="25,50,100", help="Comma-separated Ns")
    ap.add_argument("--runs-per-n", type=int, default=5, help="SA runs per N")
    ap.add_argument("--steps", type=int, default=400, help="SA steps per run")
    ap.add_argument("--t-start", type=float, default=1.0)
    ap.add_argument("--t-end", type=float, default=0.001)
    ap.add_argument("--trans-sigma", type=float, default=0.2)
    ap.add_argument("--rot-sigma", type=float, default=10.0)
    ap.add_argument("--init", type=str, default="grid", choices=["grid", "random", "mix"])
    ap.add_argument("--rand-scale", type=float, default=0.3)
    ap.add_argument("--best-only", action="store_true", help="Keep only best run per N")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "sa_bc_dataset.npz")
    args = ap.parse_args()

    ns = [int(x) for x in args.n_list.split(",") if x.strip()]
    points = np.array(TREE_POINTS, dtype=float)

    payload: Dict[str, np.ndarray] = {}
    for n in ns:
        runs: List[RunRecord] = []
        for r in range(args.runs_per_n):
            if args.init == "mix":
                init_mode = "grid" if r % 2 == 0 else "random"
            else:
                init_mode = args.init
            run = _run_sa_collect(
                n,
                steps=args.steps,
                t_start=args.t_start,
                t_end=args.t_end,
                trans_sigma=args.trans_sigma,
                rot_sigma=args.rot_sigma,
                init_mode=init_mode,
                rand_scale=args.rand_scale,
                points=points,
            )
            runs.append(run)

        if args.best_only and runs:
            runs = [min(runs, key=lambda r: r.final_score)]

        poses = np.concatenate([np.array(r.poses) for r in runs], axis=0) if runs else np.zeros((0, n, 3))
        idxs = np.concatenate([np.array(r.idxs, dtype=int) for r in runs], axis=0) if runs else np.zeros((0,), dtype=int)
        deltas = np.concatenate([np.array(r.deltas, dtype=float) for r in runs], axis=0) if runs else np.zeros((0, 3))

        payload[f"poses_n{n}"] = poses
        payload[f"idx_n{n}"] = idxs
        payload[f"delta_n{n}"] = deltas

        print(f"N={n} samples={poses.shape[0]} best_only={args.best_only}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **payload)
    print(f"Saved dataset to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
