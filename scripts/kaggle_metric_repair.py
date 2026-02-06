#!/usr/bin/env python3
"""Repair Kaggle-metric overlaps with minimal local perturbations.

This script targets the official Kaggle overlap predicate (Shapely) and tries
very small, localized nudges to remove microscopic overlaps without impacting
score. It searches deterministic + random directions for the offending pair
and only falls back to larger steps if needed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from santa_packing.scoring import load_submission
from santa_packing.geom_np import packing_score
from santa_packing.tree_data import TREE_POINTS
from santa_packing.submission_format import format_submission_value
from santa_packing.cli.improve_submission import _write_submission
from santa_packing._tools.kaggle_autofix_submission import (
    _build_kaggle_metric,
    _first_metric_overlap_pair,
    _canonicalize_for_fix,
)


def _direction_pool(base_dir: np.ndarray, rng: np.random.Generator, extra_random: int) -> list[np.ndarray]:
    orth = np.array([-base_dir[1], base_dir[0]], dtype=float)
    dirs = [
        base_dir,
        -base_dir,
        orth,
        -orth,
        np.array([1.0, 0.0], dtype=float),
        np.array([-1.0, 0.0], dtype=float),
        np.array([0.0, 1.0], dtype=float),
        np.array([0.0, -1.0], dtype=float),
    ]
    for _ in range(int(extra_random)):
        v = rng.normal(0.0, 1.0, size=(2,))
        norm = float(np.linalg.norm(v))
        if norm < 1e-12:
            continue
        dirs.append(v / norm)
    return dirs


def _pair_overlaps(metric, pose_i: np.ndarray, pose_j: np.ndarray) -> bool:
    xi = format_submission_value(float(pose_i[0]))[1:]
    yi = format_submission_value(float(pose_i[1]))[1:]
    di = format_submission_value(float(pose_i[2]))[1:]
    xj = format_submission_value(float(pose_j[0]))[1:]
    yj = format_submission_value(float(pose_j[1]))[1:]
    dj = format_submission_value(float(pose_j[2]))[1:]
    pi = metric.ChristmasTree(xi, yi, di).polygon
    pj = metric.ChristmasTree(xj, yj, dj).polygon
    return bool(pi.intersects(pj) and not pi.touches(pj))


def _try_fix_pair(
    metric,
    points: np.ndarray,
    poses: np.ndarray,
    pair: tuple[int, int],
    *,
    seed: int,
    scales: list[float],
    random_dirs: int,
    detouch_scales: list[float],
    max_rotate_deg: float,
) -> tuple[np.ndarray, float, float] | None:
    base = _canonicalize_for_fix(poses)
    base_s = float(packing_score(points, base))

    # Tiny global detouch (keeps changes minimal when overlaps are numerical).
    for eps in detouch_scales:
        if eps <= 0.0:
            continue
        try:
            cand = _canonicalize_for_fix(_scale_about_centroid_xy(base, scale=1.0 + float(eps)))
        except Exception:
            continue
        if not _pair_overlaps(metric, cand[pair[0]], cand[pair[1]]):
            s = float(packing_score(points, cand))
            return cand, base_s, s

    rng = np.random.default_rng(int(seed))
    i, j = int(pair[0]), int(pair[1])
    base_centers = np.array(base[:, 0:2], dtype=float, copy=False)
    base_delta = base_centers[j] - base_centers[i]
    base_norm = float(np.linalg.norm(base_delta))
    if not np.isfinite(base_norm) or base_norm < 1e-12:
        base_delta = rng.normal(0.0, 1.0, size=(2,))
        base_norm = float(np.linalg.norm(base_delta))
    base_dir = base_delta / float(max(base_norm, 1e-12))

    for scale in scales:
        best = None
        best_s = None
        dirs = _direction_pool(base_dir, rng, extra_random=random_dirs)
        for d in dirs:
            for mode in ("pair", "i", "j"):
                for sign in (1.0, -1.0):
                    delta = d * float(scale) * float(sign)
                    cand = np.array(base, dtype=float, copy=True)
                    if mode == "pair":
                        cand[i, 0:2] -= delta
                        cand[j, 0:2] += delta
                    elif mode == "i":
                        cand[i, 0:2] += delta
                    else:
                        cand[j, 0:2] += delta

                    # Optional tiny rotation on one of the pair.
                    if max_rotate_deg > 0.0:
                        k = i if rng.random() < 0.5 else j
                        cand[k, 2] = float(np.mod(cand[k, 2] + rng.normal(0.0, max_rotate_deg), 360.0))

                    try:
                        cand = _canonicalize_for_fix(cand)
                    except Exception:
                        continue

                    if _pair_overlaps(metric, cand[i], cand[j]):
                        continue

                    s = float(packing_score(points, cand))
                    if best is None or s < best_s:
                        best = cand
                        best_s = s
        if best is not None:
            return best, base_s, float(best_s)

    return None


def _scale_about_centroid_xy(poses: np.ndarray, *, scale: float) -> np.ndarray:
    poses = np.array(poses, dtype=float, copy=True)
    if poses.shape[0] == 0:
        return poses
    c = np.mean(poses[:, 0:2], axis=0)
    poses[:, 0:2] = c[None, :] + (poses[:, 0:2] - c[None, :]) * float(scale)
    return poses


def main() -> int:
    ap = argparse.ArgumentParser(description="Repair Kaggle-metric overlaps with minimal local perturbations.")
    ap.add_argument("submission", type=Path, help="Input submission.csv")
    ap.add_argument("--out", type=Path, required=True, help="Output submission.csv")
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle n (default: 200)")
    ap.add_argument("--seed", type=int, default=123, help="Base RNG seed (default: 123)")
    ap.add_argument("--random-dirs", type=int, default=24, help="Random directions per scale (default: 24)")
    ap.add_argument("--max-rotate-deg", type=float, default=0.0, help="Stddev for tiny angle jitters (default: 0.0)")
    args = ap.parse_args()

    nmax = int(args.nmax)
    if nmax <= 0 or nmax > 200:
        raise SystemExit("--nmax must be in [1,200]")

    puzzles = load_submission(args.submission, nmax=nmax)
    missing = [n for n in range(1, nmax + 1) if n not in puzzles or puzzles[n].shape != (n, 3)]
    if missing:
        raise SystemExit(f"Invalid/missing puzzles: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    metric = _build_kaggle_metric()
    points = np.array(TREE_POINTS, dtype=float)

    scales = [
        1e-9,
        3e-9,
        1e-8,
        3e-8,
        1e-7,
        3e-7,
        1e-6,
        3e-6,
        1e-5,
        3e-5,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
    ]
    detouch_scales = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]

    out: dict[int, np.ndarray] = {}
    fixed_count = 0

    for n in range(1, nmax + 1):
        poses = _canonicalize_for_fix(puzzles[n])
        passes = 0
        while True:
            pair = _first_metric_overlap_pair(metric, poses)
            if pair is None:
                break
            passes += 1
            res = _try_fix_pair(
                metric,
                points,
                poses,
                pair,
                seed=int(args.seed) + 1_000_003 * n + 97 * passes,
                scales=scales,
                random_dirs=int(args.random_dirs),
                detouch_scales=detouch_scales,
                max_rotate_deg=float(args.max_rotate_deg),
            )
            if res is None:
                raise SystemExit(f"Failed to repair n={n} pair={pair} after search.")
            poses, base_s, best_s = res
            fixed_count += 1
            print(f"[n={n:03d}] fixed pair {pair} (pass {passes}) base={base_s:.12f} best={best_s:.12f}")
            if passes > 10:
                raise SystemExit(f"Too many repair passes for n={n}; aborting.")

        out[n] = poses

    _write_submission(args.out, out, nmax=nmax)
    print(f"wrote: {args.out}")
    print(f"fixed pairs: {fixed_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
