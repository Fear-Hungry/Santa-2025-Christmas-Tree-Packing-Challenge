#!/usr/bin/env python3

"""Autofix a submission using the official Kaggle metric overlap predicate.

The official evaluator for Santa 2025 uses Shapely (v2.1.2) and treats trees as
overlapping iff:
  poly.intersects(other) and not poly.touches(other)

Because the evaluator builds polygons in a large scaled coordinate space
(scale_factor=1e18) and then uses floating-point geometry, "perfect touch"
solutions can occasionally register as microscopic overlaps and get rejected.

This tool fixes that by applying tiny randomized jitters (with verification
against the same Shapely predicate) only to the puzzles that fail the metric.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np

from santa_packing.cli.improve_submission import _write_submission
from santa_packing.geom_np import packing_score
from santa_packing.scoring import load_submission
from santa_packing.submission_format import format_submission_value, quantize_for_submission, fit_xy_in_bounds
from santa_packing.tree_data import TREE_POINTS


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


@dataclass(frozen=True)
class _KaggleMetric:
    scale_factor: Decimal
    base_poly: object  # shapely Polygon
    ChristmasTree: type  # metric-compatible builder


def _build_kaggle_metric() -> _KaggleMetric:
    # Match the metric's Decimal settings.
    getcontext().prec = 25
    scale_factor = Decimal("1e18")

    try:
        from shapely import affinity  # type: ignore
        from shapely.geometry import Polygon  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "shapely is required for kaggle_autofix_submission "
            "(it is used by the official metric). Install with: pip install shapely"
        ) from exc

    # Build the exact same polygon as the metric (in scaled coordinates).
    trunk_w = Decimal("0.15")
    trunk_h = Decimal("0.2")
    base_w = Decimal("0.7")
    mid_w = Decimal("0.4")
    top_w = Decimal("0.25")
    tip_y = Decimal("0.8")
    tier_1_y = Decimal("0.5")
    tier_2_y = Decimal("0.25")
    base_y = Decimal("0.0")
    trunk_bottom_y = -trunk_h

    base_poly = Polygon(
        [
            (Decimal("0.0") * scale_factor, tip_y * scale_factor),
            (top_w / Decimal("2") * scale_factor, tier_1_y * scale_factor),
            (top_w / Decimal("4") * scale_factor, tier_1_y * scale_factor),
            (mid_w / Decimal("2") * scale_factor, tier_2_y * scale_factor),
            (mid_w / Decimal("4") * scale_factor, tier_2_y * scale_factor),
            (base_w / Decimal("2") * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal("2") * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal("2") * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal("2")) * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal("2")) * scale_factor, base_y * scale_factor),
            (-(base_w / Decimal("2")) * scale_factor, base_y * scale_factor),
            (-(mid_w / Decimal("4")) * scale_factor, tier_2_y * scale_factor),
            (-(mid_w / Decimal("2")) * scale_factor, tier_2_y * scale_factor),
            (-(top_w / Decimal("4")) * scale_factor, tier_1_y * scale_factor),
            (-(top_w / Decimal("2")) * scale_factor, tier_1_y * scale_factor),
        ]
    )

    class ChristmasTree:
        def __init__(self, center_x: str = "0", center_y: str = "0", angle: str = "0") -> None:
            self.center_x = Decimal(center_x)
            self.center_y = Decimal(center_y)
            self.angle = Decimal(angle)

            rotated = affinity.rotate(base_poly, float(self.angle), origin=(0, 0))
            self.polygon = affinity.translate(
                rotated,
                xoff=float(self.center_x * scale_factor),
                yoff=float(self.center_y * scale_factor),
            )

    return _KaggleMetric(scale_factor=scale_factor, base_poly=base_poly, ChristmasTree=ChristmasTree)


def _poses_to_metric_strings(poses: np.ndarray) -> tuple[list[str], list[str], list[str]]:
    xs: list[str] = []
    ys: list[str] = []
    degs: list[str] = []
    for x, y, deg in poses:
        xs.append(format_submission_value(float(x))[1:])
        ys.append(format_submission_value(float(y))[1:])
        degs.append(format_submission_value(float(deg))[1:])
    return xs, ys, degs


def _first_metric_overlap_pair(metric: _KaggleMetric, poses: np.ndarray) -> tuple[int, int] | None:
    try:
        from shapely.strtree import STRtree  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit("shapely is required for kaggle_autofix_submission") from exc

    xs, ys, degs = _poses_to_metric_strings(poses)
    polys = [metric.ChristmasTree(x, y, deg).polygon for x, y, deg in zip(xs, ys, degs)]
    tree = STRtree(polys)
    for i, poly in enumerate(polys):
        for j in tree.query(poly):
            if int(j) == int(i):
                continue
            other = polys[int(j)]
            if poly.intersects(other) and not poly.touches(other):
                return (int(i), int(j))
    return None


def _canonicalize_for_fix(poses: np.ndarray) -> np.ndarray:
    poses = np.array(poses, dtype=float, copy=True)
    poses[:, 2] = np.mod(poses[:, 2], 360.0)
    poses = fit_xy_in_bounds(poses)
    poses = quantize_for_submission(poses)
    return poses


def _scale_about_centroid_xy(poses: np.ndarray, *, scale: float) -> np.ndarray:
    poses = np.array(poses, dtype=float, copy=True)
    if poses.shape[0] == 0:
        return poses
    c = np.mean(poses[:, 0:2], axis=0)
    poses[:, 0:2] = c[None, :] + (poses[:, 0:2] - c[None, :]) * float(scale)
    return poses


def _fix_one_puzzle(
    metric: _KaggleMetric,
    points: np.ndarray,
    poses: np.ndarray,
    *,
    seed: int,
    jitter_xy: float,
    jitter_deg: float,
    max_jitter_xy: float,
    max_jitter_deg: float,
    detouch_eps: float,
    detouch_max_eps: float,
    attempts: int,
) -> tuple[np.ndarray, dict]:
    n = int(poses.shape[0])
    base = _canonicalize_for_fix(poses)
    base_s = float(packing_score(points, base))

    pair = _first_metric_overlap_pair(metric, base)
    if pair is None:
        return base, {"fixed": False, "attempts": 0, "base_s": base_s, "best_s": base_s}

    rng = np.random.default_rng(int(seed) + 1_000_003 * n)
    attempts_done = 0

    # IMPORTANT:
    # Microscopic overlaps happen due to float geometry + scaling in the official metric.
    # We want the *smallest possible* perturbation (ideally local to the offending pair),
    # and we must keep poses within submission bounds ([-100,100]) after canonicalization.
    #
    # The previous approach exponentially increased sigma, which could explode to huge
    # offsets and fail `fit_xy_in_bounds`. We instead cap jitters and primarily "push"
    # only the overlapping pair apart.

    # First, try a tiny *detouch* (uniform expansion about centroid). This is
    # robust for metric micro-overlaps and minimally impacts score when eps is tiny.
    for t in range(int(attempts)):
        attempts_done += 1
        eps = float(detouch_eps) * (2.0**t)
        eps = float(min(max(eps, 0.0), float(detouch_max_eps)))
        if eps <= 0.0:
            continue
        try:
            cand = _canonicalize_for_fix(_scale_about_centroid_xy(base, scale=1.0 + eps))
        except Exception:
            continue
        if _first_metric_overlap_pair(metric, cand) is None:
            s = float(packing_score(points, cand))
            return cand, {"fixed": True, "attempts": attempts_done, "base_s": base_s, "best_s": s}

    pair_i, pair_j = int(pair[0]), int(pair[1])

    best = None
    best_s = float("inf")

    base_centers = np.array(base[:, 0:2], dtype=float, copy=False)
    base_delta = base_centers[pair_j] - base_centers[pair_i]
    base_norm = float(np.linalg.norm(base_delta))
    if not np.isfinite(base_norm) or base_norm < 1e-12:
        base_delta = rng.normal(0.0, 1.0, size=(2,))
        base_norm = float(np.linalg.norm(base_delta))
    base_dir = base_delta / float(max(base_norm, 1e-12))

    for t in range(int(attempts)):
        attempts_done += 1
        step_xy = float(jitter_xy) * (2.0**t)
        step_deg = float(jitter_deg) * (2.0**t)
        step_xy = float(min(max(step_xy, 0.0), float(max_jitter_xy)))
        step_deg = float(min(max(step_deg, 0.0), float(max_jitter_deg)))

        cand = np.array(base, dtype=float, copy=True)

        # Deterministic push apart along the base pair direction + a tiny random sideways component.
        ortho = np.array([-base_dir[1], base_dir[0]], dtype=float)
        ortho_scale = float(step_xy) * float(rng.normal(0.0, 0.25))
        delta = base_dir * float(step_xy) + ortho * ortho_scale

        # Move only the offending pair (keeps solution close to base score).
        cand[pair_i, 0:2] -= delta
        cand[pair_j, 0:2] += delta

        # Optional small angle nudge on one of the pair.
        if step_deg > 0.0:
            k = pair_i if rng.random() < 0.5 else pair_j
            cand[k, 2] = float(np.mod(cand[k, 2] + rng.normal(0.0, step_deg), 360.0))

        try:
            cand = _canonicalize_for_fix(cand)
        except Exception:
            continue

        if _first_metric_overlap_pair(metric, cand) is not None:
            continue

        s = float(packing_score(points, cand))
        if best is None or s < best_s:
            best = cand
            best_s = s
            # Once metric-feasible, stop early; overlaps are usually microscopic.
            break

    if best is None:
        raise RuntimeError(
            "Failed to remove kaggle-metric overlap after bounded jitter attempts. "
            "Try increasing --detouch-max-eps slightly (e.g. 1e-6 -> 1e-5) or --max-jitter-xy."
        )

    return best, {"fixed": True, "attempts": attempts_done, "base_s": base_s, "best_s": float(best_s)}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description="Auto-fix a submission using the official Kaggle overlap predicate.")
    ap.add_argument("submission", type=Path, help="Input submission.csv")
    ap.add_argument("--out", type=Path, required=True, help="Output submission.csv")
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle n (default: 200)")
    ap.add_argument("--seed", type=int, default=123, help="Base RNG seed for repairs (default: 123)")
    ap.add_argument("--jitter-xy", type=float, default=1e-7, help="Initial XY jitter sigma (default: 1e-7)")
    ap.add_argument("--jitter-deg", type=float, default=0.0, help="Initial deg jitter sigma (default: 0.0)")
    ap.add_argument("--max-jitter-xy", type=float, default=1e-4, help="Max XY jitter step (default: 1e-4)")
    ap.add_argument("--max-jitter-deg", type=float, default=1e-2, help="Max deg jitter step (default: 1e-2)")
    ap.add_argument("--detouch-eps", type=float, default=1e-12, help="Initial detouch epsilon (default: 1e-12)")
    ap.add_argument("--detouch-max-eps", type=float, default=1e-6, help="Max detouch epsilon (default: 1e-6)")
    ap.add_argument("--attempts", type=int, default=20, help="Max jitter attempts per failing puzzle (default: 20)")
    args = ap.parse_args(argv)

    nmax = int(args.nmax)
    if nmax <= 0 or nmax > 200:
        raise SystemExit("--nmax must be in [1,200]")

    puzzles = load_submission(args.submission, nmax=nmax)
    missing = [n for n in range(1, nmax + 1) if n not in puzzles or puzzles[n].shape != (n, 3)]
    if missing:
        raise SystemExit(f"Invalid/missing puzzles: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    metric = _build_kaggle_metric()
    points = np.array(TREE_POINTS, dtype=float)

    out: dict[int, np.ndarray] = {}
    fixed_count = 0
    total_attempts = 0

    for n in range(1, nmax + 1):
        poses = puzzles[n]
        fixed, meta = _fix_one_puzzle(
            metric,
            points,
            poses,
            seed=int(args.seed) + 10_000_019 * n,
            jitter_xy=float(args.jitter_xy),
            jitter_deg=float(args.jitter_deg),
            max_jitter_xy=float(args.max_jitter_xy),
            max_jitter_deg=float(args.max_jitter_deg),
            detouch_eps=float(args.detouch_eps),
            detouch_max_eps=float(args.detouch_max_eps),
            attempts=int(args.attempts),
        )
        out[n] = fixed
        if bool(meta["fixed"]):
            fixed_count += 1
            total_attempts += int(meta["attempts"])
            _eprint(f"[n={n:03d}] fixed kaggle-metric overlap in {meta['attempts']} attempt(s)")

    _write_submission(args.out, out, nmax=nmax)
    _eprint(f"wrote: {args.out}")
    _eprint(f"fixed puzzles: {fixed_count}/{nmax} (attempts={total_attempts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
