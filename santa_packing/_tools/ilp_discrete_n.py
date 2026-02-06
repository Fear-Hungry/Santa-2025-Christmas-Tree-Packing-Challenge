#!/usr/bin/env python3

"""Discrete-candidate ILP/CP-SAT packer for a single puzzle `n`.

This is an "exact" solver only with respect to a *discretized* search space.
We build a finite set of candidate poses and solve a 0/1 problem:

  - choose exactly one candidate per tree (local offsets/rotations around a base packing)
  - forbid selecting overlapping candidate pairs (precomputed conflict graph)
  - minimize the AABB square side length `side = max(width, height)` (via big-M envelope)

The resulting packing is then finalized/quantized with the same helper used by
the generator to ensure it remains overlap-free under the requested mode.

Example (run for 2 hours on n=30):

  .venv/bin/python -m santa_packing._tools.ilp_discrete_n \\
    submission.csv --n 30 --out runs/ilp/n30.csv \\
    --delta-xy 0.02 --step-xy 0.01 --rot-deltas 0,180,1,-1 \\
    --time-limit-s 7200 --workers 16 --overlap-mode strict --log-progress
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from santa_packing.cli.generate_submission import _finalize_puzzle
from santa_packing.cli.improve_submission import _write_submission
from santa_packing.geom_np import packing_score, polygon_bbox, polygon_radius, shift_poses_to_origin, transform_polygon
from santa_packing.scoring import OverlapMode, first_overlap_pair, load_submission, polygons_intersect_strict
from santa_packing.tree_data import TREE_POINTS

try:
    from ortools.sat.python import cp_model  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - optional dependency
    cp_model = None
    _ORTOOLS_IMPORT_ERROR = exc


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _parse_float_list(text: str) -> list[float]:
    raw = text.strip()
    if not raw:
        return []
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _unique_offsets(delta: float, step: float) -> list[float]:
    delta = float(delta)
    step = float(step)
    if delta < 0.0 or step <= 0.0:
        raise ValueError("delta must be >=0 and step must be >0")
    if delta == 0.0:
        return [0.0]
    k = int(round(delta / step))
    vals = [i * step for i in range(-k, k + 1)]
    # Prefer 0 first (useful as a hint).
    vals.sort(key=lambda v: (abs(v) > 1e-15, abs(v)))
    # De-dup with rounding.
    uniq: list[float] = []
    seen = set()
    for v in vals:
        key = round(float(v), 12)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(float(v))
    return uniq


@dataclass(frozen=True)
class _Candidate:
    tree: int
    pose: np.ndarray  # (3,)
    bbox: np.ndarray  # (4,) xmin,ymin,xmax,ymax


def _aabb_overlaps(a: np.ndarray, b: np.ndarray, eps: float) -> bool:
    return not (
        float(a[2]) < float(b[0]) - eps
        or float(b[2]) < float(a[0]) - eps
        or float(a[3]) < float(b[1]) - eps
        or float(b[3]) < float(a[1]) - eps
    )


def _grid_pairs(centers: np.ndarray, *, cell_size: float) -> list[tuple[int, int]]:
    from collections import defaultdict

    inv = 1.0 / float(max(cell_size, 1e-12))
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, (x, y) in enumerate(centers):
        gx = int(math.floor(float(x) * inv))
        gy = int(math.floor(float(y) * inv))
        grid[(gx, gy)].append(int(idx))

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for (gx, gy), idxs in grid.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                other = grid.get((gx + dx, gy + dy))
                if not other:
                    continue
                for i in idxs:
                    for j in other:
                        if j <= i:
                            continue
                        key = (i, j)
                        if key in seen:
                            continue
                        seen.add(key)
                        pairs.append(key)
    return pairs


class _BestSaver(cp_model.CpSolverSolutionCallback):  # type: ignore[misc]
    def __init__(
        self,
        xs: list[cp_model.IntVar],  # type: ignore[name-defined]
        *,
        candidates: list[_Candidate],
        n: int,
        scale: int,
        out_dir: Path,
        points: np.ndarray,
        overlap_mode: OverlapMode,
        seed: int,
    ) -> None:
        super().__init__()
        self._xs = xs
        self._cands = candidates
        self._n = int(n)
        self._scale = int(scale)
        self._out_dir = Path(out_dir)
        self._points = np.array(points, dtype=float, copy=False)
        self._overlap_mode = overlap_mode
        self._seed = int(seed)
        self.best_side_int: int | None = None
        self.best_poses: np.ndarray | None = None
        self.solution_count = 0
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def on_solution_callback(self) -> None:
        self.solution_count += 1
        side_int = int(round(float(self.ObjectiveValue())))
        if self.best_side_int is not None and side_int >= self.best_side_int:
            return

        selected = [i for i, v in enumerate(self._xs) if self.Value(v) == 1]
        # Decode as one per tree.
        by_tree: list[list[int]] = [[] for _ in range(self._n)]
        for idx in selected:
            by_tree[self._cands[idx].tree].append(idx)
        if any(len(lst) != 1 for lst in by_tree):
            return
        pose_idxs = [lst[0] for lst in by_tree]
        poses = np.stack([self._cands[i].pose for i in pose_idxs], axis=0)

        # Finalize/quantize + validate.
        poses = shift_poses_to_origin(self._points, poses)
        poses = _finalize_puzzle(
            self._points,
            poses,
            seed=self._seed,
            puzzle_n=self._n,
            overlap_mode=self._overlap_mode,
        )
        if first_overlap_pair(self._points, poses, mode=self._overlap_mode) is not None:
            return

        self.best_side_int = side_int
        self.best_poses = np.array(poses, dtype=float, copy=True)

        side = side_int / float(self._scale)
        score = packing_score(self._points, poses)
        path = self._out_dir / f"best_n{self._n}_side{side:.9f}_score{score:.9f}.npy"
        np.save(path, poses)
        _eprint(f"[cb] new best: side≈{side:.9f} score={score:.9f} (saved {path.name})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Discrete-candidate ILP/CP-SAT solver for a single puzzle n.")
    ap.add_argument("submission", type=Path, help="Base submission.csv (used as initial packing / hint).")
    ap.add_argument("--n", type=int, required=True, help="Puzzle size to optimize (e.g., 30).")
    ap.add_argument("--out", type=Path, required=True, help="Output submission.csv path (base with puzzle n replaced).")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/ilp"), help="Directory to dump best intermediate .npy.")

    ap.add_argument("--delta-xy", type=float, default=0.02, help="Max translation delta in x/y around base poses.")
    ap.add_argument("--step-xy", type=float, default=0.01, help="Translation step for candidate grid.")
    ap.add_argument(
        "--rot-deltas",
        type=str,
        default="0,180",
        help="Comma-separated rotation deltas in degrees (e.g. '0,180,1,-1').",
    )
    ap.add_argument(
        "--samples-per-tree",
        type=int,
        default=0,
        help="If >0, add this many random (dx,dy,dd) candidates per tree (in addition to the grid).",
    )

    ap.add_argument("--scale", type=int, default=1_000_000, help="Integer scaling factor for bbox-side objective.")
    ap.add_argument("--time-limit-s", type=float, default=3600.0, help="CP-SAT time limit in seconds.")
    ap.add_argument("--workers", type=int, default=8, help="CP-SAT parallel workers.")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for CP-SAT / finalizer.")
    ap.add_argument(
        "--overlap-mode",
        type=str,
        default="strict",
        choices=["strict", "conservative", "kaggle"],
        help="Overlap predicate for final validation/repair.",
    )
    ap.add_argument("--log-progress", action="store_true", help="Enable verbose CP-SAT search logs.")
    ap.add_argument(
        "--hint-full",
        action="store_true",
        help="If set, hints *all* candidate variables (1 for the base pose per tree; 0 otherwise).",
    )

    ns = ap.parse_args(argv)
    if cp_model is None:  # pragma: no cover - optional dependency
        raise SystemExit(f"Missing optional dependency: ortools ({_ORTOOLS_IMPORT_ERROR})\nTry: pip install ortools")

    n = int(ns.n)
    if n <= 0 or n > 200:
        raise SystemExit("--n must be in [1,200]")

    base = load_submission(ns.submission, nmax=max(n, 200))
    if n not in base or base[n].shape != (n, 3):
        raise SystemExit(f"Base submission missing puzzle {n} or wrong shape.")

    points = np.array(TREE_POINTS, dtype=float)
    rad = float(polygon_radius(points))
    base_poses = np.array(base[n], dtype=float, copy=True)
    base_poses = shift_poses_to_origin(points, base_poses)
    base_s = float(packing_score(points, base_poses))
    _eprint(f"Base n={n} score: {base_s:.12f}")

    dxs = _unique_offsets(float(ns.delta_xy), float(ns.step_xy))
    dys = list(dxs)
    rot_deltas = _parse_float_list(str(ns.rot_deltas))
    if not rot_deltas:
        rot_deltas = [0.0]
    # Prefer 0 first for hint.
    rot_deltas.sort(key=lambda v: (abs(v) > 1e-12, abs(v)))

    # --- Build candidates (one set per tree)
    cands: list[_Candidate] = []
    group_indices: list[list[int]] = [[] for _ in range(n)]
    poly_by_idx: list[np.ndarray] = []

    t_build0 = time.time()
    rng = np.random.default_rng(int(ns.seed))
    for t in range(n):
        x0, y0, deg0 = base_poses[t]
        # De-dup per tree (important when mixing grid + random).
        seen_local: set[tuple[float, float, float]] = set()

        def _add_candidate(dx: float, dy: float, dd: float) -> None:
            pose = np.array([x0 + float(dx), y0 + float(dy), float(deg0 + float(dd))], dtype=float)
            key = (round(float(pose[0]), 12), round(float(pose[1]), 12), round(float(pose[2]), 12))
            if key in seen_local:
                return
            seen_local.add(key)
            poly = transform_polygon(points, pose)
            bbox = polygon_bbox(poly)
            idx = len(cands)
            cands.append(_Candidate(tree=t, pose=pose, bbox=bbox))
            poly_by_idx.append(poly)
            group_indices[t].append(idx)

        for dx in dxs:
            for dy in dys:
                for dd in rot_deltas:
                    _add_candidate(dx, dy, dd)

        k_rand = int(ns.samples_per_tree)
        if k_rand > 0:
            # Add random jitter candidates; sample dd from provided discrete set.
            for _ in range(k_rand):
                dx = rng.uniform(-float(ns.delta_xy), float(ns.delta_xy))
                dy = rng.uniform(-float(ns.delta_xy), float(ns.delta_xy))
                dd = float(rng.choice(rot_deltas))
                _add_candidate(dx, dy, dd)

    _eprint(f"Candidates: {len(cands)} ({len(dxs)}x{len(dys)}x{len(rot_deltas)} per tree), build={time.time()-t_build0:.2f}s")

    # Precompute arrays.
    poses = np.stack([c.pose for c in cands], axis=0)
    bboxes = np.stack([c.bbox for c in cands], axis=0)
    trees = np.array([c.tree for c in cands], dtype=int)
    centers = poses[:, 0:2]

    # --- Build conflict graph (overlaps)
    # Two polygons can only overlap if their center distance <= 2*rad (since all vertices lie within radius rad).
    eps = 1e-9
    dist_thr = 2.0 * rad + eps
    thr2 = dist_thr * dist_thr

    # Tree-level pruning: only compare candidate sets of tree pairs that can ever get within 2*rad,
    # given the max translation radius (sqrt(2)*delta_xy) for each tree.
    delta_max = math.sqrt(2.0) * float(ns.delta_xy)
    base_centers = base_poses[:, 0:2]
    neighbor_tree_pairs: list[tuple[int, int]] = []
    for a in range(n):
        for b in range(a + 1, n):
            dx0 = float(base_centers[a, 0] - base_centers[b, 0])
            dy0 = float(base_centers[a, 1] - base_centers[b, 1])
            d0 = math.hypot(dx0, dy0)
            # Minimal possible distance after independent moves within delta_max each.
            if d0 - 2.0 * delta_max > dist_thr:
                continue
            neighbor_tree_pairs.append((a, b))

    t_edges0 = time.time()
    edges: list[tuple[int, int]] = []
    checked = 0
    for a, b in neighbor_tree_pairs:
        ia = group_indices[a]
        ib = group_indices[b]
        for i in ia:
            ci = centers[i]
            bi = bboxes[i]
            for j in ib:
                checked += 1
                dx = float(ci[0] - centers[j, 0])
                dy = float(ci[1] - centers[j, 1])
                if dx * dx + dy * dy > thr2:
                    continue
                if not _aabb_overlaps(bi, bboxes[j], eps):
                    continue
                if polygons_intersect_strict(poly_by_idx[i], poly_by_idx[j]):
                    edges.append((int(i), int(j)))
    _eprint(
        f"Tree-neighbor pairs: {len(neighbor_tree_pairs)}; checked={checked}; edges={len(edges)} (build={time.time()-t_edges0:.2f}s)"
    )

    # --- CP-SAT model
    model = cp_model.CpModel()
    xs = [model.NewBoolVar(f"x[{i}]") for i in range(len(cands))]

    for t in range(n):
        model.Add(sum(xs[i] for i in group_indices[t]) == 1)

    for i, j in edges:
        model.Add(xs[i] + xs[j] <= 1)

    # Envelope objective: minimize max(width, height) of the selected candidate bboxes.
    scale = int(ns.scale)
    if scale <= 0:
        raise SystemExit("--scale must be > 0")
    xmin = np.floor(bboxes[:, 0] * scale).astype(np.int64)
    ymin = np.floor(bboxes[:, 1] * scale).astype(np.int64)
    xmax = np.ceil(bboxes[:, 2] * scale).astype(np.int64)
    ymax = np.ceil(bboxes[:, 3] * scale).astype(np.int64)

    xmin_lo, xmin_hi = int(np.min(xmin)), int(np.max(xmin))
    xmax_lo, xmax_hi = int(np.min(xmax)), int(np.max(xmax))
    ymin_lo, ymin_hi = int(np.min(ymin)), int(np.max(ymin))
    ymax_lo, ymax_hi = int(np.min(ymax)), int(np.max(ymax))

    mx_min = int(xmin_hi - xmin_lo)
    mx_max = int(xmax_hi - xmax_lo)
    my_min = int(ymin_hi - ymin_lo)
    my_max = int(ymax_hi - ymax_lo)

    min_x = model.NewIntVar(xmin_lo, xmin_hi, "min_x")
    max_x = model.NewIntVar(xmax_lo, xmax_hi, "max_x")
    min_y = model.NewIntVar(ymin_lo, ymin_hi, "min_y")
    max_y = model.NewIntVar(ymax_lo, ymax_hi, "max_y")
    side_hi = int(max(xmax_hi - xmin_lo, ymax_hi - ymin_lo))
    side = model.NewIntVar(0, max(0, side_hi), "side")

    for i in range(len(cands)):
        # If selected, min/max must cover this bbox.
        model.Add(min_x <= int(xmin[i]) + mx_min * (1 - xs[i]))
        model.Add(max_x >= int(xmax[i]) - mx_max * (1 - xs[i]))
        model.Add(min_y <= int(ymin[i]) + my_min * (1 - xs[i]))
        model.Add(max_y >= int(ymax[i]) - my_max * (1 - xs[i]))

    w = model.NewIntVar(0, max(0, side_hi), "w")
    h = model.NewIntVar(0, max(0, side_hi), "h")
    model.Add(w == max_x - min_x)
    model.Add(h == max_y - min_y)
    model.AddMaxEquality(side, [w, h])
    model.Minimize(side)

    # Hints: original base poses correspond to dx=dy=dd=0 candidates.
    # Our offset ordering ensures (dx,dy,dd)=(0,0,0) is first in each group.
    # With large candidate sets the solver may take a while to find its first
    # feasible incumbent; a full hint can make this immediate.
    for t in range(n):
        i0 = group_indices[t][0]
        model.AddHint(xs[i0], 1)
        if ns.hint_full:
            for i in group_indices[t][1:]:
                model.AddHint(xs[i], 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(max(0.0, ns.time_limit_s))
    solver.parameters.num_search_workers = int(max(1, ns.workers))
    solver.parameters.random_seed = int(ns.seed)
    if ns.log_progress:
        solver.parameters.log_search_progress = True
        solver.parameters.cp_model_presolve = True

    overlap_mode: OverlapMode = str(ns.overlap_mode)  # type: ignore[assignment]
    saver = _BestSaver(
        xs,
        candidates=cands,
        n=n,
        scale=scale,
        out_dir=Path(ns.out_dir),
        points=points,
        overlap_mode=overlap_mode,
        seed=int(ns.seed),
    )

    _eprint(
        f"Solving CP-SAT (n={n}, vars={len(xs)}, edges={len(edges)}, time_limit={solver.parameters.max_time_in_seconds}s)..."
    )
    t_solve0 = time.time()
    # OR-Tools Python API: Solve(model, callback) calls the callback on each new
    # incumbent for optimization problems.
    status = solver.Solve(model, saver)
    dt = time.time() - t_solve0
    _eprint(f"CP-SAT status: {solver.StatusName(status)} in {dt:.1f}s; solutions={saver.solution_count}")

    if saver.best_poses is None:
        _eprint("No feasible solution found by callback; keeping base poses.")
        best_poses = base[n]
        best_s = float(packing_score(points, shift_poses_to_origin(points, best_poses)))
    else:
        best_poses = saver.best_poses
        best_s = float(packing_score(points, best_poses))

    _eprint(f"Best n={n} score: {best_s:.12f} (delta {best_s - base_s:+.12f})")
    if first_overlap_pair(points, best_poses, mode=overlap_mode) is not None:
        raise RuntimeError("Internal error: best_poses has overlap after finalization.")

    # Write a new submission with puzzle n replaced.
    out_puzzles = {k: np.array(v, dtype=float, copy=True) for k, v in base.items() if 1 <= k <= 200}
    out_puzzles[n] = np.array(best_poses, dtype=float, copy=True)
    _write_submission(ns.out, out_puzzles, nmax=200)
    _eprint(f"Wrote: {ns.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
