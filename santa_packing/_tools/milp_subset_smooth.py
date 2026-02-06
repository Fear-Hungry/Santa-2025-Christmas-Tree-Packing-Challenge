#!/usr/bin/env python3

"""Exact subset-smoothing via MILP (OR-Tools).

This tool tries to improve a strong `submission.csv` by replacing a puzzle `n`
with a *subset* of poses taken from a larger puzzle `m > n` in the same
submission.

Because removing trees from an overlap-free packing cannot introduce overlaps,
we can search over subsets without any collision constraints. The objective is
the puzzle score `s_n`: the side length of the smallest axis-aligned square
containing the packing (AABB square side).

For each candidate source puzzle `m`, we solve the following MILP:

  minimize   side
  s.t.       sum_i z_i = n
             side >= (max_x - min_x)
             side >= (max_y - min_y)
             min_x <= xmin_i + Mx_min*(1 - z_i)
             max_x >= xmax_i - Mx_max*(1 - z_i)
             min_y <= ymin_i + My_min*(1 - z_i)
             max_y >= ymax_i - My_max*(1 - z_i)
             z_i in {0,1}

Where (xmin_i, ymin_i, xmax_i, ymax_i) is the AABB of the i-th tree polygon in
puzzle `m`.

Typical usage (start conservative; increase budgets/windows as needed):

  .venv/bin/python -m santa_packing._tools.milp_subset_smooth \\
    submission.csv --out runs/milp/milp.csv --window 199 --ns 1..200 \\
    --time-limit-s 0.2 --topk-m 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from santa_packing.cli.improve_submission import _write_submission
from santa_packing.geom_np import packing_score, polygon_bbox, transform_polygon
from santa_packing.scoring import load_submission, score_submission
from santa_packing.tree_data import TREE_POINTS

try:
    from ortools.linear_solver import pywraplp  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - optional dependency
    pywraplp = None
    _ORTOOLS_IMPORT_ERROR = exc


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _parse_int_list(text: str) -> list[int]:
    raw = text.strip()
    if not raw:
        return []
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ".." in part:
            a, b = part.split("..", 1)
            start = int(a)
            end = int(b)
            step = 1 if end >= start else -1
            out.extend(range(start, end + step, step))
            continue
        if "-" in part and part.count("-") == 1 and part[0] != "-":
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
            continue
        out.append(int(part))
    return out


def _compute_bboxes(points: np.ndarray, poses: np.ndarray) -> np.ndarray:
    polys = [transform_polygon(points, pose) for pose in poses]
    return np.array([polygon_bbox(p) for p in polys], dtype=float)


def _create_solver(name: str):
    if pywraplp is None:  # pragma: no cover - optional dependency
        raise RuntimeError(f"OR-Tools is required for MILP smoothing: {_ORTOOLS_IMPORT_ERROR}")

    key = name.strip().lower()
    if key in {"cbc", "default"}:
        solver = pywraplp.Solver.CreateSolver("CBC")
    elif key in {"scip"}:
        solver = pywraplp.Solver.CreateSolver("SCIP")
    elif key in {"sat", "cp-sat", "cpsat"}:
        solver = pywraplp.Solver.CreateSolver("SAT")
    else:
        solver = pywraplp.Solver.CreateSolver(name)

    if solver is None:
        raise RuntimeError(f"Failed to create OR-Tools solver '{name}'.")
    return solver


def _milp_select_subset(
    bboxes: np.ndarray,
    *,
    n: int,
    time_limit_s: float,
    threads: int,
    solver_name: str,
) -> tuple[np.ndarray, float, str] | None:
    """Return (indices, side, status) for the best subset of size n, or None if no solution."""
    m = int(bboxes.shape[0])
    n = int(n)
    if n <= 0 or n > m:
        raise ValueError(f"Invalid subset size n={n} for m={m}.")
    if n == m:
        side = float(
            max(
                float(np.max(bboxes[:, 2]) - np.min(bboxes[:, 0])),
                float(np.max(bboxes[:, 3]) - np.min(bboxes[:, 1])),
            )
        )
        return np.arange(m, dtype=int), side, "trivial"

    solver = _create_solver(solver_name)
    if int(threads) > 0:
        solver.SetNumThreads(int(threads))
    if float(time_limit_s) > 0:
        solver.SetTimeLimit(int(float(time_limit_s) * 1000.0))

    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    xmin_lo = float(np.min(xmin))
    xmin_hi = float(np.max(xmin))
    xmax_lo = float(np.min(xmax))
    xmax_hi = float(np.max(xmax))
    ymin_lo = float(np.min(ymin))
    ymin_hi = float(np.max(ymin))
    ymax_lo = float(np.min(ymax))
    ymax_hi = float(np.max(ymax))

    # Tight big-M values given variable bounds.
    mx_min = float(xmin_hi - xmin_lo)
    mx_max = float(xmax_hi - xmax_lo)
    my_min = float(ymin_hi - ymin_lo)
    my_max = float(ymax_hi - ymax_lo)

    # Decision vars.
    z = [solver.BoolVar(f"z[{i}]") for i in range(m)]

    min_x = solver.NumVar(xmin_lo, xmin_hi, "min_x")
    max_x = solver.NumVar(xmax_lo, xmax_hi, "max_x")
    min_y = solver.NumVar(ymin_lo, ymin_hi, "min_y")
    max_y = solver.NumVar(ymax_lo, ymax_hi, "max_y")

    side_hi = float(max(xmax_hi - xmin_lo, ymax_hi - ymin_lo))
    side = solver.NumVar(0.0, side_hi, "side")

    # Cardinality.
    solver.Add(solver.Sum(z) == n)

    # Envelope constraints (activated when selected).
    for i in range(m):
        solver.Add(min_x <= float(xmin[i]) + mx_min * (1.0 - z[i]))
        solver.Add(max_x >= float(xmax[i]) - mx_max * (1.0 - z[i]))
        solver.Add(min_y <= float(ymin[i]) + my_min * (1.0 - z[i]))
        solver.Add(max_y >= float(ymax[i]) - my_max * (1.0 - z[i]))

    solver.Add(min_x <= max_x)
    solver.Add(min_y <= max_y)
    solver.Add(side >= max_x - min_x)
    solver.Add(side >= max_y - min_y)

    solver.Minimize(side)

    status = solver.Solve()
    if pywraplp is None:  # pragma: no cover
        raise RuntimeError("Unexpected: OR-Tools missing after solver creation.")

    if status == pywraplp.Solver.OPTIMAL:
        status_name = "optimal"
    elif status == pywraplp.Solver.FEASIBLE:
        status_name = "feasible"
    else:
        return None

    vals = np.array([float(v.solution_value()) for v in z], dtype=float)
    order = np.argsort(-vals, kind="mergesort")
    idx = np.array(order[:n], dtype=int)
    side_val = float(side.solution_value())
    return idx, side_val, status_name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MILP exact subset smoothing for a submission.csv (OR-Tools).")
    ap.add_argument("submission", type=Path, help="Input submission.csv")
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path")
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle n (default: 200)")
    ap.add_argument("--window", type=int, default=60, help="Lookahead m window (default: 60)")
    ap.add_argument("--ns", type=str, default="1..200", help="Which n to attempt (e.g. '1..120,150').")
    ap.add_argument(
        "--ms",
        type=str,
        default="",
        help="Optional explicit list of source m to consider (overrides --window).",
    )
    ap.add_argument("--topk-m", type=int, default=8, help="Only solve MILP for the best K m by a cheap heuristic.")
    ap.add_argument("--time-limit-s", type=float, default=0.2, help="Time limit per MILP (seconds). 0 disables.")
    ap.add_argument("--threads", type=int, default=1, help="MILP solver threads (default: 1)")
    ap.add_argument("--solver", type=str, default="cbc", help="OR-Tools solver backend (cbc/scip/sat).")
    ap.add_argument("--tol", type=float, default=1e-9, help="Strict improvement tolerance on s_n.")
    ap.add_argument("--verbose", action="store_true", help="Print per-n progress.")

    ns = ap.parse_args(argv)

    nmax = int(ns.nmax)
    if nmax <= 0 or nmax > 200:
        raise SystemExit("--nmax must be in [1,200]")
    window = int(ns.window)
    if window < 0:
        raise SystemExit("--window must be >= 0")

    if pywraplp is None:  # pragma: no cover - optional dependency
        raise SystemExit(f"Missing optional dependency: ortools ({_ORTOOLS_IMPORT_ERROR})\nTry: pip install ortools")

    base = load_submission(ns.submission, nmax=nmax)
    missing = [n for n in range(1, nmax + 1) if n not in base or base[n].shape != (n, 3)]
    if missing:
        raise SystemExit(f"Invalid/missing puzzles: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    points = np.array(TREE_POINTS, dtype=float)

    n_list = _parse_int_list(str(ns.ns))
    n_list = [n for n in n_list if 1 <= n <= nmax]
    if not n_list:
        raise SystemExit("No valid n in --ns.")

    explicit_ms = _parse_int_list(str(ns.ms)) if str(ns.ms).strip() else []
    explicit_ms = [m for m in explicit_ms if 1 <= m <= nmax]

    bbox_cache: dict[int, np.ndarray] = {}
    for m in range(1, nmax + 1):
        bbox_cache[m] = _compute_bboxes(points, base[m])

    out: dict[int, np.ndarray] = {n: np.array(p, dtype=float, copy=True) for n, p in base.items()}

    tol = float(ns.tol)
    total_solves = 0
    improved = 0

    t0 = time.time()
    for n in n_list:
        if n >= nmax:
            continue

        base_s = float(packing_score(points, out[n]))
        best_s = base_s
        best_poses = out[n]
        best_from = None

        if explicit_ms:
            ms = [m for m in explicit_ms if m > n]
        else:
            ms = list(range(n + 1, min(nmax, n + window) + 1))

        if not ms:
            continue

        # Cheap pre-ranking: take the best K ms based on the radial subset heuristic.
        k = max(1, int(ns.topk_m))
        m_scores: list[tuple[float, int]] = []
        for m in ms:
            bboxes_m = bbox_cache[m]
            poses_m = base[m]
            # Center-of-AABB + closest-by-radius (same as improve_submission._radial_subset).
            min_x = float(np.min(bboxes_m[:, 0]))
            min_y = float(np.min(bboxes_m[:, 1]))
            max_x = float(np.max(bboxes_m[:, 2]))
            max_y = float(np.max(bboxes_m[:, 3]))
            center = np.array([(min_x + max_x) * 0.5, (min_y + max_y) * 0.5], dtype=float)
            d = poses_m[:, :2] - center[None, :]
            dist2 = np.sum(d * d, axis=1)
            order = np.argsort(dist2, kind="mergesort")[:n]
            cand = poses_m[order]
            s = float(packing_score(points, cand))
            m_scores.append((s, int(m)))

        m_scores.sort()
        ms_ranked = [m for _, m in m_scores[:k]] if k > 0 else [m for _, m in m_scores]

        for m in ms_ranked:
            bboxes_m = bbox_cache[m]
            solved = _milp_select_subset(
                bboxes_m,
                n=n,
                time_limit_s=float(ns.time_limit_s),
                threads=int(ns.threads),
                solver_name=str(ns.solver),
            )
            total_solves += 1
            if solved is None:
                continue
            idx, _, status = solved
            cand = base[m][idx]
            s = float(packing_score(points, cand))
            if s + tol < best_s:
                best_s = s
                best_poses = np.array(cand, dtype=float, copy=True)
                best_from = (m, status)

        if best_from is not None:
            out[n] = best_poses
            improved += 1
            if ns.verbose:
                m, status = best_from
                _eprint(f"n={n:3d}: {base_s:.9f} -> {best_s:.9f} (from m={m}, {status})")
        elif ns.verbose:
            _eprint(f"n={n:3d}: {base_s:.9f} (no improvement)")

    _write_submission(ns.out, out, nmax=nmax)
    res = score_submission(ns.out, nmax=nmax, overlap_mode="strict")
    dt = time.time() - t0
    _eprint(f"Wrote: {ns.out}")
    _eprint(f"Strict score: {res.score:.12f} (solves={total_solves}, improved_puzzles={improved}, time={dt:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

