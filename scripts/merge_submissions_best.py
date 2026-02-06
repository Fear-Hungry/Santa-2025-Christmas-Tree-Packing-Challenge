#!/usr/bin/env python3
"""Merge multiple submissions by picking the best (lowest s_n) per puzzle.

Usage:
  python scripts/merge_submissions_best.py --base submission.csv --out merged.csv other1.csv other2.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from santa_packing.scoring import load_submission
from santa_packing.geom_np import packing_score
from santa_packing.tree_data import TREE_POINTS
from santa_packing.cli.improve_submission import _write_submission


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge submissions by best per-n packing score.")
    ap.add_argument("--base", type=Path, required=True, help="Base submission.csv")
    ap.add_argument("--out", type=Path, required=True, help="Output submission.csv")
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle n (default: 200)")
    ap.add_argument(
        "--max-smax-delta",
        type=float,
        default=0.0,
        help="Allow s_max to increase by at most this amount (default: 0.0)",
    )
    ap.add_argument("inputs", nargs="+", type=Path, help="Other submissions to merge")
    args = ap.parse_args()

    points = np.array(TREE_POINTS, dtype=float)

    base = load_submission(args.base, nmax=args.nmax)
    inputs = [load_submission(p, nmax=args.nmax) for p in args.inputs]

    out: dict[int, np.ndarray] = {}
    improved = 0

    base_s = {n: float(packing_score(points, base[n])) for n in range(1, args.nmax + 1)}
    base_smax = max(base_s.values())
    smax_limit = float(base_smax + args.max_smax_delta)

    current_smax = base_smax

    for n in range(1, args.nmax + 1):
        best = base[n]
        best_s = base_s[n]
        for sub in inputs:
            cand = sub[n]
            s = float(packing_score(points, cand))
            cand_smax = max(current_smax, s)
            if s < best_s and cand_smax <= smax_limit:
                best = cand
                best_s = s
                improved += 1
        out[n] = best
        if best_s > current_smax:
            current_smax = best_s

    _write_submission(args.out, out, nmax=args.nmax)
    print(f"wrote: {args.out}")
    print(f"improved puzzles: {improved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
