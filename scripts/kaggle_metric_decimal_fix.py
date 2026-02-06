#!/usr/bin/env python3
"""Fix Kaggle metric overlaps by micro-adjusting decimal strings.

This script operates at the string/Decimal level (1e-17 increments) to avoid
Shapely's microscopic overlap detections without materially changing the score.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from decimal import Decimal, getcontext
from pathlib import Path

from shapely.strtree import STRtree

from santa_packing._tools.kaggle_autofix_submission import _build_kaggle_metric


def _overlap_pairs(metric, xs, ys, degs):
    polys = [metric.ChristmasTree(x, y, d).polygon for x, y, d in zip(xs, ys, degs)]
    tree = STRtree(polys)
    pairs = []
    for i, poly in enumerate(polys):
        for j in tree.query(poly):
            j = int(j)
            if j <= i:
                continue
            other = polys[j]
            if poly.intersects(other) and not poly.touches(other):
                pairs.append((i, j))
    return pairs


def _format_decimal(val: Decimal) -> str:
    return f"{val:.17f}"


def _try_reduce(metric, xs, ys, degs):
    step_sizes = [
        Decimal("1e-17"),
        Decimal("1e-16"),
        Decimal("1e-15"),
        Decimal("1e-14"),
        Decimal("1e-13"),
        Decimal("1e-12"),
    ]
    mults = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    max_passes = 50

    for _ in range(max_passes):
        pairs = _overlap_pairs(metric, xs, ys, degs)
        if not pairs:
            return True
        cur_count = len(pairs)
        improved = False

        # Try single-coordinate micro tweaks on the first few overlapping pairs.
        for (i, j) in pairs[:4]:
            for idx in (i, j):
                for coord in ("x", "y", "deg"):
                    for step in step_sizes:
                        for mult in mults:
                            for sign in (1, -1):
                                xs2, ys2, d2 = xs[:], ys[:], degs[:]
                                if coord == "x":
                                    val = Decimal(xs2[idx]) + sign * step * mult
                                    xs2[idx] = _format_decimal(val)
                                elif coord == "y":
                                    val = Decimal(ys2[idx]) + sign * step * mult
                                    ys2[idx] = _format_decimal(val)
                                else:
                                    val = Decimal(d2[idx]) + sign * step * mult
                                    d2[idx] = _format_decimal(val)

                                new_pairs = _overlap_pairs(metric, xs2, ys2, d2)
                                if not new_pairs:
                                    xs[:] = xs2
                                    ys[:] = ys2
                                    degs[:] = d2
                                    return True
                                if len(new_pairs) < cur_count:
                                    xs[:] = xs2
                                    ys[:] = ys2
                                    degs[:] = d2
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if improved:
            continue

        # Try diagonal x+y tweaks (same sign) if single coord didn't help.
        for (i, j) in pairs[:4]:
            for idx in (i, j):
                for step in step_sizes:
                    for mult in mults:
                        for sign in (1, -1):
                            xs2, ys2, d2 = xs[:], ys[:], degs[:]
                            valx = Decimal(xs2[idx]) + sign * step * mult
                            valy = Decimal(ys2[idx]) + sign * step * mult
                            xs2[idx] = _format_decimal(valx)
                            ys2[idx] = _format_decimal(valy)
                            new_pairs = _overlap_pairs(metric, xs2, ys2, d2)
                            if not new_pairs:
                                xs[:] = xs2
                                ys[:] = ys2
                                degs[:] = d2
                                return True
                            if len(new_pairs) < cur_count:
                                xs[:] = xs2
                                ys[:] = ys2
                                degs[:] = d2
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            return False

    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Fix Kaggle metric overlaps via decimal micro-adjustments.")
    ap.add_argument("submission", type=Path, help="Input submission.csv")
    ap.add_argument("--out", type=Path, required=True, help="Output submission.csv")
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle n (default: 200)")
    args = ap.parse_args()

    getcontext().prec = 50
    metric = _build_kaggle_metric()

    rows = []
    by_n = defaultdict(list)
    with args.submission.open() as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            rows.append(row)
            pid = int(row["id"].split("_")[0])
            if pid <= args.nmax:
                by_n[pid].append(idx)

    failed = []
    fixed = 0

    for n in range(1, args.nmax + 1):
        idxs = by_n.get(n)
        if not idxs:
            continue
        xs = [rows[i]["x"][1:] for i in idxs]
        ys = [rows[i]["y"][1:] for i in idxs]
        degs = [rows[i]["deg"][1:] for i in idxs]

        if _overlap_pairs(metric, xs, ys, degs):
            ok = _try_reduce(metric, xs, ys, degs)
            if not ok:
                failed.append(n)
            else:
                fixed += 1

        # write back (with prefix)
        for k, row_idx in enumerate(idxs):
            rows[row_idx]["x"] = "s" + xs[k]
            rows[row_idx]["y"] = "s" + ys[k]
            rows[row_idx]["deg"] = "s" + degs[k]

    if failed:
        print(f"Failed puzzles: {failed}")

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "x", "y", "deg"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote: {args.out}")
    print(f"fixed puzzles: {fixed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
