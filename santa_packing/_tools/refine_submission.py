#!/usr/bin/env python3

"""Refine an existing `submission.csv` via per-puzzle local search (parallel).

This tool is meant to improve a strong baseline by applying additional
metaheuristics per puzzle `n`:
- hill-climb (deterministic local search)
- LNS/ALNS (ruin & recreate + optional group moves)
- optional GA (for very small n)

It keeps only strict improvements in `s_n` (after finalization/quantization) and
ensures the output is overlap-free under the requested overlap mode.

Typical usage (focus the highest-weight puzzles first):

  .venv/bin/python -m santa_packing._tools.refine_submission \\
    --base submission.csv --out runs/refine/refined.csv \\
    --ns 1..60 --jobs 16 --repeats 3 \\
    --lns-nmax 60 --lns-passes 200 --lns-destroy-mode alns \\
    --hc-nmax 60 --hc-passes 2
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from santa_packing.cli.improve_submission import _write_submission
from santa_packing.cli.generate_submission import _finalize_puzzle
from santa_packing.geom_np import packing_score
from santa_packing.postopt_np import genetic_optimize, hill_climb, large_neighborhood_search
from santa_packing.scoring import OverlapMode, first_overlap_pair, load_submission, score_submission
from santa_packing.tree_data import TREE_POINTS


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


_POINTS_NP = np.array(TREE_POINTS, dtype=float)


@dataclass(frozen=True)
class _Job:
    n: int
    poses: np.ndarray
    seed: int
    overlap_mode: OverlapMode
    repeats: int
    tol: float

    lns_nmax: int
    lns_passes: int
    lns_destroy_k: int
    lns_destroy_mode: str
    lns_tabu_tenure: int
    lns_candidates: int
    lns_angle_samples: int
    lns_pad_scale: float
    lns_group_moves: int
    lns_group_size: int
    lns_group_trans_sigma: float
    lns_group_rot_sigma: float
    lns_t_start: float
    lns_t_end: float

    ga_nmax: int
    ga_pop: int
    ga_gens: int
    ga_elite_frac: float
    ga_crossover_prob: float
    ga_mut_sigma_xy: float
    ga_mut_sigma_deg: float
    ga_directed_prob: float
    ga_directed_step_xy: float
    ga_directed_k: int
    ga_repair_iters: int
    ga_hc_passes: int
    ga_hc_step_xy: float
    ga_hc_step_deg: float

    hc_nmax: int
    hc_passes: int
    hc_step_xy: float
    hc_step_deg: float


@dataclass(frozen=True)
class _JobResult:
    n: int
    base_score: float
    best_score: float
    improved: bool
    poses: np.ndarray
    repeats_done: int
    elapsed_s: float


def _refine_one(job: _Job) -> _JobResult:
    t0 = time.time()
    n = int(job.n)
    poses = np.array(job.poses, dtype=float, copy=True)
    if poses.shape != (n, 3):
        raise ValueError(f"Puzzle {n}: expected shape {(n,3)} got {poses.shape}")

    # Always start from a finalized/quantized baseline so "improvement" reflects
    # what will actually be written to CSV (post-quantization).
    base = _finalize_puzzle(
        _POINTS_NP,
        poses,
        seed=int(job.seed) + 1_000_003 * n,
        puzzle_n=n,
        overlap_mode=str(job.overlap_mode),
    )
    if first_overlap_pair(_POINTS_NP, base, mode=str(job.overlap_mode)) is not None:
        raise ValueError(f"Puzzle {n}: base overlaps after finalize (mode={job.overlap_mode})")

    base_score = float(packing_score(_POINTS_NP, base))
    best = base
    best_score = base_score
    repeats_done = 0

    for rep in range(int(job.repeats)):
        repeats_done += 1
        seed = int(job.seed) + 1_000_003 * n + 97 * rep
        cand = np.array(best, dtype=float, copy=True)

        if int(job.lns_passes) > 0 and n <= int(job.lns_nmax):
            cand = large_neighborhood_search(
                _POINTS_NP,
                cand,
                seed=seed + 11_000,
                passes=int(job.lns_passes),
                destroy_k=int(job.lns_destroy_k),
                destroy_mode=str(job.lns_destroy_mode),
                tabu_tenure=int(job.lns_tabu_tenure),
                candidates=int(job.lns_candidates),
                angle_samples=int(job.lns_angle_samples),
                pad_scale=float(job.lns_pad_scale),
                group_moves=int(job.lns_group_moves),
                group_size=int(job.lns_group_size),
                group_trans_sigma=float(job.lns_group_trans_sigma),
                group_rot_sigma=float(job.lns_group_rot_sigma),
                t_start=float(job.lns_t_start),
                t_end=float(job.lns_t_end),
                overlap_mode=str(job.overlap_mode),
            )

        if int(job.ga_gens) > 0 and n <= int(job.ga_nmax):
            cand = genetic_optimize(
                _POINTS_NP,
                [cand],
                seed=seed + 21_000,
                pop_size=int(job.ga_pop),
                generations=int(job.ga_gens),
                elite_frac=float(job.ga_elite_frac),
                crossover_prob=float(job.ga_crossover_prob),
                mutation_sigma_xy=float(job.ga_mut_sigma_xy),
                mutation_sigma_deg=float(job.ga_mut_sigma_deg),
                directed_mut_prob=float(job.ga_directed_prob),
                directed_step_xy=float(job.ga_directed_step_xy),
                directed_k=int(job.ga_directed_k),
                repair_iters=int(job.ga_repair_iters),
                hill_climb_passes=int(job.ga_hc_passes),
                hill_climb_step_xy=float(job.ga_hc_step_xy),
                hill_climb_step_deg=float(job.ga_hc_step_deg),
                overlap_mode=str(job.overlap_mode),
            )

        if int(job.hc_passes) > 0 and n <= int(job.hc_nmax):
            cand = hill_climb(
                _POINTS_NP,
                cand,
                step_xy=float(job.hc_step_xy),
                step_deg=float(job.hc_step_deg),
                max_passes=int(job.hc_passes),
                overlap_mode=str(job.overlap_mode),
            )

        cand = _finalize_puzzle(
            _POINTS_NP,
            cand,
            seed=seed + 13_000,
            puzzle_n=n,
            overlap_mode=str(job.overlap_mode),
        )
        if first_overlap_pair(_POINTS_NP, cand, mode=str(job.overlap_mode)) is not None:
            # Should be rare: finalize_puzzle already tries to repair.
            continue

        s = float(packing_score(_POINTS_NP, cand))
        if s + float(job.tol) < best_score:
            best = cand
            best_score = s

    t1 = time.time()
    return _JobResult(
        n=n,
        base_score=base_score,
        best_score=best_score,
        improved=bool(best_score + float(job.tol) < base_score),
        poses=best,
        repeats_done=repeats_done,
        elapsed_s=t1 - t0,
    )


def _write_report(path: Path, results: list[_JobResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["puzzle", "base_s", "best_s", "improved", "delta_s", "elapsed_s", "repeats"])
        for r in sorted(results, key=lambda x: x.n):
            w.writerow(
                [
                    int(r.n),
                    f"{float(r.base_score):.17f}",
                    f"{float(r.best_score):.17f}",
                    int(bool(r.improved)),
                    f"{float(r.best_score - r.base_score):.17f}",
                    f"{float(r.elapsed_s):.3f}",
                    int(r.repeats_done),
                ]
            )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description="Refine an existing submission.csv via per-puzzle local search (parallel).")
    ap.add_argument("--base", type=Path, required=True, help="Base submission.csv to refine.")
    ap.add_argument("--out", type=Path, required=True, help="Output submission.csv.")
    ap.add_argument("--report", type=Path, default=None, help="Optional CSV report path.")
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle n (default 200).")
    ap.add_argument("--ns", type=str, default="", help="Puzzle list like '1..60,80,200' (empty = all).")
    ap.add_argument("--jobs", type=int, default=16, help="Parallel workers (default 16).")
    ap.add_argument("--seed", type=int, default=123, help="Base RNG seed (default 123).")
    ap.add_argument("--repeats", type=int, default=2, help="Independent repeats per puzzle (default 2).")
    ap.add_argument("--tol", type=float, default=1e-12, help="Improvement tolerance on s_n (default 1e-12).")
    ap.add_argument(
        "--overlap-mode",
        type=str,
        default="kaggle",
        choices=["strict", "conservative", "kaggle"],
        help="Overlap predicate for validation/finalization (default: kaggle).",
    )

    # LNS options
    ap.add_argument("--lns-nmax", type=int, default=60, help="Apply LNS for n <= this (default 60; 0 disables).")
    ap.add_argument("--lns-passes", type=int, default=150, help="LNS passes (default 150; 0 disables).")
    ap.add_argument("--lns-destroy-k", type=int, default=6, help="Removed trees per ruin pass (default 6).")
    ap.add_argument(
        "--lns-destroy-mode",
        type=str,
        default="alns",
        choices=["mixed", "boundary", "random", "cluster", "alns"],
        help="Ruin-set sampling strategy (default alns).",
    )
    ap.add_argument("--lns-tabu-tenure", type=int, default=20, help="Tabu tenure for ruin-sets (default 20).")
    ap.add_argument("--lns-candidates", type=int, default=1200, help="Reinsert center samples (default 1200).")
    ap.add_argument("--lns-angle-samples", type=int, default=24, help="Reinsert angle samples (default 24).")
    ap.add_argument("--lns-pad-scale", type=float, default=0.15, help="Sampling padding scale in radii (default 0.15).")
    ap.add_argument("--lns-group-moves", type=int, default=3, help="Group moves per pass (default 3).")
    ap.add_argument("--lns-group-size", type=int, default=10, help="Group size for group moves (default 10).")
    ap.add_argument("--lns-group-trans-sigma", type=float, default=0.03, help="Group translation sigma (default 0.03).")
    ap.add_argument("--lns-group-rot-sigma", type=float, default=8.0, help="Group rotation sigma in deg (default 8).")
    ap.add_argument("--lns-t-start", type=float, default=0.0, help="If >0, enable SA acceptance in LNS (start T).")
    ap.add_argument("--lns-t-end", type=float, default=0.0, help="End temperature for LNS SA acceptance.")

    # GA options (disabled by default; enable for small n only)
    ap.add_argument("--ga-nmax", type=int, default=0, help="Apply GA for n <= this (default 0=disabled).")
    ap.add_argument("--ga-pop", type=int, default=128, help="GA population size (default 128).")
    ap.add_argument("--ga-gens", type=int, default=40, help="GA generations (default 40).")
    ap.add_argument("--ga-elite-frac", type=float, default=0.25, help="GA elite fraction (default 0.25).")
    ap.add_argument("--ga-crossover-prob", type=float, default=0.3, help="GA crossover probability (default 0.3).")
    ap.add_argument("--ga-mut-sigma-xy", type=float, default=0.04, help="GA mutation sigma xy (default 0.04).")
    ap.add_argument("--ga-mut-sigma-deg", type=float, default=10.0, help="GA mutation sigma deg (default 10).")
    ap.add_argument("--ga-directed-prob", type=float, default=0.4, help="GA directed mutation prob (default 0.4).")
    ap.add_argument("--ga-directed-step-xy", type=float, default=0.03, help="GA directed step xy (default 0.03).")
    ap.add_argument("--ga-directed-k", type=int, default=6, help="GA directed neighborhood size (default 6).")
    ap.add_argument("--ga-repair-iters", type=int, default=1500, help="Repair iterations budget (default 1500).")
    ap.add_argument("--ga-hc-passes", type=int, default=1, help="Hill-climb passes on children (default 1).")
    ap.add_argument("--ga-hc-step-xy", type=float, default=0.01, help="Hill-climb step xy for GA (default 0.01).")
    ap.add_argument("--ga-hc-step-deg", type=float, default=2.0, help="Hill-climb step deg for GA (default 2).")

    # Hill-climb options
    ap.add_argument("--hc-nmax", type=int, default=60, help="Apply hill-climb for n <= this (default 60; 0 disables).")
    ap.add_argument("--hc-passes", type=int, default=2, help="Hill-climb passes (default 2; 0 disables).")
    ap.add_argument("--hc-step-xy", type=float, default=0.01, help="Hill-climb translation step (default 0.01).")
    ap.add_argument("--hc-step-deg", type=float, default=2.0, help="Hill-climb rotation step in deg (default 2).")

    ns = ap.parse_args(argv)

    nmax = int(ns.nmax)
    if nmax <= 0 or nmax > 200:
        raise SystemExit("--nmax must be in [1,200]")
    if int(ns.jobs) <= 0:
        raise SystemExit("--jobs must be > 0")
    if int(ns.repeats) <= 0:
        raise SystemExit("--repeats must be > 0")

    overlap_mode: OverlapMode = str(ns.overlap_mode)  # type: ignore[assignment]

    puzzles = load_submission(ns.base, nmax=nmax)
    missing = [n for n in range(1, nmax + 1) if n not in puzzles or puzzles[n].shape != (n, 3)]
    if missing:
        raise SystemExit(f"--base is missing puzzles or wrong shape: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    targets = _parse_int_list(str(ns.ns))
    if targets:
        targets = [n for n in targets if 1 <= n <= nmax]
        if not targets:
            raise SystemExit("--ns did not select any puzzles within [1..nmax]")
    else:
        targets = list(range(1, nmax + 1))

    jobs: list[_Job] = []
    for n in targets:
        jobs.append(
            _Job(
                n=int(n),
                poses=puzzles[int(n)],
                seed=int(ns.seed),
                overlap_mode=overlap_mode,
                repeats=int(ns.repeats),
                tol=float(ns.tol),
                lns_nmax=int(ns.lns_nmax),
                lns_passes=int(ns.lns_passes),
                lns_destroy_k=int(ns.lns_destroy_k),
                lns_destroy_mode=str(ns.lns_destroy_mode),
                lns_tabu_tenure=int(ns.lns_tabu_tenure),
                lns_candidates=int(ns.lns_candidates),
                lns_angle_samples=int(ns.lns_angle_samples),
                lns_pad_scale=float(ns.lns_pad_scale),
                lns_group_moves=int(ns.lns_group_moves),
                lns_group_size=int(ns.lns_group_size),
                lns_group_trans_sigma=float(ns.lns_group_trans_sigma),
                lns_group_rot_sigma=float(ns.lns_group_rot_sigma),
                lns_t_start=float(ns.lns_t_start),
                lns_t_end=float(ns.lns_t_end),
                ga_nmax=int(ns.ga_nmax),
                ga_pop=int(ns.ga_pop),
                ga_gens=int(ns.ga_gens),
                ga_elite_frac=float(ns.ga_elite_frac),
                ga_crossover_prob=float(ns.ga_crossover_prob),
                ga_mut_sigma_xy=float(ns.ga_mut_sigma_xy),
                ga_mut_sigma_deg=float(ns.ga_mut_sigma_deg),
                ga_directed_prob=float(ns.ga_directed_prob),
                ga_directed_step_xy=float(ns.ga_directed_step_xy),
                ga_directed_k=int(ns.ga_directed_k),
                ga_repair_iters=int(ns.ga_repair_iters),
                ga_hc_passes=int(ns.ga_hc_passes),
                ga_hc_step_xy=float(ns.ga_hc_step_xy),
                ga_hc_step_deg=float(ns.ga_hc_step_deg),
                hc_nmax=int(ns.hc_nmax),
                hc_passes=int(ns.hc_passes),
                hc_step_xy=float(ns.hc_step_xy),
                hc_step_deg=float(ns.hc_step_deg),
            )
        )

    _eprint(f"Refining puzzles: {len(jobs)}/{nmax} jobs (jobs={ns.jobs}, repeats={ns.repeats}, mode={overlap_mode})")

    results: list[_JobResult] = []
    improved = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=int(ns.jobs)) as ex:
        futs = {ex.submit(_refine_one, job): job.n for job in jobs}
        for fut in as_completed(futs):
            n = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:
                raise SystemExit(f"Puzzle {n} failed: {exc}") from exc
            results.append(r)
            if r.improved:
                improved += 1
            _eprint(
                f"[n={r.n:3d}] s {r.base_score:.12f} -> {r.best_score:.12f}  "
                f"({'improved' if r.improved else 'same'})  time={r.elapsed_s:.1f}s"
            )

    t1 = time.time()
    _eprint(f"Done: improved {improved}/{len(results)} puzzles in {t1 - t0:.1f}s")

    # Build output mapping: keep base for untouched puzzles; replace targets with refined poses.
    out_puzzles: dict[int, np.ndarray] = {n: np.array(puzzles[n], dtype=float, copy=True) for n in range(1, nmax + 1)}
    for r in results:
        out_puzzles[int(r.n)] = np.array(r.poses, dtype=float, copy=True)

    _write_submission(ns.out, out_puzzles, nmax=nmax)

    if ns.report is not None:
        _write_report(ns.report, results)

    # Validate and print score.
    res = score_submission(ns.out, nmax=nmax, overlap_mode=overlap_mode, check_overlap=True, require_complete=True)
    print(str(ns.out))
    print(f"score({overlap_mode}): {res.score:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
