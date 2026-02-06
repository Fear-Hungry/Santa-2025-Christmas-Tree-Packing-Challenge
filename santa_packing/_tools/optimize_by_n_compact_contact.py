#!/usr/bin/env python3

"""Optimize puzzles by `n` using `bin/compact_contact --target-n`.

This tool is meant for focused optimization on a subset of puzzles (e.g. medium
`n` range) without spending time re-optimizing the whole dataset.

It runs `compact_contact` multiple times per puzzle `n` (multi-seed), keeps the
best overlap-free packing for that puzzle, and writes a blended submission.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from santa_packing.cli.generate_submission import _finalize_puzzle
from santa_packing.cli.improve_submission import _write_submission
from santa_packing.geom_np import packing_score
from santa_packing.scoring import OverlapMode, first_overlap_pair, load_submission, score_submission
from santa_packing.submission_format import fit_xy_in_bounds, format_submission_value, quantize_for_submission
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


_POINTS = np.array(TREE_POINTS, dtype=float)


def _canonicalize_poses(poses: np.ndarray) -> np.ndarray:
    poses = np.array(poses, dtype=float, copy=True)
    if poses.shape[0] == 0:
        return poses
    poses[:, 2] = np.mod(poses[:, 2], 360.0)
    poses = fit_xy_in_bounds(poses)
    poses = quantize_for_submission(poses)
    return poses


def _load_one_puzzle(csv_path: Path, *, n: int) -> np.ndarray:
    """Load only puzzle `n` from a submission CSV (fast path)."""
    def _parse_val(value: str) -> float:
        value = value.strip()
        if value.startswith("s") or value.startswith("S"):
            value = value[1:]
        return float(value)

    rows: list[list[float]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("id", "")
            try:
                puzzle = int(pid.split("_", 1)[0])
            except Exception:
                continue
            if puzzle < n:
                continue
            if puzzle > n:
                break
            rows.append([_parse_val(row["x"]), _parse_val(row["y"]), _parse_val(row["deg"])])
    if len(rows) != n:
        raise ValueError(f"{csv_path}: expected {n} rows for puzzle {n}, got {len(rows)}")
    return np.array(rows, dtype=float)


@dataclass(frozen=True)
class _BestN:
    n: int
    base_s: float
    best_s: float
    improved: bool
    best_seed: int | None
    poses: np.ndarray
    elapsed_s: float


def _load_base_text_and_offsets(*, base_csv: Path, nmax: int) -> tuple[str, dict[int, tuple[int, int]]]:
    """Load base CSV as text and return byte offsets for each puzzle block.

    This is used to quickly write "temporary base CSVs" for restart runs by
    replacing only the lines for a single puzzle `n` while leaving all other
    lines untouched.
    """
    text = base_csv.read_text(encoding="utf-8")
    offsets: dict[int, tuple[int, int]] = {}

    # Compute [start,end) offsets (in Python string indices) for each puzzle.
    # Assumes standard submission ordering where each puzzle's rows are contiguous
    # and puzzles appear in increasing order (1..nmax).
    lines = text.splitlines(keepends=True)
    if not lines:
        raise ValueError(f"{base_csv}: empty file")

    idx = len(lines[0])  # header line
    current_puzzle: int | None = None
    start: int | None = None

    for line in lines[1:]:
        if not line.strip():
            idx += len(line)
            continue
        pid = line.split(",", 1)[0]
        try:
            puzzle = int(pid.split("_", 1)[0])
        except Exception:
            idx += len(line)
            continue

        if current_puzzle is None:
            current_puzzle = puzzle
            start = idx
        elif puzzle != current_puzzle:
            assert start is not None
            offsets[int(current_puzzle)] = (int(start), int(idx))
            current_puzzle = puzzle
            start = idx

        idx += len(line)

    if current_puzzle is not None:
        assert start is not None
        offsets[int(current_puzzle)] = (int(start), int(idx))

    missing = [n for n in range(1, int(nmax) + 1) if n not in offsets]
    if missing:
        raise ValueError(f"{base_csv}: could not locate puzzle blocks for: {missing[:10]}")
    return text, offsets


def _render_puzzle_block(n: int, poses: np.ndarray) -> str:
    poses = np.array(poses, dtype=float, copy=False)
    if poses.shape != (int(n), 3):
        raise ValueError(f"render_puzzle_block: expected shape {(int(n),3)} got {poses.shape}")
    parts: list[str] = []
    for i in range(int(n)):
        x = format_submission_value(float(poses[i, 0]))
        y = format_submission_value(float(poses[i, 1]))
        deg = format_submission_value(float(poses[i, 2]))
        parts.append(f"{int(n):03d}_{int(i)},{x},{y},{deg}\n")
    return "".join(parts)


def _write_temp_base_for_n(
    *,
    base_text: str,
    offsets: dict[int, tuple[int, int]],
    n: int,
    poses: np.ndarray,
    out_csv: Path,
) -> None:
    start, end = offsets[int(n)]
    block = _render_puzzle_block(int(n), poses)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(base_text[:start])
        f.write(block)
        f.write(base_text[end:])


def _perturb_restart(
    poses: np.ndarray,
    rng: np.random.Generator,
    *,
    scale_min: float,
    scale_max: float,
    jitter_xy: float,
    jitter_deg: float,
    ruin_k: int,
    ruin_box_scale: float,
    base_side: float,
) -> np.ndarray:
    cand = np.array(poses, dtype=float, copy=True)
    n = int(cand.shape[0])

    if int(ruin_k) > 0:
        k = min(int(ruin_k), n)
        idx = rng.choice(n, size=k, replace=False)
        center = np.mean(cand[:, 0:2], axis=0)
        half = 0.5 * float(base_side) * float(ruin_box_scale)
        cand[idx, 0] = center[0] + rng.uniform(-half, half, size=(k,))
        cand[idx, 1] = center[1] + rng.uniform(-half, half, size=(k,))
        cand[idx, 2] = rng.uniform(0.0, 360.0, size=(k,))

    smin = float(scale_min)
    smax = float(scale_max)
    if smin <= 0.0 or smax <= 0.0:
        raise ValueError("--restart-scale-min/max must be > 0")
    if smax < smin:
        smin, smax = smax, smin
    if smax != 1.0 or smin != 1.0:
        scale = float(rng.uniform(smin, smax))
        center = np.mean(cand[:, 0:2], axis=0)
        cand[:, 0:2] = center[None, :] + (cand[:, 0:2] - center[None, :]) * scale

    if float(jitter_xy) > 0.0:
        cand[:, 0:2] += rng.normal(0.0, float(jitter_xy), size=(n, 2))
    if float(jitter_deg) > 0.0:
        cand[:, 2] += rng.normal(0.0, float(jitter_deg), size=(n,))

    cand[:, 2] = np.mod(cand[:, 2], 360.0)
    cand = fit_xy_in_bounds(cand)
    cand = quantize_for_submission(cand)
    return cand


def _run_compact_contact_target_n(
    *,
    root: Path,
    base_csv: Path,
    out_csv: Path,
    n: int,
    nmax: int,
    seed: int,
    timeout_s: float | None,
    passes: int,
    attempts_per_pass: int,
    patience: int,
    boundary_topk: int,
    push_bisect_iters: int,
    push_max_step_frac: float,
    plateau_eps: float,
    diag_frac: float,
    diag_rand: float,
    center_bias: float,
    interior_prob: float,
    shake_pos: float,
    shake_rot_deg: float,
    shake_prob: float,
    quantize_decimals: int,
) -> None:
    cmd = [
        str((root / "bin" / "compact_contact").resolve()),
        "--base",
        str(base_csv),
        "--out",
        str(out_csv),
        "--n-min",
        "1",
        "--n-max",
        str(int(nmax)),
        "--target-n",
        str(int(n)),
        "--seed",
        str(int(seed)),
        "--passes",
        str(int(passes)),
        "--attempts-per-pass",
        str(int(attempts_per_pass)),
        "--patience",
        str(int(patience)),
        "--boundary-topk",
        str(int(boundary_topk)),
        "--push-bisect-iters",
        str(int(push_bisect_iters)),
        "--push-max-step-frac",
        str(float(push_max_step_frac)),
        "--plateau-eps",
        str(float(plateau_eps)),
        "--diag-frac",
        str(float(diag_frac)),
        "--diag-rand",
        str(float(diag_rand)),
        "--center-bias",
        str(float(center_bias)),
        "--interior-prob",
        str(float(interior_prob)),
        "--shake-pos",
        str(float(shake_pos)),
        "--shake-rot-deg",
        str(float(shake_rot_deg)),
        "--shake-prob",
        str(float(shake_prob)),
        "--quantize-decimals",
        str(int(quantize_decimals)),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"compact_contact failed: n={n} seed={seed} rc={proc.returncode}\n{proc.stdout}")


def _optimize_one_n(
    *,
    root: Path,
    base_csv: Path,
    base_text: str,
    base_offsets: dict[int, tuple[int, int]],
    base_poses: np.ndarray,
    n: int,
    nmax: int,
    seeds: list[int],
    restarts: int,
    restart_seed: int,
    restart_scale_min: float,
    restart_scale_max: float,
    restart_jitter_xy: float,
    restart_jitter_deg: float,
    restart_ruin_k: int,
    restart_ruin_box_scale: float,
    restart_allow_overlap: bool,
    restart_attempts: int,
    finalize_candidates: bool,
    overlap_mode: OverlapMode,
    tmp_dir: Path,
    keep_logs: bool,
    timeout_s: float | None,
    passes: int,
    attempts_per_pass: int,
    patience: int,
    boundary_topk: int,
    push_bisect_iters: int,
    push_max_step_frac: float,
    plateau_eps: float,
    diag_frac: float,
    diag_rand: float,
    center_bias: float,
    interior_prob: float,
    shake_pos: float,
    shake_rot_deg: float,
    shake_prob: float,
    quantize_decimals: int,
    tol: float,
) -> _BestN:
    t0 = time.time()
    base_q = _canonicalize_poses(base_poses)
    if first_overlap_pair(_POINTS, base_q, mode=overlap_mode) is not None:
        raise ValueError(f"Base overlaps for puzzle n={n} (mode={overlap_mode}).")

    base_s = float(packing_score(_POINTS, base_q))
    best_s = base_s
    best_pose = base_q
    best_seed: int | None = None

    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tmp_dir / f"cand_n{int(n):03d}.csv"

    restarts = max(1, int(restarts))
    for restart in range(restarts):
        run_base = base_csv
        tmp_base: Path | None = None

        if restart != 0:
            tmp_base = tmp_dir / f"restart{int(restart):03d}_base.csv"
            rng0 = np.random.default_rng(int(restart_seed) + 1_000_003 * int(n) + 97 * int(restart))
            perturbed: np.ndarray | None = None
            for attempt in range(max(1, int(restart_attempts))):
                rng = np.random.default_rng(int(rng0.integers(0, 2**63 - 1)))
                try:
                    cand0 = _perturb_restart(
                        base_q,
                        rng,
                        scale_min=float(restart_scale_min),
                        scale_max=float(restart_scale_max),
                        jitter_xy=float(restart_jitter_xy),
                        jitter_deg=float(restart_jitter_deg),
                        ruin_k=int(restart_ruin_k),
                        ruin_box_scale=float(restart_ruin_box_scale),
                        base_side=float(base_s),
                    )
                except Exception:
                    continue
                if not bool(restart_allow_overlap):
                    if first_overlap_pair(_POINTS, cand0, mode=overlap_mode) is not None:
                        continue
                perturbed = cand0
                break
            if perturbed is None:
                perturbed = base_q

            _write_temp_base_for_n(
                base_text=base_text,
                offsets=base_offsets,
                n=int(n),
                poses=perturbed,
                out_csv=tmp_base,
            )
            run_base = tmp_base

        for seed in seeds:
            try:
                _run_compact_contact_target_n(
                    root=root,
                    base_csv=run_base,
                    out_csv=out_csv,
                    n=n,
                    nmax=nmax,
                    seed=int(seed) + 10_000_019 * int(restart),
                    timeout_s=timeout_s,
                    passes=passes,
                    attempts_per_pass=attempts_per_pass,
                    patience=patience,
                    boundary_topk=boundary_topk,
                    push_bisect_iters=push_bisect_iters,
                    push_max_step_frac=push_max_step_frac,
                    plateau_eps=plateau_eps,
                    diag_frac=diag_frac,
                    diag_rand=diag_rand,
                    center_bias=center_bias,
                    interior_prob=interior_prob,
                    shake_pos=shake_pos,
                    shake_rot_deg=shake_rot_deg,
                    shake_prob=shake_prob,
                    quantize_decimals=quantize_decimals,
                )
            except Exception as exc:
                if keep_logs:
                    (tmp_dir / f"cand_n{int(n):03d}_r{int(restart):03d}_seed{int(seed)}.err.txt").write_text(
                        str(exc), encoding="utf-8"
                    )
                continue

            try:
                cand = _load_one_puzzle(out_csv, n=n)
            except Exception as exc:
                if keep_logs:
                    (tmp_dir / f"cand_n{int(n):03d}_r{int(restart):03d}_seed{int(seed)}.parse.txt").write_text(
                        str(exc), encoding="utf-8"
                    )
                continue

            cand_q = _canonicalize_poses(cand)
            if first_overlap_pair(_POINTS, cand_q, mode=overlap_mode) is not None:
                if not bool(finalize_candidates):
                    continue
                cand_q = _finalize_puzzle(
                    _POINTS,
                    cand_q,
                    seed=int(seed) + 7_700_000 * int(n) + 97 * int(restart),
                    puzzle_n=int(n),
                    overlap_mode=str(overlap_mode),
                )
                cand_q = _canonicalize_poses(cand_q)
                if first_overlap_pair(_POINTS, cand_q, mode=overlap_mode) is not None:
                    continue

            s = float(packing_score(_POINTS, cand_q))
            if s + float(tol) < best_s:
                best_s = s
                best_pose = cand_q
                best_seed = int(seed) + 10_000_019 * int(restart)

        if tmp_base is not None and not keep_logs:
            try:
                tmp_base.unlink()
            except Exception:
                pass

    t1 = time.time()
    return _BestN(
        n=int(n),
        base_s=base_s,
        best_s=best_s,
        improved=bool(best_s + float(tol) < base_s),
        best_seed=best_seed,
        poses=best_pose,
        elapsed_s=t1 - t0,
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description="Per-n optimization via bin/compact_contact --target-n.")
    ap.add_argument("--base", type=Path, required=True, help="Base submission.csv")
    ap.add_argument("--out", type=Path, required=True, help="Output submission.csv")
    ap.add_argument("--report", type=Path, default=None, help="Optional per-n report CSV")
    ap.add_argument("--nmax", type=int, default=200, help="Max n (default 200)")
    ap.add_argument("--ns", type=str, default="", help="Puzzle list like '50..150,200' (empty=all)")
    ap.add_argument("--seeds", type=str, default="1..8", help="Seed list like '1..32' or '1,2,3'")
    ap.add_argument("--jobs", type=int, default=16, help="Parallel puzzles (default 16)")
    ap.add_argument("--tol", type=float, default=1e-12, help="Improvement tolerance on s_n (default 1e-12)")
    ap.add_argument(
        "--overlap-mode",
        type=str,
        default="strict",
        choices=["strict", "conservative", "kaggle"],
        help="Overlap predicate to enforce (default strict).",
    )
    ap.add_argument("--keep-logs", action="store_true", help="Keep per-n error logs in <out_dir>/tmp/")
    ap.add_argument(
        "--cc-timeout-s",
        type=float,
        default=10.0,
        help="Timeout (seconds) for each compact_contact run (<=0 disables; default 10).",
    )

    # compact_contact knobs (defaults mirror hunt_compact_contact)
    ap.add_argument("--passes", type=int, default=300)
    ap.add_argument("--attempts-per-pass", type=int, default=150)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--boundary-topk", type=int, default=24)
    ap.add_argument("--push-bisect-iters", type=int, default=12)
    ap.add_argument("--push-max-step-frac", type=float, default=0.9)
    ap.add_argument("--plateau-eps", type=float, default=0.0)
    ap.add_argument("--diag-frac", type=float, default=0.3)
    ap.add_argument("--diag-rand", type=float, default=0.25)
    ap.add_argument("--center-bias", type=float, default=0.25)
    ap.add_argument("--interior-prob", type=float, default=0.15)
    ap.add_argument("--shake-pos", type=float, default=0.02)
    ap.add_argument("--shake-rot-deg", type=float, default=8.0)
    ap.add_argument("--shake-prob", type=float, default=0.25)
    ap.add_argument("--quantize-decimals", type=int, default=11)

    # restart diversification (optional)
    ap.add_argument("--restarts", type=int, default=1, help="Initial-state restarts per puzzle (default 1 = none).")
    ap.add_argument("--restart-seed", type=int, default=12345, help="RNG seed for restarts.")
    ap.add_argument("--restart-scale-min", type=float, default=1.0, help="Min centroid scale factor for restart detouch.")
    ap.add_argument("--restart-scale-max", type=float, default=1.0, help="Max centroid scale factor for restart detouch.")
    ap.add_argument("--restart-jitter-xy", type=float, default=0.0, help="Gaussian XY jitter sigma for restarts.")
    ap.add_argument("--restart-jitter-deg", type=float, default=0.0, help="Gaussian deg jitter sigma for restarts.")
    ap.add_argument("--restart-ruin-k", type=int, default=0, help="Ruin: randomly re-place k trees (default 0).")
    ap.add_argument(
        "--restart-ruin-box-scale",
        type=float,
        default=1.0,
        help="Ruin: sample inside a box of side (base_s * scale) around centroid.",
    )
    ap.add_argument(
        "--restart-allow-overlap",
        action="store_true",
        help="Allow restart base states that overlap (may diversify but can be unstable).",
    )
    ap.add_argument("--restart-attempts", type=int, default=6, help="Attempts to sample a non-overlapping restart base.")
    ap.add_argument(
        "--no-finalize-candidates",
        dest="finalize_candidates",
        action="store_false",
        help="Skip repair/finalization for overlapping compact_contact candidates.",
    )
    ap.set_defaults(finalize_candidates=True)

    ns = ap.parse_args(argv)

    root = Path.cwd().resolve()
    base_csv = Path(ns.base)
    if not base_csv.is_absolute():
        base_csv = (root / base_csv).resolve()
    if not base_csv.is_file():
        raise SystemExit(f"--base not found: {base_csv}")

    out_csv = Path(ns.out)
    if not out_csv.is_absolute():
        out_csv = (root / out_csv).resolve()

    nmax = int(ns.nmax)
    if nmax <= 0 or nmax > 200:
        raise SystemExit("--nmax must be in [1,200]")
    overlap_mode: OverlapMode = str(ns.overlap_mode)  # type: ignore[assignment]

    base = load_submission(base_csv, nmax=nmax)
    missing = [n for n in range(1, nmax + 1) if n not in base or base[n].shape != (n, 3)]
    if missing:
        raise SystemExit(f"Base CSV missing puzzles/wrong shape: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    base_text, base_offsets = _load_base_text_and_offsets(base_csv=base_csv, nmax=nmax)

    targets = _parse_int_list(str(ns.ns))
    if targets:
        targets = [n for n in targets if 1 <= n <= nmax]
        if not targets:
            raise SystemExit("--ns did not select any puzzles within [1..nmax]")
    else:
        targets = list(range(1, nmax + 1))

    seeds = _parse_int_list(str(ns.seeds))
    if not seeds:
        raise SystemExit("--seeds must select at least one seed")

    tmp_dir = out_csv.parent / "tmp" / f"opt_by_n_cc_{int(time.time())}"
    _eprint(
        f"Optimizing {len(targets)} puzzles with {len(seeds)} seeds each, restarts={int(ns.restarts)} (jobs={ns.jobs})"
    )

    results: list[_BestN] = []
    improved = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=int(ns.jobs)) as ex:
        timeout_s: float | None = None
        if float(ns.cc_timeout_s) > 0.0:
            timeout_s = float(ns.cc_timeout_s)
        futs = {
            ex.submit(
                _optimize_one_n,
                root=root,
                base_csv=base_csv,
                base_text=base_text,
                base_offsets=base_offsets,
                base_poses=base[int(n)],
                n=int(n),
                nmax=nmax,
                seeds=seeds,
                restarts=int(ns.restarts),
                restart_seed=int(ns.restart_seed),
                restart_scale_min=float(ns.restart_scale_min),
                restart_scale_max=float(ns.restart_scale_max),
                restart_jitter_xy=float(ns.restart_jitter_xy),
                restart_jitter_deg=float(ns.restart_jitter_deg),
                restart_ruin_k=int(ns.restart_ruin_k),
                restart_ruin_box_scale=float(ns.restart_ruin_box_scale),
                restart_allow_overlap=bool(ns.restart_allow_overlap),
                restart_attempts=int(ns.restart_attempts),
                finalize_candidates=bool(ns.finalize_candidates),
                overlap_mode=overlap_mode,
                tmp_dir=tmp_dir / f"n{int(n):03d}",
                keep_logs=bool(ns.keep_logs),
                timeout_s=timeout_s,
                passes=int(ns.passes),
                attempts_per_pass=int(ns.attempts_per_pass),
                patience=int(ns.patience),
                boundary_topk=int(ns.boundary_topk),
                push_bisect_iters=int(ns.push_bisect_iters),
                push_max_step_frac=float(ns.push_max_step_frac),
                plateau_eps=float(ns.plateau_eps),
                diag_frac=float(ns.diag_frac),
                diag_rand=float(ns.diag_rand),
                center_bias=float(ns.center_bias),
                interior_prob=float(ns.interior_prob),
                shake_pos=float(ns.shake_pos),
                shake_rot_deg=float(ns.shake_rot_deg),
                shake_prob=float(ns.shake_prob),
                quantize_decimals=int(ns.quantize_decimals),
                tol=float(ns.tol),
            ): int(n)
            for n in targets
        }
        for fut in as_completed(futs):
            n = futs[fut]
            r = fut.result()
            results.append(r)
            if r.improved:
                improved += 1
            _eprint(
                f"[n={r.n:3d}] s {r.base_s:.12f} -> {r.best_s:.12f} "
                f"({'improved' if r.improved else 'same'})"
                f" seed={r.best_seed} time={r.elapsed_s:.1f}s"
            )

    t1 = time.time()
    _eprint(f"Done: improved {improved}/{len(results)} puzzles in {t1 - t0:.1f}s")

    out_puzzles: dict[int, np.ndarray] = {n: _canonicalize_poses(base[n]) for n in range(1, nmax + 1)}
    for r in results:
        out_puzzles[int(r.n)] = np.array(r.poses, dtype=float, copy=True)
    _write_submission(out_csv, out_puzzles, nmax=nmax)

    if ns.report is not None:
        report = Path(ns.report)
        if not report.is_absolute():
            report = (root / report).resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["puzzle", "base_s", "best_s", "delta_s", "improved", "best_seed", "elapsed_s"])
            for r in sorted(results, key=lambda x: x.n):
                w.writerow(
                    [
                        int(r.n),
                        f"{float(r.base_s):.17f}",
                        f"{float(r.best_s):.17f}",
                        f"{float(r.best_s - r.base_s):.17f}",
                        int(bool(r.improved)),
                        "" if r.best_seed is None else int(r.best_seed),
                        f"{float(r.elapsed_s):.3f}",
                    ]
                )

    res = score_submission(out_csv, nmax=nmax, overlap_mode=overlap_mode)
    print(str(out_csv))
    print(f"score({overlap_mode}): {res.score:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
