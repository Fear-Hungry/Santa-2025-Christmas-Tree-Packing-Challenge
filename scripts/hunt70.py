#!/usr/bin/env python3

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_capture(cmd: list[str], *, cwd: Path, log_path: Path) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd,
                          cwd=str(cwd),
                          text=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    return proc


def _run_log(cmd: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, text=True)
        return proc.returncode


def _require_exe(path: Path, *, hint: str) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"Missing executable: {path} ({hint})")


def _parse_score(output: str) -> float:
    m = re.search(r"^Score:\s*([0-9]+(?:\.[0-9]+)?)\s*$", output, flags=re.MULTILINE)
    if not m:
        raise RuntimeError("Could not parse score from score_submission output.")
    return float(m.group(1))


def score_submission(score_exe: Path,
                     submission_csv: Path,
                     *,
                     cwd: Path,
                     out_dir: Path,
                     tag: str,
                     top_k: int = 30) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"score_{tag}.log"
    breakdown_csv = out_dir / f"breakdown_{tag}.csv"
    report_csv = out_dir / f"report_{tag}.csv"
    cmd = [
        str(score_exe),
        str(submission_csv),
        "--top",
        str(top_k),
        "--csv",
        str(breakdown_csv),
        "--report",
        str(report_csv),
    ]
    proc = _run_capture(cmd, cwd=cwd, log_path=log_path)
    if proc.returncode != 0:
        raise RuntimeError(f"score_submission failed for {submission_csv} (see {log_path}).")
    return _parse_score(proc.stdout)


def ensemble_in_batches(ensemble_exe: Path,
                        *,
                        cwd: Path,
                        output_csv: Path,
                        inputs: list[Path],
                        batch_size: int,
                        no_final_rigid: bool,
                        cross_check_n: bool,
                        log_dir: Path,
                        tag: str) -> None:
    if not inputs:
        raise ValueError("No inputs to ensemble.")

    log_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = log_dir / "_tmp_ensemble"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    curr = tmp_dir / "curr.csv"
    shutil.copyfile(inputs[0], curr)

    i = 1
    step = 0
    while i < len(inputs):
        batch = inputs[i:i + batch_size]
        nxt = tmp_dir / f"step_{step:04d}.csv"
        cmd = [str(ensemble_exe), str(nxt), str(curr), *[str(p) for p in batch]]
        if no_final_rigid:
            cmd.append("--no-final-rigid")
        if cross_check_n:
            cmd.append("--cross-check-n")

        log_path = log_dir / f"ensemble_{tag}_{step:04d}.log"
        rc = _run_log(cmd, cwd=cwd, log_path=log_path)
        if rc != 0:
            raise RuntimeError(f"ensemble_submissions failed (see {log_path}).")

        curr = nxt
        i += batch_size
        step += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(curr, output_csv)


def _chunks_1_to_200(num_chunks: int) -> list[tuple[int, int]]:
    if num_chunks <= 0:
        raise ValueError("num_chunks must be > 0")
    out = []
    lo = 1
    for i in range(num_chunks):
        rem = 200 - lo + 1
        left = num_chunks - i
        size = rem // left
        if rem % left != 0:
            size += 1
        hi = min(200, lo + size - 1)
        out.append((lo, hi))
        lo = hi + 1
        if lo > 200:
            break
    return out


def _git_info(cwd: Path) -> dict:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True).strip()
    except Exception:
        head = ""
    try:
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(cwd), text=True).strip()
        is_dirty = bool(dirty)
    except Exception:
        is_dirty = False
    return {"head": head, "dirty": is_dirty}


def _promote_if_better(*,
                       candidate_csv: Path,
                       candidate_score: float,
                       cwd: Path,
                       archive_dir: Path,
                       best_csv: Path,
                       best_txt: Path,
                       scores_tsv: Path,
                       score_exe: Path) -> bool:
    if not best_csv.exists():
        best_score = float("inf")
    else:
        best_score = score_submission(score_exe, best_csv, cwd=cwd, out_dir=archive_dir, tag="current_best", top_k=10)

    if not (candidate_score + 1e-12 < best_score):
        return False

    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = _now_tag()
    if best_csv.exists():
        archived_name = f"submission_best_{ts}_{best_score:.12f}.csv"
        shutil.copyfile(best_csv, archive_dir / archived_name)

    shutil.copyfile(candidate_csv, best_csv)

    best_txt.write_text(
        "best_file\tsubmissions/submission_best.csv\n"
        f"best_score\t{candidate_score:.12f}\n",
        encoding="utf-8",
    )
    scores_tsv.write_text(
        "status\tscore\tfile\tpath\terror\n"
        f"BEST\t{candidate_score:.12f}\tsubmission_best.csv\tsubmissions/submission_best.csv\t\n",
        encoding="utf-8",
    )
    return True


def _as_int_list(xs: str) -> list[int]:
    out = []
    for part in xs.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Hunt <70 by running many compact_contact variants + ensembling per-n, "
            "optionally followed by targeted blend_repair/MZ refinement."
        )
    )
    ap.add_argument("--base",
                    default="submissions/submission_best.csv",
                    help="Base submission CSV (default: submissions/submission_best.csv).")
    ap.add_argument("--run-dir",
                    default=None,
                    help="Output directory under runs/ (default: runs/hunt70_<timestamp>).")
    ap.add_argument("--jobs", type=int, default=8, help="Parallel jobs (default: 8).")
    ap.add_argument("--preset", choices=["smoke", "full"], default="full")
    ap.add_argument("--seed0", type=int, default=1, help="Base seed for randomization (default: 1).")
    ap.add_argument("--rounds", type=int, default=1, help="How many rounds to run (default: 1).")
    ap.add_argument("--target-score", type=float, default=70.0, help="Stop when score <= target (default: 70).")
    ap.add_argument("--resume", action="store_true", help="Reuse existing outputs if present.")
    ap.add_argument("--no-promote", action="store_true", help="Do not update submissions/ even if better.")

    ap.add_argument("--no-safe", action="store_true", help="Skip safe chunked compact_contact stage.")
    ap.add_argument("--no-noisy", action="store_true", help="Skip noisy compact_contact stage.")
    ap.add_argument("--no-finalize", action="store_true", help="Skip finalize blend_repair stage.")

    ap.add_argument("--safe-chunks", type=int, default=0, help="Chunks in safe stage (default: jobs).")
    ap.add_argument("--noisy-runs", type=int, default=64, help="Noisy runs per round (default: 64).")
    ap.add_argument("--ensemble-batch-size",
                    type=int,
                    default=120,
                    help="Max inputs per ensemble call (default: 120).")
    ap.add_argument("--keep-noisy", action="store_true", help="Keep individual noisy CSVs (default: delete after merge).")
    ap.add_argument("--no-final-rigid", action="store_true", help="Disable final rigid during ensembling.")
    ap.add_argument("--cross-check-n", action="store_true", help="Enable --cross-check-n during ensembling.")

    ap.add_argument("--finalize-runs",
                    type=int,
                    default=1,
                    help="How many finalize attempts (different seeds) per round (default: 1).")
    ap.add_argument("--finalize-target-top",
                    type=int,
                    default=60,
                    help="blend_repair --target-top K (default: 60).")
    ap.add_argument("--finalize-seed", default=None, help="Comma-separated seed list for finalize (overrides finalize-runs).")

    args = ap.parse_args()

    cwd = Path.cwd()
    base_csv = (cwd / args.base).resolve()
    if not base_csv.exists():
        raise SystemExit(f"Base CSV not found: {base_csv}")

    bin_dir = cwd / "bin"
    compact_exe = bin_dir / "compact_contact"
    ensemble_exe = bin_dir / "ensemble_submissions"
    score_exe = bin_dir / "score_submission"
    blend_exe = bin_dir / "blend_repair"

    _require_exe(compact_exe, hint="build target compact_contact")
    _require_exe(ensemble_exe, hint="build target ensemble_submissions")
    _require_exe(score_exe, hint="build target score_submission")
    _require_exe(blend_exe, hint="build target blend_repair")

    if args.jobs <= 0:
        raise SystemExit("--jobs must be > 0")
    if args.rounds <= 0:
        raise SystemExit("--rounds must be > 0")
    if args.noisy_runs < 0:
        raise SystemExit("--noisy-runs must be >= 0")
    if args.finalize_runs < 0:
        raise SystemExit("--finalize-runs must be >= 0")
    if args.ensemble_batch_size <= 0:
        raise SystemExit("--ensemble-batch-size must be > 0")

    run_dir = Path(args.run_dir) if args.run_dir else (cwd / "runs" / f"hunt70_{_now_tag()}")
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "cwd": str(cwd),
        "args": vars(args),
        "git": _git_info(cwd),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Presets (can be overridden by editing this script or passing different flags later).
    if args.preset == "smoke":
        safe_cfg = dict(passes=120, attempts=80, patience=40, boundary_topk=20)
        noisy_cfg = dict(passes=180, attempts=120, patience=60, boundary_topk=24)
        noisy_noise = dict(shake_pos=0.02, shake_rot_deg=8.0, shake_prob=0.25)
        finalize_cfg = dict(repair_passes=400, repair_attempts=180, repair_mtv_passes=200,
                            squeeze_tries=20, squeeze_steps=2, squeeze_alpha_min=0.99, squeeze_alpha_max=0.9995,
                            squeeze_repair_passes=150,
                            global_contract_steps=10, global_contract_scale=0.999, global_contract_relax_iters=60,
                            global_contract_repair_passes=150,
                            interlock_passes=1, pocket_iters=1,
                            sa_iters=600, sa_restarts=1, sa_mz_two_phase=True, sa_mz_a_iters=300)
    else:
        safe_cfg = dict(passes=800, attempts=200, patience=100, boundary_topk=28)
        noisy_cfg = dict(passes=1200, attempts=240, patience=200, boundary_topk=32)
        noisy_noise = dict(shake_pos=0.05, shake_rot_deg=15.0, shake_prob=0.35)
        finalize_cfg = dict(repair_passes=900, repair_attempts=220, repair_mtv_passes=300,
                            squeeze_tries=80, squeeze_steps=3, squeeze_alpha_min=0.985, squeeze_alpha_max=0.9995,
                            squeeze_repair_passes=300,
                            global_contract_steps=30, global_contract_scale=0.999, global_contract_relax_iters=80,
                            global_contract_repair_passes=300,
                            interlock_passes=2, pocket_iters=4,
                            sa_iters=1200, sa_restarts=1, sa_mz_two_phase=True, sa_mz_a_iters=600)

    # Fixed geometry-ish knobs for compact_contact.
    cc_common = dict(push_max_step_frac=0.9,
                     push_bisect_iters=12,
                     plateau_eps=0.0,
                     diag_frac=0.30,
                     diag_rand=0.25,
                     center_bias=0.25,
                     interior_prob=0.15,
                     quantize_decimals=9)

    best_path = base_csv
    best_score = score_submission(score_exe, best_path, cwd=cwd, out_dir=run_dir, tag="base", top_k=30)
    (run_dir / "best.json").write_text(json.dumps({"path": str(best_path), "score": best_score}, indent=2) + "\n",
                                       encoding="utf-8")
    print(f"[base] score={best_score:.12f} path={best_path}")

    for rnd in range(args.rounds):
        if best_score <= args.target_score + 1e-12:
            print(f"[stop] reached target score: {best_score:.12f} <= {args.target_score:.6f}")
            break

        rdir = run_dir / f"round_{rnd:03d}"
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "base.csv").write_text(best_path.read_text(encoding="utf-8"), encoding="utf-8")
        base_round = rdir / "base.csv"

        curr_best = base_round
        curr_score = best_score

        # Stage 1: safe chunked passes (no shake).
        if not args.no_safe:
            chunks = args.safe_chunks if args.safe_chunks > 0 else args.jobs
            ranges = _chunks_1_to_200(chunks)
            out_dir = rdir / "cc_chunks"
            out_dir.mkdir(parents=True, exist_ok=True)
            jobs = []
            for idx, (a, b) in enumerate(ranges):
                out_csv = out_dir / f"cc_{a:03d}_{b:03d}.csv"
                log = out_dir / f"cc_{a:03d}_{b:03d}.log"
                if args.resume and out_csv.exists():
                    continue
                seed = args.seed0 + 1000 * rnd + idx
                cmd = [
                    str(compact_exe),
                    "--base",
                    str(curr_best),
                    "--out",
                    str(out_csv),
                    "--n-min",
                    "1",
                    "--n-max",
                    "200",
                    "--target-range",
                    f"{a},{b}",
                    "--seed",
                    str(seed),
                    "--passes",
                    str(safe_cfg["passes"]),
                    "--attempts-per-pass",
                    str(safe_cfg["attempts"]),
                    "--patience",
                    str(safe_cfg["patience"]),
                    "--boundary-topk",
                    str(safe_cfg["boundary_topk"]),
                    "--push-bisect-iters",
                    str(cc_common["push_bisect_iters"]),
                    "--push-max-step-frac",
                    str(cc_common["push_max_step_frac"]),
                    "--plateau-eps",
                    str(cc_common["plateau_eps"]),
                    "--diag-frac",
                    str(cc_common["diag_frac"]),
                    "--diag-rand",
                    str(cc_common["diag_rand"]),
                    "--center-bias",
                    str(cc_common["center_bias"]),
                    "--interior-prob",
                    str(cc_common["interior_prob"]),
                    "--shake-pos",
                    "0.0",
                    "--shake-rot-deg",
                    "0.0",
                    "--shake-prob",
                    "0.0",
                    "--quantize-decimals",
                    str(cc_common["quantize_decimals"]),
                ]
                jobs.append((cmd, log, out_csv))

            if jobs:
                print(f"[round {rnd}] safe chunks: running {len(jobs)} jobs (parallel={args.jobs})")
                with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
                    futs = [ex.submit(_run_log, cmd, cwd=cwd, log_path=log) for cmd, log, _ in jobs]
                    for f in cf.as_completed(futs):
                        rc = f.result()
                        if rc != 0:
                            raise RuntimeError("compact_contact failed in safe stage (check logs).")

            chunk_csvs = sorted(out_dir.glob("cc_*.csv"))
            if not chunk_csvs:
                raise RuntimeError("No chunk CSVs produced in safe stage.")

            ens_csv = rdir / "cc_safe_ens.csv"
            if not (args.resume and ens_csv.exists()):
                print(f"[round {rnd}] ensembling safe chunks -> {ens_csv.name}")
                ensemble_in_batches(ensemble_exe,
                                    cwd=cwd,
                                    output_csv=ens_csv,
                                    inputs=[curr_best, *chunk_csvs],
                                    batch_size=args.ensemble_batch_size,
                                    no_final_rigid=args.no_final_rigid,
                                    cross_check_n=args.cross_check_n,
                                    log_dir=rdir / "ensemble_safe_logs",
                                    tag="safe")
            ens_score = score_submission(score_exe, ens_csv, cwd=cwd, out_dir=rdir, tag="safe", top_k=30)
            print(f"[round {rnd}] safe score={ens_score:.12f} (base={curr_score:.12f})")
            if ens_score + 1e-12 < curr_score:
                curr_best = ens_csv
                curr_score = ens_score

        # Stage 2: noisy explore.
        if not args.no_noisy and args.noisy_runs > 0:
            out_dir = rdir / "cc_noisy"
            out_dir.mkdir(parents=True, exist_ok=True)

            def build_noisy_job(run_id: int, seed: int) -> tuple[list[str], Path, Path]:
                out_csv = out_dir / f"cc_{run_id:04d}_seed{seed}.csv"
                log = out_dir / f"cc_{run_id:04d}_seed{seed}.log"
                cmd = [
                    str(compact_exe),
                    "--base",
                    str(curr_best),
                    "--out",
                    str(out_csv),
                    "--n-min",
                    "1",
                    "--n-max",
                    "200",
                    "--target-range",
                    "1,200",
                    "--seed",
                    str(seed),
                    "--passes",
                    str(noisy_cfg["passes"]),
                    "--attempts-per-pass",
                    str(noisy_cfg["attempts"]),
                    "--patience",
                    str(noisy_cfg["patience"]),
                    "--boundary-topk",
                    str(noisy_cfg["boundary_topk"]),
                    "--push-bisect-iters",
                    str(cc_common["push_bisect_iters"]),
                    "--push-max-step-frac",
                    str(cc_common["push_max_step_frac"]),
                    "--plateau-eps",
                    str(cc_common["plateau_eps"]),
                    "--diag-frac",
                    str(cc_common["diag_frac"]),
                    "--diag-rand",
                    str(cc_common["diag_rand"]),
                    "--center-bias",
                    str(cc_common["center_bias"]),
                    "--interior-prob",
                    str(cc_common["interior_prob"]),
                    "--shake-pos",
                    str(noisy_noise["shake_pos"]),
                    "--shake-rot-deg",
                    str(noisy_noise["shake_rot_deg"]),
                    "--shake-prob",
                    str(noisy_noise["shake_prob"]),
                    "--quantize-decimals",
                    str(cc_common["quantize_decimals"]),
                ]
                return cmd, log, out_csv

            jobs = []
            for run_id in range(args.noisy_runs):
                seed = args.seed0 + 10000 * rnd + 2000 + run_id
                cmd, log, out_csv = build_noisy_job(run_id, seed)
                if args.resume and out_csv.exists():
                    continue
                jobs.append((cmd, log, out_csv))

            if jobs:
                print(f"[round {rnd}] noisy: running {len(jobs)} jobs (parallel={args.jobs})")
                with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
                    futs = [ex.submit(_run_log, cmd, cwd=cwd, log_path=log) for cmd, log, _ in jobs]
                    for f in cf.as_completed(futs):
                        rc = f.result()
                        if rc != 0:
                            raise RuntimeError("compact_contact failed in noisy stage (check logs).")

            noisy_csvs = sorted(out_dir.glob("cc_*.csv"))
            if not noisy_csvs:
                print(f"[round {rnd}] noisy: no outputs (skipping).")
            else:
                ens_csv = rdir / "cc_noisy_ens.csv"
                if not (args.resume and ens_csv.exists()):
                    print(f"[round {rnd}] ensembling noisy runs -> {ens_csv.name}")
                    ensemble_in_batches(ensemble_exe,
                                        cwd=cwd,
                                        output_csv=ens_csv,
                                        inputs=[curr_best, *noisy_csvs],
                                        batch_size=args.ensemble_batch_size,
                                        no_final_rigid=args.no_final_rigid,
                                        cross_check_n=args.cross_check_n,
                                        log_dir=rdir / "ensemble_noisy_logs",
                                        tag="noisy")
                ens_score = score_submission(score_exe, ens_csv, cwd=cwd, out_dir=rdir, tag="noisy", top_k=30)
                print(f"[round {rnd}] noisy score={ens_score:.12f} (prev={curr_score:.12f})")
                if ens_score + 1e-12 < curr_score:
                    curr_best = ens_csv
                    curr_score = ens_score

                if not args.keep_noisy:
                    for p in noisy_csvs:
                        try:
                            p.unlink()
                        except OSError:
                            pass

        # Stage 3: finalize (targeted blend_repair).
        if not args.no_finalize and ((args.finalize_runs > 0) or args.finalize_seed):
            out_dir = rdir / "finalize"
            out_dir.mkdir(parents=True, exist_ok=True)

            if args.finalize_seed:
                seeds = _as_int_list(args.finalize_seed)
            else:
                seeds = [args.seed0 + 30000 * rnd + i for i in range(args.finalize_runs)]

            fin_csvs = []
            for i, seed in enumerate(seeds):
                out_csv = out_dir / f"finish_{i:03d}_seed{seed}.csv"
                log = out_dir / f"finish_{i:03d}_seed{seed}.log"
                fin_csvs.append(out_csv)
                if args.resume and out_csv.exists():
                    continue

                cmd = [
                    str(blend_exe),
                    str(out_csv),
                    str(curr_best),
                    "--base",
                    str(curr_best),
                    "--target-top",
                    str(args.finalize_target_top),
                    "--topk-per-n",
                    "1",
                    "--blend-iters",
                    "0",
                    "--seed",
                    str(seed),
                    "--repair-passes",
                    str(finalize_cfg["repair_passes"]),
                    "--repair-attempts",
                    str(finalize_cfg["repair_attempts"]),
                    "--repair-mtv-passes",
                    str(finalize_cfg["repair_mtv_passes"]),
                    "--squeeze-tries",
                    str(finalize_cfg["squeeze_tries"]),
                    "--squeeze-steps",
                    str(finalize_cfg["squeeze_steps"]),
                    "--squeeze-alpha-min",
                    str(finalize_cfg["squeeze_alpha_min"]),
                    "--squeeze-alpha-max",
                    str(finalize_cfg["squeeze_alpha_max"]),
                    "--squeeze-repair-passes",
                    str(finalize_cfg["squeeze_repair_passes"]),
                    "--global-contract-steps",
                    str(finalize_cfg["global_contract_steps"]),
                    "--global-contract-scale",
                    str(finalize_cfg["global_contract_scale"]),
                    "--global-contract-relax-iters",
                    str(finalize_cfg["global_contract_relax_iters"]),
                    "--global-contract-repair-passes",
                    str(finalize_cfg["global_contract_repair_passes"]),
                    "--interlock-passes",
                    str(finalize_cfg["interlock_passes"]),
                    "--pocket-iters",
                    str(finalize_cfg["pocket_iters"]),
                    "--sa-iters",
                    str(finalize_cfg["sa_iters"]),
                    "--sa-restarts",
                    str(finalize_cfg["sa_restarts"]),
                ]
                if finalize_cfg.get("sa_mz_two_phase"):
                    cmd.append("--sa-mz-two-phase")
                    cmd += ["--sa-mz-a-iters", str(finalize_cfg["sa_mz_a_iters"])]

                rc = _run_log(cmd, cwd=cwd, log_path=log)
                if rc != 0:
                    raise RuntimeError(f"blend_repair failed (see {log}).")

            existing_fins = [p for p in fin_csvs if p.exists()]
            if existing_fins:
                ens_csv = rdir / "finish_ens.csv"
                if not (args.resume and ens_csv.exists()):
                    ensemble_in_batches(ensemble_exe,
                                        cwd=cwd,
                                        output_csv=ens_csv,
                                        inputs=[curr_best, *existing_fins],
                                        batch_size=args.ensemble_batch_size,
                                        no_final_rigid=args.no_final_rigid,
                                        cross_check_n=args.cross_check_n,
                                        log_dir=rdir / "ensemble_finish_logs",
                                        tag="finish")
                ens_score = score_submission(score_exe, ens_csv, cwd=cwd, out_dir=rdir, tag="finish", top_k=30)
                print(f"[round {rnd}] finalize score={ens_score:.12f} (prev={curr_score:.12f})")
                if ens_score + 1e-12 < curr_score:
                    curr_best = ens_csv
                    curr_score = ens_score

        # Round result.
        (rdir / "round_best.json").write_text(
            json.dumps({"path": str(curr_best), "score": curr_score}, indent=2) + "\n", encoding="utf-8")
        print(f"[round {rnd}] best score={curr_score:.12f} path={curr_best}")

        if curr_score + 1e-12 < best_score:
            best_score = curr_score
            best_path = curr_best
            (run_dir / "best.json").write_text(
                json.dumps({"path": str(best_path), "score": best_score}, indent=2) + "\n", encoding="utf-8")

        if not args.no_promote:
            promoted = _promote_if_better(candidate_csv=best_path,
                                          candidate_score=best_score,
                                          cwd=cwd,
                                          archive_dir=cwd / "submissions" / "archive",
                                          best_csv=cwd / "submissions" / "submission_best.csv",
                                          best_txt=cwd / "submissions" / "BEST.txt",
                                          scores_tsv=cwd / "submissions" / "scores.tsv",
                                          score_exe=score_exe)
            if promoted:
                print(f"[promote] updated submissions/submission_best.csv to score={best_score:.12f}")

    print(f"[done] best score={best_score:.12f} path={best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

