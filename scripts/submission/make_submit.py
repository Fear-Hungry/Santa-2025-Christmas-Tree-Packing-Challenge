#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run(
    cmd: list[str],
    *,
    cwd: Path = ROOT,
    check: bool = True,
    capture: bool = False,
    stdout_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    if capture and stdout_path is not None:
        raise ValueError("Use either capture=True or stdout_path=..., not both.")

    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True)
        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        return proc

    proc = subprocess.run(cmd, cwd=cwd, capture_output=capture, text=True)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)
    return proc


def _git(*args: str) -> str | None:
    try:
        proc = _run(["git", *args], capture=True, check=False)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip()


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug


def _filter_passthrough_args(extra: list[str]) -> list[str]:
    blocked = {"--config", "--out", "--nmax"}
    extra = [tok for tok in extra if tok != "--"]
    for tok in extra:
        if tok in blocked or tok.startswith("--config=") or tok.startswith("--out=") or tok.startswith("--nmax="):
            raise SystemExit(f"Passe {tok} via flags do make_submit.py (evita inconsistência).")
    return extra


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate + strict-score a submission and archive it under submissions/…")
    ap.add_argument("--config", type=Path, default=None, help="JSON config passed to generate_submission.py")
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle n (passed to generator and scorer).")
    ap.add_argument("--seed", type=int, default=None, help="Base seed for generator (optional).")
    ap.add_argument("--name", type=str, default=None, help="Optional label appended to the run folder name.")
    ap.add_argument("--submissions-dir", type=Path, default=ROOT / "submissions", help="Archive root directory.")
    args, extra = ap.parse_known_args()

    extra = _filter_passthrough_args(extra)

    sha = _git("rev-parse", "HEAD") or "unknown"
    short = sha[:7] if sha != "unknown" else "nogit"
    dirty = bool(_git("status", "--porcelain"))
    ts = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    label = _safe_slug(args.name) if args.name else None

    run_name = f"{ts}_{short}" + (f"_{label}" if label else "")
    run_dir = args.submissions_dir / run_name
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = args.submissions_dir / f"{run_name}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)

    submission_path = run_dir / "submission.csv"
    gen_log = run_dir / "generate.log"
    score_log = run_dir / "score.log"

    gen_cmd = [sys.executable, str(ROOT / "scripts/submission/generate_submission.py")]
    if args.config is not None:
        gen_cmd += ["--config", str(args.config)]
    gen_cmd += ["--nmax", str(int(args.nmax))]
    if args.seed is not None:
        gen_cmd += ["--seed", str(int(args.seed))]
    gen_cmd += list(extra)
    gen_cmd += ["--out", str(submission_path)]

    _run(gen_cmd, stdout_path=gen_log)

    score_cmd = [
        sys.executable,
        str(ROOT / "scripts/evaluation/score_submission.py"),
        str(submission_path),
        "--nmax",
        str(int(args.nmax)),
        "--pretty",
    ]
    proc = _run(score_cmd, capture=True, check=False)
    score_log.write_text((proc.stderr or "").strip() + "\n", encoding="utf-8")
    if proc.returncode != 0:
        (run_dir / "score_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        raise SystemExit(f"Scoring failed (see {score_log} and score_stdout.txt).")

    score_text = proc.stdout or "{}"
    (run_dir / "score.json").write_text(score_text, encoding="utf-8")
    score = json.loads(score_text)

    meta = {
        "timestamp_utc": ts,
        "git": {"sha": sha, "dirty": dirty},
        "paths": {"run_dir": str(run_dir), "submission": str(submission_path)},
        "generator": {"cmd": gen_cmd, "config": str(args.config) if args.config is not None else None, "log": str(gen_log)},
        "scorer": {"cmd": score_cmd, "log": str(score_log)},
        "score": score,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Run: {run_dir}")
    if "score" in score:
        print(f"Score: {score['score']}")
    print(f"Submission: {submission_path}")
    if dirty:
        print("Warning: git working tree is dirty (recorded in meta.json).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
