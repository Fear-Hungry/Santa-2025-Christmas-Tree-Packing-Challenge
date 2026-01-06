from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import re
import subprocess
import sys
import traceback
from pathlib import Path

from santa_packing.cli.generate_submission import main as generate_main
from santa_packing.scoring import score_submission


def _git(*args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            text=True,
            capture_output=True,
            check=False,
        )
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
    blocked = {"--config", "--out", "--nmax", "--seed", "--overlap-mode"}
    extra = [tok for tok in extra if tok != "--"]
    for tok in extra:
        if (
            tok in blocked
            or tok.startswith("--config=")
            or tok.startswith("--out=")
            or tok.startswith("--nmax=")
            or tok.startswith("--seed=")
            or tok.startswith("--overlap-mode=")
        ):
            raise SystemExit(f"Passe {tok} via flags do make_submit.py (evita inconsistência).")
    return extra


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    ap = argparse.ArgumentParser(description="Generate + strict-score a submission and archive it under submissions/…")
    ap.add_argument("--config", type=Path, default=None, help="JSON config passed to generate_submission")
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle n (passed to generator and scorer).")
    ap.add_argument("--seed", type=int, default=None, help="Base seed for generator (optional).")
    ap.add_argument(
        "--overlap-mode",
        type=str,
        default="strict",
        choices=["strict", "conservative", "kaggle"],
        help="Overlap predicate used for scoring/validation (strict allows touching; kaggle enforces clearance).",
    )
    ap.add_argument("--name", type=str, default=None, help="Optional label appended to the run folder name.")
    ap.add_argument("--submissions-dir", type=Path, default=Path("submissions"), help="Archive root directory.")
    args, extra = ap.parse_known_args(argv)

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

    gen_argv: list[str] = []
    if args.config is not None:
        gen_argv += ["--config", str(args.config)]
    gen_argv += ["--nmax", str(int(args.nmax))]
    if args.seed is not None:
        gen_argv += ["--seed", str(int(args.seed))]
    gen_argv += ["--overlap-mode", str(args.overlap_mode)]
    gen_argv += list(extra)
    gen_argv += ["--out", str(submission_path)]

    gen_rc = 1
    with gen_log.open("w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            try:
                gen_rc = int(generate_main(gen_argv))
            except SystemExit as e:
                gen_rc = int(e.code or 1)
            except Exception:
                traceback.print_exc()
                gen_rc = 1
    if gen_rc != 0:
        raise SystemExit(f"Generation failed (see {gen_log}).")

    try:
        result = score_submission(
            submission_path,
            nmax=int(args.nmax),
            check_overlap=True,
            overlap_mode=str(args.overlap_mode),
            require_complete=True,
        )
        score_data = result.to_json()
        (run_dir / "score.json").write_text(json.dumps(score_data, indent=2) + "\n", encoding="utf-8")
        score_log.write_text("", encoding="utf-8")
    except Exception:
        score_log.write_text(traceback.format_exc() + "\n", encoding="utf-8")
        raise SystemExit(f"Scoring failed (see {score_log}).")

    meta = {
        "timestamp_utc": ts,
        "git": {"sha": sha, "dirty": dirty},
        "paths": {"run_dir": str(run_dir), "submission": str(submission_path)},
        "generator": {"argv": gen_argv, "config": str(args.config) if args.config is not None else None, "log": str(gen_log)},
        "scorer": {
            "nmax": int(args.nmax),
            "check_overlap": True,
            "overlap_mode": str(args.overlap_mode),
            "require_complete": True,
            "log": str(score_log),
        },
        "score": score_data,
        "python": sys.version,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Run: {run_dir}")
    if "score" in score_data:
        print(f"Score: {score_data['score']}")
    print(f"Submission: {submission_path}")
    if dirty:
        print("Warning: git working tree is dirty (recorded in meta.json).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
