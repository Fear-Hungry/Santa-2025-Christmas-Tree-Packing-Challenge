#!/usr/bin/env python3
"""Archive old submission CSVs, keeping only the best/top files in repo root."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from santa_packing.scoring import score_submission


def score_csv(path: Path) -> float:
    result = score_submission(path, nmax=200, check_overlap=False, overlap_mode="kaggle", require_complete=True)
    return float(result.score)


def main() -> int:
    ap = argparse.ArgumentParser(description="Archive old submission CSVs (root folder).")
    ap.add_argument("--keep-top", type=int, default=8, help="Keep top-K scoring CSVs in root (default: 8)")
    ap.add_argument("--dry-run", action="store_true", help="Do not move files, only print actions")
    args = ap.parse_args()

    repo = Path(".").resolve()
    csvs = sorted(p for p in repo.glob("submission*.csv") if p.is_file())

    keep_names = {
        "submission.csv",
        "submission_kaggle_fixed.csv",
        "submission_kaggle_safe.csv",
        "submission_best_hours.csv",
        "submission_merge_best_public.csv",
    }

    scored = []
    for p in csvs:
        if p.name in keep_names:
            continue
        try:
            s = score_csv(p)
        except Exception:
            s = None
        scored.append((s, p))

    scored.sort(key=lambda x: (-x[0] if x[0] is not None else float("inf"), x[1].name))
    keep_top = {p for _, p in scored[: args.keep_top] if p is not None}

    keep = {repo / name for name in keep_names if (repo / name).exists()}
    keep |= keep_top

    archive_dir = repo / "submissions" / "archive" / datetime.now().strftime("%Y%m%d")
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved = []
    for p in csvs:
        if p in keep:
            continue
        target = archive_dir / p.name
        if args.dry_run:
            moved.append({"from": str(p), "to": str(target)})
            continue
        shutil.move(str(p), str(target))
        moved.append({"from": str(p), "to": str(target)})

    index = {
        "timestamp": datetime.now().isoformat(),
        "keep": [str(p) for p in sorted(keep)],
        "moved": moved,
    }
    index_path = archive_dir / "index.json"
    if args.dry_run:
        print(json.dumps(index, indent=2))
    else:
        index_path.write_text(json.dumps(index, indent=2))
        print(f"Archived {len(moved)} file(s) to {archive_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
