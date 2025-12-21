#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar


@dataclass(frozen=True)
class RunRow:
    seed: int
    preset: str
    score: float
    submission: Path


@dataclass(frozen=True)
class RunConfig:
    seeds: list[int]
    presets: list[str]
    solver_args: list[str]
    solver: Path
    scorer: Path
    output: Path
    run_dir: Path
    root: Path


def _sanitize_tag(tag: str) -> str:
    tag = tag.strip()
    if not tag:
        return "auto"
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", tag)
    tag = re.sub(r"_{2,}", "_", tag)
    return tag.strip("._-") or "auto"


def _parse_csv_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def _strip_leading_double_dash(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def _extract_n_max(args: list[str]) -> int | None:
    for idx, arg in enumerate(args):
        if arg == "--n-max" and idx + 1 < len(args):
            try:
                return int(args[idx + 1])
            except ValueError:
                return None
    return None


def _run_and_log(cmd: list[str], cwd: Path, log_path: Path) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.stdout


def _parse_score(text: str) -> float:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Score:"):
            return float(line.split(":", 1)[1].strip())
    raise SystemExit("Score not found in score_submission output.")


T = TypeVar("T")


def _require_non_empty(values: list[T], message: str) -> list[T]:
    if not values:
        raise SystemExit(message)
    return values


def _parse_int_list(value: str) -> list[int]:
    return [int(s) for s in _parse_csv_list(value)]


def _validate_n_max(args: list[str]) -> None:
    n_max = _extract_n_max(args)
    if n_max is not None and n_max != 200:
        raise SystemExit("--n-max must be 200 when using auto_tessellation.")


def _require_file(path: Path, message: str) -> Path:
    if not path.is_file():
        raise SystemExit(message)
    return path


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _make_run_dir(runs_dir: str, tag: str) -> Path:
    safe_tag = _sanitize_tag(tag)
    run_dir = Path(runs_dir) / f"{safe_tag}_{_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-select tessellation presets by local score.",
    )
    parser.add_argument("--seeds", default="1", help="Comma-separated seeds.")
    parser.add_argument(
        "--presets",
        default="quick,balanced,quality",
        help="Comma-separated presets (e.g. quick,balanced,quality).",
    )
    parser.add_argument(
        "--output",
        default="submission_auto.csv",
        help="Final submission output path.",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs/auto_tessellation",
        help="Base runs directory.",
    )
    parser.add_argument("--tag", default="auto", help="Run tag prefix.")
    parser.add_argument(
        "--solver",
        default="./bin/solver_tessellation",
        help="Path to solver_tessellation binary.",
    )
    parser.add_argument(
        "--scorer",
        default="./bin/score_submission",
        help="Path to score_submission binary.",
    )
    parser.add_argument("solver_args", nargs=argparse.REMAINDER)

    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace) -> RunConfig:
    seeds = _require_non_empty(_parse_int_list(args.seeds), "--seeds cannot be empty.")
    presets = _require_non_empty(_parse_csv_list(args.presets), "--presets cannot be empty.")
    solver_args = _strip_leading_double_dash(args.solver_args)
    _validate_n_max(solver_args)

    root = Path.cwd()
    solver = _require_file(Path(args.solver), f"Solver not found: {args.solver}")
    scorer = _require_file(Path(args.scorer), f"Scorer not found: {args.scorer}")
    run_dir = _make_run_dir(args.runs_dir, args.tag)

    return RunConfig(
        seeds=seeds,
        presets=presets,
        solver_args=solver_args,
        solver=solver,
        scorer=scorer,
        output=Path(args.output),
        run_dir=run_dir,
        root=root,
    )


def _run_sweep(config: RunConfig) -> list[RunRow]:
    rows: list[RunRow] = []
    for seed in config.seeds:
        for preset in config.presets:
            out_csv = config.run_dir / f"submission_{preset}_seed{seed}.csv"
            solver_log = config.run_dir / f"solver_{preset}_seed{seed}.log"
            score_log = config.run_dir / f"score_{preset}_seed{seed}.txt"

            cmd = [
                str(config.solver),
                "--seed",
                str(seed),
                "--preset",
                preset,
                "--output",
                str(out_csv),
            ] + config.solver_args
            _run_and_log(cmd, config.root, solver_log)

            score_text = _run_and_log([str(config.scorer), str(out_csv)], config.root, score_log)
            score = _parse_score(score_text)
            rows.append(
                RunRow(
                    seed=seed,
                    preset=preset,
                    score=score,
                    submission=out_csv,
                )
            )

    return rows


def _select_best(rows: list[RunRow]) -> RunRow:
    if not rows:
        raise SystemExit("No runs completed.")
    return min(rows, key=lambda r: r.score)


def _write_outputs(config: RunConfig, rows: list[RunRow], best: RunRow) -> None:
    config.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best.submission, config.output)

    (config.run_dir / "selected.csv").write_text(
        f"{best.seed},{best.preset},{best.score:.9f},{best.submission.name}\n",
        encoding="utf-8",
    )

    scores_csv = config.run_dir / "scores.csv"
    scores_lines = ["seed,preset,score,submission\n"]
    for row in rows:
        scores_lines.append(
            f"{row.seed},{row.preset},{row.score:.9f},{row.submission.name}\n"
        )
    scores_csv.write_text("".join(scores_lines), encoding="utf-8")

    summary = [
        "Auto tessellation summary\n",
        "\n",
        "seed,preset,score,submission\n",
    ]
    for row in rows:
        summary.append(
            f"{row.seed},{row.preset},{row.score:.9f},{row.submission.name}\n"
        )
    summary.append("\n")
    summary.append(
        f"best_seed,{best.seed}\n"
        f"best_preset,{best.preset}\n"
        f"best_score,{best.score:.9f}\n"
        f"best_submission,{best.submission.name}\n"
        f"final_output,{config.output}\n"
    )
    (config.run_dir / "summary.md").write_text("".join(summary), encoding="utf-8")


def _print_summary(config: RunConfig, best: RunRow) -> None:
    print(f"Best preset: {best.preset} (seed={best.seed}) score={best.score:.9f}")
    print(f"Output: {config.output}")
    print(f"Run dir: {config.run_dir}")


def main() -> int:
    args = _parse_args()
    config = _build_config(args)
    rows = _run_sweep(config)
    best = _select_best(rows)
    _write_outputs(config, rows, best)
    _print_summary(config, best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
