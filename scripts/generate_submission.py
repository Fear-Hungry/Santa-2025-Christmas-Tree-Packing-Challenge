#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geom_np import shift_poses_to_origin  # noqa: E402
from lattice import lattice_poses  # noqa: E402
from tree_data import TREE_POINTS  # noqa: E402


def _format_val(value: float) -> str:
    return f"s{value:.17f}"


def _grid_initial_poses(n: int, spacing: float) -> np.ndarray:
    cols = int(math.ceil(math.sqrt(n)))
    poses = np.zeros((n, 3), dtype=float)
    for i in range(n):
        row = i // cols
        col = i % cols
        poses[i] = (col * spacing, row * spacing, 0.0)
    return poses


_JAX_AVAILABLE = None


def _run_sa(
    n: int,
    *,
    seed: int,
    batch_size: int,
    n_steps: int,
    trans_sigma: float,
    rot_sigma: float,
    rot_prob: float,
) -> np.ndarray | None:
    global _JAX_AVAILABLE
    if _JAX_AVAILABLE is False:
        return None
    try:
        import jax
        import jax.numpy as jnp
        _JAX_AVAILABLE = True
    except Exception:
        _JAX_AVAILABLE = False
        return None

    from optimizer import run_sa_batch  # noqa: E402
    from geom_np import polygon_radius  # noqa: E402

    points = np.array(TREE_POINTS, dtype=float)
    radius = polygon_radius(points)
    spacing = 2.0 * radius * 1.2

    initial = _grid_initial_poses(n, spacing)
    initial_batch = jnp.tile(jnp.array(initial)[None, :, :], (batch_size, 1, 1))

    key = jax.random.PRNGKey(seed)
    best_poses, best_scores = run_sa_batch(
        key,
        n_steps,
        n,
        initial_batch,
        trans_sigma=trans_sigma,
        rot_sigma=rot_sigma,
        rot_prob=rot_prob,
    )
    best_scores.block_until_ready()
    best_idx = int(jnp.argmin(best_scores))
    poses = np.array(best_poses[best_idx])
    return shift_poses_to_origin(points, poses)


def solve_n(
    n: int,
    *,
    seed: int,
    lattice_pattern: str,
    lattice_margin: float,
    lattice_rotate_deg: float,
    sa_nmax: int,
    sa_batch_size: int,
    sa_steps: int,
    sa_trans_sigma: float,
    sa_rot_sigma: float,
    sa_rot_prob: float,
) -> np.ndarray:
    if n <= sa_nmax:
        poses = _run_sa(
            n,
            seed=seed,
            batch_size=sa_batch_size,
            n_steps=sa_steps,
            trans_sigma=sa_trans_sigma,
            rot_sigma=sa_rot_sigma,
            rot_prob=sa_rot_prob,
        )
        if poses is not None:
            return poses

    return lattice_poses(
        n,
        pattern=lattice_pattern,
        margin=lattice_margin,
        rotate_deg=lattice_rotate_deg,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate submission.csv (hybrid SA + lattice)")
    ap.add_argument("--out", type=Path, default=ROOT / "submission.csv", help="Output CSV path")
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle n (default: 200)")
    ap.add_argument("--seed", type=int, default=1, help="Base seed for SA")

    ap.add_argument("--sa-nmax", type=int, default=50, help="Use SA for n <= this threshold")
    ap.add_argument("--sa-batch", type=int, default=64, help="SA batch size")
    ap.add_argument("--sa-steps", type=int, default=400, help="SA steps per puzzle")
    ap.add_argument("--sa-trans-sigma", type=float, default=0.2, help="SA translation step scale")
    ap.add_argument("--sa-rot-sigma", type=float, default=15.0, help="SA rotation step scale (deg)")
    ap.add_argument("--sa-rot-prob", type=float, default=0.3, help="SA rotation move probability")

    ap.add_argument("--lattice-pattern", type=str, default="hex", choices=["hex", "square"])
    ap.add_argument("--lattice-margin", type=float, default=0.02, help="Relative spacing margin")
    ap.add_argument("--lattice-rotate", type=float, default=0.0, help="Constant rotation (deg)")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "deg"])

        for n in range(1, args.nmax + 1):
            poses = solve_n(
                n,
                seed=args.seed + n,
                lattice_pattern=args.lattice_pattern,
                lattice_margin=args.lattice_margin,
                lattice_rotate_deg=args.lattice_rotate,
                sa_nmax=args.sa_nmax,
                sa_batch_size=args.sa_batch,
                sa_steps=args.sa_steps,
                sa_trans_sigma=args.sa_trans_sigma,
                sa_rot_sigma=args.sa_rot_sigma,
                sa_rot_prob=args.sa_rot_prob,
            )

            # Ensure numeric array
            poses = np.array(poses, dtype=float)
            poses[:, 2] = np.mod(poses[:, 2], 360.0)

            for i, (x, y, deg) in enumerate(poses):
                writer.writerow([f"{n:03d}_{i}", _format_val(x), _format_val(y), _format_val(deg)])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
