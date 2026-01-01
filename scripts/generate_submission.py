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

from geom_np import polygon_radius, shift_poses_to_origin, transform_polygon  # noqa: E402
from lattice import lattice_poses  # noqa: E402
from scoring import polygons_intersect  # noqa: E402
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
    initial_poses: np.ndarray | None = None,
    objective: str = "packing",
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

    if initial_poses is None:
        initial = _grid_initial_poses(n, spacing)
    else:
        initial = np.array(initial_poses, dtype=float)
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
        objective=objective,
    )
    best_scores.block_until_ready()
    best_idx = int(jnp.argmin(best_scores))
    poses = np.array(best_poses[best_idx])
    return shift_poses_to_origin(points, poses)


def _run_sa_guided(
    n: int,
    *,
    model_path: Path,
    seed: int,
    batch_size: int,
    n_steps: int,
    trans_sigma: float,
    rot_sigma: float,
    rot_prob: float,
    objective: str,
    initial_poses: np.ndarray | None = None,
    policy_prob: float = 1.0,
    policy_pmax: float = 0.05,
) -> np.ndarray | None:
    try:
        import jax
        import jax.numpy as jnp
    except Exception:
        return None

    from optimizer import run_sa_batch_guided  # noqa: E402
    from l2o import L2OConfig, load_params_npz  # noqa: E402
    from geom_np import polygon_radius  # noqa: E402

    points = np.array(TREE_POINTS, dtype=float)
    radius = polygon_radius(points)
    spacing = 2.0 * radius * 1.2

    if initial_poses is None:
        initial = _grid_initial_poses(n, spacing)
    else:
        initial = np.array(initial_poses, dtype=float)
    initial_batch = jnp.tile(jnp.array(initial)[None, :, :], (batch_size, 1, 1))

    params, meta = load_params_npz(model_path)

    def _meta_bool(value, default: bool) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return bool(int(value))
        if isinstance(value, np.ndarray) and value.shape == ():
            return bool(value.item())
        return default

    def _meta_float(value, default: float) -> float:
        if isinstance(value, (float, np.floating)):
            return float(value)
        if isinstance(value, (int, np.integer)):
            return float(value)
        if isinstance(value, np.ndarray) and value.shape == ():
            return float(value.item())
        return default

    hidden = int(meta.get("hidden", 32)) if hasattr(meta.get("hidden", 32), "__int__") else 32
    policy = str(meta.get("policy", "mlp"))
    knn_k = int(meta.get("knn_k", 4)) if hasattr(meta.get("knn_k", 4), "__int__") else 4
    mlp_depth = int(meta.get("mlp_depth", 1)) if hasattr(meta.get("mlp_depth", 1), "__int__") else 1
    gnn_steps = int(meta.get("gnn_steps", 1)) if hasattr(meta.get("gnn_steps", 1), "__int__") else 1
    gnn_attention = _meta_bool(meta.get("gnn_attention", False), False)
    feature_mode = str(meta.get("feature_mode", "raw"))
    action_scale = _meta_float(meta.get("action_scale", 1.0), 1.0)

    config = L2OConfig(
        hidden_size=hidden,
        policy=policy,
        knn_k=knn_k,
        mlp_depth=mlp_depth,
        gnn_steps=gnn_steps,
        gnn_attention=gnn_attention,
        feature_mode=feature_mode,
        action_scale=action_scale,
        action_noise=False,
    )

    key = jax.random.PRNGKey(seed)
    best_poses, best_scores = run_sa_batch_guided(
        key,
        n_steps,
        n,
        initial_batch,
        params,
        config,
        trans_sigma=trans_sigma,
        rot_sigma=rot_sigma,
        rot_prob=rot_prob,
        objective=objective,
        policy_prob=policy_prob,
        policy_pmax=policy_pmax,
    )
    best_scores.block_until_ready()
    best_idx = int(jnp.argmin(best_scores))
    poses = np.array(best_poses[best_idx])
    return shift_poses_to_origin(points, poses)


def _run_l2o(
    n: int,
    *,
    model_path: Path,
    seed: int,
    steps: int,
    trans_sigma: float,
    rot_sigma: float,
    deterministic: bool,
    initial_poses: np.ndarray | None = None,
) -> np.ndarray | None:
    try:
        import jax
        import jax.numpy as jnp
    except Exception:
        return None

    from l2o import L2OConfig, load_params_npz, optimize_with_l2o  # noqa: E402
    from geom_np import polygon_radius  # noqa: E402

    points = np.array(TREE_POINTS, dtype=float)
    radius = polygon_radius(points)
    spacing = 2.0 * radius * 1.2
    if initial_poses is None:
        initial = _grid_initial_poses(n, spacing)
    else:
        initial = np.array(initial_poses, dtype=float)

    params, meta = load_params_npz(model_path)
    policy = meta.get("policy", "mlp")
    knn_k = int(meta.get("knn_k", 4)) if hasattr(meta.get("knn_k", 4), "__int__") else 4
    mlp_depth = int(meta.get("mlp_depth", 1)) if hasattr(meta.get("mlp_depth", 1), "__int__") else 1
    gnn_steps = int(meta.get("gnn_steps", 1)) if hasattr(meta.get("gnn_steps", 1), "__int__") else 1
    feature_mode = str(meta.get("feature_mode", "raw"))
    def _meta_bool(value, default: bool) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return bool(int(value))
        if isinstance(value, np.ndarray) and value.shape == ():
            return bool(value.item())
        return default

    gnn_attention = _meta_bool(meta.get("gnn_attention", False), False)
    hidden = int(meta.get("hidden", 32)) if hasattr(meta.get("hidden", 32), "__int__") else 32
    config = L2OConfig(
        hidden_size=hidden,
        policy=str(policy),
        knn_k=knn_k,
        mlp_depth=mlp_depth,
        gnn_steps=gnn_steps,
        gnn_attention=gnn_attention,
        feature_mode=feature_mode,
        trans_sigma=trans_sigma,
        rot_sigma=rot_sigma,
        action_noise=not deterministic,
    )
    key = jax.random.PRNGKey(seed)
    poses = optimize_with_l2o(
        key,
        params,
        jnp.array(initial),
        steps,
        config,
    )
    poses = np.array(poses)
    poses = shift_poses_to_origin(points, poses)
    if _has_overlaps(points, poses):
        return None
    return poses


def _has_overlaps(points: np.ndarray, poses: np.ndarray) -> bool:
    poses = np.array(poses, dtype=float)
    if poses.shape[0] <= 1:
        return False
    centers = poses[:, :2]
    rad = float(polygon_radius(points))
    thr2 = (2.0 * rad) ** 2
    polys = [None] * poses.shape[0]
    for i in range(poses.shape[0]):
        polys[i] = transform_polygon(points, poses[i])
    for i in range(poses.shape[0]):
        for j in range(i + 1, poses.shape[0]):
            dx = float(centers[i, 0] - centers[j, 0])
            dy = float(centers[i, 1] - centers[j, 1])
            if dx * dx + dy * dy > thr2:
                continue
            if polygons_intersect(polys[i], polys[j]):
                return True
    return False


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
    sa_objective: str,
    meta_init_model: Path | None,
    heatmap_model: Path | None,
    heatmap_nmax: int,
    heatmap_steps: int,
    l2o_model: Path | None,
    l2o_init: str,
    l2o_nmax: int,
    l2o_steps: int,
    l2o_trans_sigma: float,
    l2o_rot_sigma: float,
    l2o_deterministic: bool,
    refine_nmin: int,
    refine_batch_size: int,
    refine_steps: int,
    refine_trans_sigma: float,
    refine_rot_sigma: float,
    refine_rot_prob: float,
    refine_objective: str,
    guided_model: Path | None,
    guided_prob: float,
    guided_pmax: float,
) -> np.ndarray:
    base: np.ndarray | None = None

    if heatmap_model is not None and n <= heatmap_nmax:
        try:
            from heatmap_meta import HeatmapConfig, heatmap_search, load_params  # noqa: E402
        except Exception:
            heatmap_model = None
        if heatmap_model is not None:
            params, meta = load_params(heatmap_model)
            config = HeatmapConfig(
                hidden_size=int(meta.get("hidden", 32)) if hasattr(meta.get("hidden", 32), "__int__") else 32,
                policy=str(meta.get("policy", "gnn")),
                knn_k=int(meta.get("knn_k", 4)) if hasattr(meta.get("knn_k", 4), "__int__") else 4,
                heatmap_lr=float(meta.get("heatmap_lr", 0.1)),
                trans_sigma=float(meta.get("trans_sigma", 0.2)),
                rot_sigma=float(meta.get("rot_sigma", 10.0)),
            )
            points = np.array(TREE_POINTS, dtype=float)
            radius = polygon_radius(points)
            spacing = 2.0 * radius * 1.2
            base = _grid_initial_poses(n, spacing)
            rng = np.random.default_rng(seed)
            poses, _ = heatmap_search(params, base, config, heatmap_steps, rng)
            base = poses

    if base is None and l2o_model is not None and n <= l2o_nmax:
        l2o_initial = None
        if l2o_init == "lattice":
            l2o_initial = lattice_poses(
                n,
                pattern=lattice_pattern,
                margin=lattice_margin,
                rotate_deg=lattice_rotate_deg,
            )
        poses = _run_l2o(
            n,
            model_path=l2o_model,
            seed=seed,
            steps=l2o_steps,
            trans_sigma=l2o_trans_sigma,
            rot_sigma=l2o_rot_sigma,
            deterministic=l2o_deterministic,
            initial_poses=l2o_initial,
        )
        if poses is not None:
            base = poses

    if base is None and n <= sa_nmax:
        init_override = None
        if meta_init_model is not None:
            try:
                import jax
                import jax.numpy as jnp
            except Exception:
                meta_init_model = None
            if meta_init_model is not None:
                from meta_init import MetaInitConfig, apply_meta_init, load_meta_params  # noqa: E402
                from geom_np import polygon_radius  # noqa: E402
                points = np.array(TREE_POINTS, dtype=float)
                radius = polygon_radius(points)
                spacing = 2.0 * radius * 1.2
                base = _grid_initial_poses(n, spacing)
                params, meta = load_meta_params(meta_init_model)
                config = MetaInitConfig(
                    hidden_size=int(meta.get("hidden", 32)) if hasattr(meta.get("hidden", 32), "__int__") else 32,
                    delta_xy=float(meta.get("delta_xy", 0.2)),
                    delta_theta=float(meta.get("delta_theta", 10.0)),
                )
                init_override = np.array(apply_meta_init(params, jnp.array(base), config))
        if guided_model is not None:
            poses = _run_sa_guided(
                n,
                model_path=guided_model,
                seed=seed,
                batch_size=sa_batch_size,
                n_steps=sa_steps,
                trans_sigma=sa_trans_sigma,
                rot_sigma=sa_rot_sigma,
                rot_prob=sa_rot_prob,
                initial_poses=init_override,
                objective=sa_objective,
                policy_prob=guided_prob,
                policy_pmax=guided_pmax,
            )
        else:
            poses = _run_sa(
                n,
                seed=seed,
                batch_size=sa_batch_size,
                n_steps=sa_steps,
                trans_sigma=sa_trans_sigma,
                rot_sigma=sa_rot_sigma,
                rot_prob=sa_rot_prob,
                initial_poses=init_override,
                objective=sa_objective,
            )
        if poses is not None:
            base = poses

    if base is None:
        base = lattice_poses(
            n,
            pattern=lattice_pattern,
            margin=lattice_margin,
            rotate_deg=lattice_rotate_deg,
        )

    if refine_steps > 0 and n >= refine_nmin:
        if guided_model is not None:
            refined = _run_sa_guided(
                n,
                model_path=guided_model,
                seed=seed,
                batch_size=refine_batch_size,
                n_steps=refine_steps,
                trans_sigma=refine_trans_sigma,
                rot_sigma=refine_rot_sigma,
                rot_prob=refine_rot_prob,
                initial_poses=base,
                objective=refine_objective,
                policy_prob=guided_prob,
                policy_pmax=guided_pmax,
            )
        else:
            refined = _run_sa(
                n,
                seed=seed,
                batch_size=refine_batch_size,
                n_steps=refine_steps,
                trans_sigma=refine_trans_sigma,
                rot_sigma=refine_rot_sigma,
                rot_prob=refine_rot_prob,
                initial_poses=base,
                objective=refine_objective,
            )
        if refined is not None:
            base = refined

    return base


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
    ap.add_argument("--sa-objective", type=str, default="packing", choices=["packing", "prefix"])
    ap.add_argument("--meta-init-model", type=Path, default=None, help="Meta-init model (.npz) for SA init")
    ap.add_argument("--heatmap-model", type=Path, default=None, help="Heatmap meta-optimizer model (.npz)")
    ap.add_argument("--heatmap-nmax", type=int, default=10, help="Use heatmap for n <= this threshold")
    ap.add_argument("--heatmap-steps", type=int, default=200, help="Heatmap search steps per puzzle")

    ap.add_argument("--l2o-model", type=Path, default=None, help="Path to L2O policy (.npz)")
    ap.add_argument("--l2o-init", type=str, default="grid", choices=["grid", "lattice"], help="Initial poses for L2O")
    ap.add_argument("--l2o-nmax", type=int, default=10, help="Use L2O for n <= this threshold")
    ap.add_argument("--l2o-steps", type=int, default=200, help="L2O rollout steps per puzzle")
    ap.add_argument("--l2o-trans-sigma", type=float, default=0.2, help="L2O translation step scale")
    ap.add_argument("--l2o-rot-sigma", type=float, default=10.0, help="L2O rotation step scale")
    ap.add_argument("--l2o-deterministic", action="store_true", help="Disable L2O action noise")

    ap.add_argument("--lattice-pattern", type=str, default="hex", choices=["hex", "square"])
    ap.add_argument("--lattice-margin", type=float, default=0.02, help="Relative spacing margin")
    ap.add_argument("--lattice-rotate", type=float, default=0.0, help="Constant rotation (deg)")

    ap.add_argument("--refine-nmin", type=int, default=0, help="Refine lattice with SA for n >= this threshold (0=disabled)")
    ap.add_argument("--refine-batch", type=int, default=16, help="Refine SA batch size")
    ap.add_argument("--refine-steps", type=int, default=0, help="Refine SA steps per puzzle (0=disabled)")
    ap.add_argument("--refine-trans-sigma", type=float, default=0.2, help="Refine SA translation step scale")
    ap.add_argument("--refine-rot-sigma", type=float, default=15.0, help="Refine SA rotation step scale (deg)")
    ap.add_argument("--refine-rot-prob", type=float, default=0.3, help="Refine SA rotation move probability")
    ap.add_argument("--refine-objective", type=str, default="packing", choices=["packing", "prefix"])

    ap.add_argument("--guided-model", type=Path, default=None, help="L2O policy (.npz) used as SA proposal generator")
    ap.add_argument("--guided-prob", type=float, default=1.0, help="Probability of using policy proposal (when confident)")
    ap.add_argument("--guided-pmax", type=float, default=0.05, help="Min max-softmax(logits) to consider policy confident")
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
                sa_objective=args.sa_objective,
                meta_init_model=args.meta_init_model,
                heatmap_model=args.heatmap_model,
                heatmap_nmax=args.heatmap_nmax,
                heatmap_steps=args.heatmap_steps,
                l2o_model=args.l2o_model,
                l2o_init=args.l2o_init,
                l2o_nmax=args.l2o_nmax,
                l2o_steps=args.l2o_steps,
                l2o_trans_sigma=args.l2o_trans_sigma,
                l2o_rot_sigma=args.l2o_rot_sigma,
                l2o_deterministic=args.l2o_deterministic,
                refine_nmin=args.refine_nmin,
                refine_batch_size=args.refine_batch,
                refine_steps=args.refine_steps,
                refine_trans_sigma=args.refine_trans_sigma,
                refine_rot_sigma=args.refine_rot_sigma,
                refine_rot_prob=args.refine_rot_prob,
                refine_objective=args.refine_objective,
                guided_model=args.guided_model,
                guided_prob=args.guided_prob,
                guided_pmax=args.guided_pmax,
            )

            # Ensure numeric array
            poses = np.array(poses, dtype=float)
            poses[:, 2] = np.mod(poses[:, 2], 360.0)

            for i, (x, y, deg) in enumerate(poses):
                writer.writerow([f"{n:03d}_{i}", _format_val(x), _format_val(y), _format_val(deg)])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
