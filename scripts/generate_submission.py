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

from geom_np import packing_bbox, packing_score, polygon_radius, shift_poses_to_origin, transform_polygon  # noqa: E402
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
    rot_prob_end: float,
    cooling: str,
    cooling_power: float,
    trans_sigma_nexp: float,
    rot_sigma_nexp: float,
    sigma_nref: float,
    proposal: str = "random",
    smart_prob: float = 1.0,
    smart_beta: float = 8.0,
    smart_drift: float = 1.0,
    smart_noise: float = 0.25,
    overlap_lambda: float = 0.0,
    allow_collisions: bool = False,
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
        rot_prob_end=rot_prob_end,
        cooling=cooling,
        cooling_power=cooling_power,
        trans_sigma_nexp=trans_sigma_nexp,
        rot_sigma_nexp=rot_sigma_nexp,
        sigma_nref=sigma_nref,
        proposal=proposal,
        smart_prob=smart_prob,
        smart_beta=smart_beta,
        smart_drift=smart_drift,
        smart_noise=smart_noise,
        overlap_lambda=overlap_lambda,
        allow_collisions=allow_collisions,
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
    rot_prob_end: float,
    cooling: str,
    cooling_power: float,
    trans_sigma_nexp: float,
    rot_sigma_nexp: float,
    sigma_nref: float,
    proposal: str = "random",
    smart_prob: float = 1.0,
    smart_beta: float = 8.0,
    smart_drift: float = 1.0,
    smart_noise: float = 0.25,
    overlap_lambda: float = 0.0,
    allow_collisions: bool = False,
    objective: str,
    initial_poses: np.ndarray | None = None,
    policy_prob: float = 1.0,
    policy_pmax: float = 0.05,
    policy_prob_end: float = -1.0,
    policy_pmax_end: float = -1.0,
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
        rot_prob_end=rot_prob_end,
        cooling=cooling,
        cooling_power=cooling_power,
        trans_sigma_nexp=trans_sigma_nexp,
        rot_sigma_nexp=rot_sigma_nexp,
        sigma_nref=sigma_nref,
        proposal=proposal,
        smart_prob=smart_prob,
        smart_beta=smart_beta,
        smart_drift=smart_drift,
        smart_noise=smart_noise,
        overlap_lambda=overlap_lambda,
        allow_collisions=allow_collisions,
        objective=objective,
        policy_prob=policy_prob,
        policy_pmax=policy_pmax,
        policy_prob_end=policy_prob_end,
        policy_pmax_end=policy_pmax_end,
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


def _parse_float_list(text: str | None) -> list[float]:
    if text is None:
        return []
    raw = text.strip()
    if not raw:
        return []
    if raw.lower() in {"none", "off", "false"}:
        return []
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _best_lattice_poses(
    n: int,
    *,
    pattern: str,
    margin: float,
    rotate_deg: float,
    rotate_degs: list[float] | None,
) -> np.ndarray:
    if not rotate_degs:
        return lattice_poses(n, pattern=pattern, margin=margin, rotate_deg=rotate_deg)

    points = np.array(TREE_POINTS, dtype=float)
    best_score = float("inf")
    best_poses: np.ndarray | None = None
    for deg in rotate_degs:
        poses = lattice_poses(n, pattern=pattern, margin=margin, rotate_deg=deg)
        s = packing_score(points, poses)
        if s < best_score:
            best_score = s
            best_poses = poses
    assert best_poses is not None
    return best_poses


def _radial_reorder(points: np.ndarray, poses: np.ndarray) -> np.ndarray:
    poses = np.array(poses, dtype=float)
    if poses.shape[0] <= 1:
        return poses

    bbox = packing_bbox(points, poses)
    center = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=float)
    d = poses[:, :2] - center[None, :]
    dist2 = np.sum(d * d, axis=1)

    order = np.argsort(dist2, kind="mergesort")
    return poses[order]


def solve_n(
    n: int,
    *,
    seed: int,
    lattice_pattern: str,
    lattice_margin: float,
    lattice_rotate_deg: float,
    lattice_rotate_degs: list[float] | None,
    sa_nmax: int,
    sa_batch_size: int,
    sa_steps: int,
    sa_trans_sigma: float,
    sa_rot_sigma: float,
    sa_rot_prob: float,
    sa_rot_prob_end: float,
    sa_cooling: str,
    sa_cooling_power: float,
    sa_trans_sigma_nexp: float,
    sa_rot_sigma_nexp: float,
    sa_sigma_nref: float,
    sa_proposal: str,
    sa_smart_prob: float,
    sa_smart_beta: float,
    sa_smart_drift: float,
    sa_smart_noise: float,
    sa_overlap_lambda: float,
    sa_allow_collisions: bool,
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
    refine_rot_prob_end: float,
    refine_cooling: str,
    refine_cooling_power: float,
    refine_trans_sigma_nexp: float,
    refine_rot_sigma_nexp: float,
    refine_sigma_nref: float,
    refine_proposal: str,
    refine_smart_prob: float,
    refine_smart_beta: float,
    refine_smart_drift: float,
    refine_smart_noise: float,
    refine_overlap_lambda: float,
    refine_allow_collisions: bool,
    refine_objective: str,
    hc_nmax: int,
    hc_passes: int,
    hc_step_xy: float,
    hc_step_deg: float,
    ga_nmax: int,
    ga_pop: int,
    ga_gens: int,
    ga_elite_frac: float,
    ga_crossover_prob: float,
    ga_mut_sigma_xy: float,
    ga_mut_sigma_deg: float,
    ga_directed_prob: float,
    ga_directed_step_xy: float,
    ga_directed_k: int,
    ga_repair_iters: int,
    ga_hc_passes: int,
    ga_hc_step_xy: float,
    ga_hc_step_deg: float,
    guided_model: Path | None,
    guided_prob: float,
    guided_pmax: float,
    guided_prob_end: float,
    guided_pmax_end: float,
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
                t_start=float(meta.get("t_start", 1.0)),
                t_end=float(meta.get("t_end", 0.001)),
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
            l2o_initial = _best_lattice_poses(
                n,
                pattern=lattice_pattern,
                margin=lattice_margin,
                rotate_deg=lattice_rotate_deg,
                rotate_degs=lattice_rotate_degs,
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
                points = np.array(TREE_POINTS, dtype=float)
                radius = polygon_radius(points)
                spacing = 2.0 * radius * 1.2
                grid_base = _grid_initial_poses(n, spacing)
                params, meta = load_meta_params(meta_init_model)
                config = MetaInitConfig(
                    hidden_size=int(meta.get("hidden", 32)) if hasattr(meta.get("hidden", 32), "__int__") else 32,
                    delta_xy=float(meta.get("delta_xy", 0.2)),
                    delta_theta=float(meta.get("delta_theta", 10.0)),
                )
                init_override = np.array(apply_meta_init(params, jnp.array(grid_base), config))
                if _has_overlaps(points, init_override):
                    init_override = grid_base
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
                rot_prob_end=sa_rot_prob_end,
                cooling=sa_cooling,
                cooling_power=sa_cooling_power,
                trans_sigma_nexp=sa_trans_sigma_nexp,
                rot_sigma_nexp=sa_rot_sigma_nexp,
                sigma_nref=sa_sigma_nref,
                proposal=sa_proposal,
                smart_prob=sa_smart_prob,
                smart_beta=sa_smart_beta,
                smart_drift=sa_smart_drift,
                smart_noise=sa_smart_noise,
                overlap_lambda=sa_overlap_lambda,
                allow_collisions=sa_allow_collisions,
                initial_poses=init_override,
                objective=sa_objective,
                policy_prob=guided_prob,
                policy_pmax=guided_pmax,
                policy_prob_end=guided_prob_end,
                policy_pmax_end=guided_pmax_end,
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
                rot_prob_end=sa_rot_prob_end,
                cooling=sa_cooling,
                cooling_power=sa_cooling_power,
                trans_sigma_nexp=sa_trans_sigma_nexp,
                rot_sigma_nexp=sa_rot_sigma_nexp,
                sigma_nref=sa_sigma_nref,
                proposal=sa_proposal,
                smart_prob=sa_smart_prob,
                smart_beta=sa_smart_beta,
                smart_drift=sa_smart_drift,
                smart_noise=sa_smart_noise,
                overlap_lambda=sa_overlap_lambda,
                allow_collisions=sa_allow_collisions,
                initial_poses=init_override,
                objective=sa_objective,
            )
        if poses is not None:
            base = poses

    if base is None:
        base = _best_lattice_poses(
            n,
            pattern=lattice_pattern,
            margin=lattice_margin,
            rotate_deg=lattice_rotate_deg,
            rotate_degs=lattice_rotate_degs,
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
                rot_prob_end=refine_rot_prob_end,
                cooling=refine_cooling,
                cooling_power=refine_cooling_power,
                trans_sigma_nexp=refine_trans_sigma_nexp,
                rot_sigma_nexp=refine_rot_sigma_nexp,
                sigma_nref=refine_sigma_nref,
                proposal=refine_proposal,
                smart_prob=refine_smart_prob,
                smart_beta=refine_smart_beta,
                smart_drift=refine_smart_drift,
                smart_noise=refine_smart_noise,
                overlap_lambda=refine_overlap_lambda,
                allow_collisions=refine_allow_collisions,
                initial_poses=base,
                objective=refine_objective,
                policy_prob=guided_prob,
                policy_pmax=guided_pmax,
                policy_prob_end=guided_prob_end,
                policy_pmax_end=guided_pmax_end,
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
                rot_prob_end=refine_rot_prob_end,
                cooling=refine_cooling,
                cooling_power=refine_cooling_power,
                trans_sigma_nexp=refine_trans_sigma_nexp,
                rot_sigma_nexp=refine_rot_sigma_nexp,
                sigma_nref=refine_sigma_nref,
                proposal=refine_proposal,
                smart_prob=refine_smart_prob,
                smart_beta=refine_smart_beta,
                smart_drift=refine_smart_drift,
                smart_noise=refine_smart_noise,
                overlap_lambda=refine_overlap_lambda,
                allow_collisions=refine_allow_collisions,
                initial_poses=base,
                objective=refine_objective,
            )
        if refined is not None:
            base = refined

    points = np.array(TREE_POINTS, dtype=float)

    if ga_gens > 0 and ga_nmax > 0 and n <= ga_nmax:
        from postopt_np import genetic_optimize  # noqa: E402

        base = genetic_optimize(
            points,
            [base],
            seed=seed,
            pop_size=ga_pop,
            generations=ga_gens,
            elite_frac=ga_elite_frac,
            crossover_prob=ga_crossover_prob,
            mutation_sigma_xy=ga_mut_sigma_xy,
            mutation_sigma_deg=ga_mut_sigma_deg,
            directed_mut_prob=ga_directed_prob,
            directed_step_xy=ga_directed_step_xy,
            directed_k=ga_directed_k,
            repair_iters=ga_repair_iters,
            hill_climb_passes=ga_hc_passes,
            hill_climb_step_xy=ga_hc_step_xy,
            hill_climb_step_deg=ga_hc_step_deg,
        )

    if hc_passes > 0 and hc_nmax > 0 and n <= hc_nmax:
        from postopt_np import hill_climb  # noqa: E402

        base = hill_climb(
            points,
            base,
            step_xy=hc_step_xy,
            step_deg=hc_step_deg,
            max_passes=hc_passes,
        )

    return base


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate submission.csv (hybrid SA + lattice)")
    ap.add_argument("--out", type=Path, default=ROOT / "submission.csv", help="Output CSV path")
    ap.add_argument("--nmax", type=int, default=200, help="Max puzzle n (default: 200)")
    ap.add_argument("--seed", type=int, default=1, help="Base seed for SA")

    ap.add_argument(
        "--mother-prefix",
        action="store_true",
        help="Solve once for N=nmax and emit radial prefixes for n=1..nmax (nested solutions).",
    )

    ap.add_argument("--sa-nmax", type=int, default=50, help="Use SA for n <= this threshold")
    ap.add_argument("--sa-batch", type=int, default=64, help="SA batch size")
    ap.add_argument("--sa-steps", type=int, default=400, help="SA steps per puzzle")
    ap.add_argument("--sa-trans-sigma", type=float, default=0.2, help="SA translation step scale")
    ap.add_argument("--sa-rot-sigma", type=float, default=15.0, help="SA rotation step scale (deg)")
    ap.add_argument("--sa-rot-prob", type=float, default=0.3, help="SA rotation move probability")
    ap.add_argument(
        "--sa-rot-prob-end",
        type=float,
        default=-1.0,
        help="Final SA rotation move probability (linear schedule; -1 keeps constant).",
    )
    ap.add_argument("--sa-cooling", type=str, default="geom", choices=["geom", "linear", "log"])
    ap.add_argument("--sa-cooling-power", type=float, default=1.0, help="Power on anneal fraction (>=1 slows early cooling).")
    ap.add_argument("--sa-trans-nexp", type=float, default=0.0, help="Scale trans_sigma by (n/nref)^nexp.")
    ap.add_argument("--sa-rot-nexp", type=float, default=0.0, help="Scale rot_sigma by (n/nref)^nexp.")
    ap.add_argument("--sa-sigma-nref", type=float, default=50.0, help="Reference n for sigma scaling.")
    ap.add_argument("--sa-objective", type=str, default="packing", choices=["packing", "prefix"])
    ap.add_argument(
        "--sa-proposal",
        type=str,
        default="random",
        choices=["random", "bbox_inward", "bbox", "inward", "smart", "mixed"],
        help="SA proposal mode. 'bbox_inward/smart' targets boundary trees; 'mixed' blends with random.",
    )
    ap.add_argument("--sa-smart-prob", type=float, default=1.0, help="For proposal=mixed: probability of smart move.")
    ap.add_argument("--sa-smart-beta", type=float, default=8.0, help="Edge focus strength (higher=more boundary-biased).")
    ap.add_argument("--sa-smart-drift", type=float, default=1.0, help="Inward drift multiplier (translation moves).")
    ap.add_argument("--sa-smart-noise", type=float, default=0.25, help="Noise multiplier for smart inward moves.")
    ap.add_argument("--sa-overlap-lambda", type=float, default=0.0, help="Energy penalty weight for circle overlap (0 disables).")
    ap.add_argument("--sa-allow-collisions", action="store_true", help="Allow accepting colliding states (best kept feasible).")
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
    ap.add_argument(
        "--lattice-rotations",
        type=str,
        default="0,15,30",
        help="Comma-separated rotations to try (deg). If set, pick the best lattice per n.",
    )

    ap.add_argument("--refine-nmin", type=int, default=0, help="Refine lattice with SA for n >= this threshold (0=disabled)")
    ap.add_argument("--refine-batch", type=int, default=16, help="Refine SA batch size")
    ap.add_argument("--refine-steps", type=int, default=0, help="Refine SA steps per puzzle (0=disabled)")
    ap.add_argument("--refine-trans-sigma", type=float, default=0.2, help="Refine SA translation step scale")
    ap.add_argument("--refine-rot-sigma", type=float, default=15.0, help="Refine SA rotation step scale (deg)")
    ap.add_argument("--refine-rot-prob", type=float, default=0.3, help="Refine SA rotation move probability")
    ap.add_argument(
        "--refine-rot-prob-end",
        type=float,
        default=-1.0,
        help="Final refine rotation move probability (linear schedule; -1 keeps constant).",
    )
    ap.add_argument("--refine-cooling", type=str, default="geom", choices=["geom", "linear", "log"])
    ap.add_argument("--refine-cooling-power", type=float, default=1.0)
    ap.add_argument("--refine-trans-nexp", type=float, default=0.0)
    ap.add_argument("--refine-rot-nexp", type=float, default=0.0)
    ap.add_argument("--refine-sigma-nref", type=float, default=50.0)
    ap.add_argument("--refine-objective", type=str, default="packing", choices=["packing", "prefix"])
    ap.add_argument(
        "--refine-proposal",
        type=str,
        default="random",
        choices=["random", "bbox_inward", "bbox", "inward", "smart", "mixed"],
        help="Refine SA proposal mode. 'bbox_inward/smart' targets boundary trees; 'mixed' blends with random.",
    )
    ap.add_argument("--refine-smart-prob", type=float, default=1.0, help="For proposal=mixed: probability of smart move.")
    ap.add_argument("--refine-smart-beta", type=float, default=8.0, help="Edge focus strength (higher=more boundary-biased).")
    ap.add_argument("--refine-smart-drift", type=float, default=1.0, help="Inward drift multiplier (translation moves).")
    ap.add_argument("--refine-smart-noise", type=float, default=0.25, help="Noise multiplier for smart inward moves.")
    ap.add_argument("--refine-overlap-lambda", type=float, default=0.0, help="Energy penalty weight for circle overlap (0 disables).")
    ap.add_argument("--refine-allow-collisions", action="store_true", help="Allow accepting colliding states (best kept feasible).")

    ap.add_argument("--guided-model", type=Path, default=None, help="L2O policy (.npz) used as SA proposal generator")
    ap.add_argument("--guided-prob", type=float, default=1.0, help="Probability of using policy proposal (when confident)")
    ap.add_argument("--guided-pmax", type=float, default=0.05, help="Min max-softmax(logits) to consider policy confident")
    ap.add_argument(
        "--guided-prob-end",
        type=float,
        default=-1.0,
        help="Final probability of using policy proposal (linear schedule; -1 keeps constant).",
    )
    ap.add_argument(
        "--guided-pmax-end",
        type=float,
        default=-1.0,
        help="Final policy confidence threshold (linear schedule; -1 keeps constant).",
    )

    ap.add_argument("--hc-nmax", type=int, default=0, help="Apply deterministic hill-climb for n <= this threshold (0=disabled).")
    ap.add_argument("--hc-passes", type=int, default=2, help="Hill-climb passes over trees.")
    ap.add_argument("--hc-step-xy", type=float, default=0.01, help="Hill-climb translation step.")
    ap.add_argument("--hc-step-deg", type=float, default=2.0, help="Hill-climb rotation step (deg).")

    ap.add_argument("--ga-nmax", type=int, default=0, help="Apply GA refinement for n <= this threshold (0=disabled).")
    ap.add_argument("--ga-pop", type=int, default=24, help="GA population size.")
    ap.add_argument("--ga-gens", type=int, default=20, help="GA generations.")
    ap.add_argument("--ga-elite-frac", type=float, default=0.25, help="Elite fraction carried over each generation.")
    ap.add_argument("--ga-crossover-prob", type=float, default=0.5, help="Crossover probability.")
    ap.add_argument("--ga-mut-sigma-xy", type=float, default=0.01, help="Mutation translation sigma.")
    ap.add_argument("--ga-mut-sigma-deg", type=float, default=2.0, help="Mutation rotation sigma (deg).")
    ap.add_argument("--ga-directed-prob", type=float, default=0.5, help="Probability of using directed (bbox-inward) mutation.")
    ap.add_argument("--ga-directed-step-xy", type=float, default=0.02, help="Directed mutation drift step.")
    ap.add_argument("--ga-directed-k", type=int, default=8, help="Directed mutation samples from the k most boundary-ish trees.")
    ap.add_argument("--ga-repair-iters", type=int, default=200, help="Max repair iterations for colliding children.")
    ap.add_argument("--ga-hc-passes", type=int, default=0, help="Optional hill-climb passes applied inside GA (0=disabled).")
    ap.add_argument("--ga-hc-step-xy", type=float, default=0.01)
    ap.add_argument("--ga-hc-step-deg", type=float, default=2.0)
    args = ap.parse_args()
    lattice_rotate_degs = _parse_float_list(args.lattice_rotations)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "deg"])

        points = np.array(TREE_POINTS, dtype=float)

        if args.mother_prefix:
            mother = solve_n(
                args.nmax,
                seed=args.seed + args.nmax,
                lattice_pattern=args.lattice_pattern,
                lattice_margin=args.lattice_margin,
                lattice_rotate_deg=args.lattice_rotate,
                lattice_rotate_degs=lattice_rotate_degs,
                sa_nmax=args.sa_nmax,
                sa_batch_size=args.sa_batch,
                sa_steps=args.sa_steps,
                sa_trans_sigma=args.sa_trans_sigma,
                sa_rot_sigma=args.sa_rot_sigma,
                sa_rot_prob=args.sa_rot_prob,
                sa_rot_prob_end=args.sa_rot_prob_end,
                sa_cooling=args.sa_cooling,
                sa_cooling_power=args.sa_cooling_power,
                sa_trans_sigma_nexp=args.sa_trans_nexp,
                sa_rot_sigma_nexp=args.sa_rot_nexp,
                sa_sigma_nref=args.sa_sigma_nref,
                sa_proposal=args.sa_proposal,
                sa_smart_prob=args.sa_smart_prob,
                sa_smart_beta=args.sa_smart_beta,
                sa_smart_drift=args.sa_smart_drift,
                sa_smart_noise=args.sa_smart_noise,
                sa_overlap_lambda=args.sa_overlap_lambda,
                sa_allow_collisions=args.sa_allow_collisions,
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
                refine_rot_prob_end=args.refine_rot_prob_end,
                refine_cooling=args.refine_cooling,
                refine_cooling_power=args.refine_cooling_power,
                refine_trans_sigma_nexp=args.refine_trans_nexp,
                refine_rot_sigma_nexp=args.refine_rot_nexp,
                refine_sigma_nref=args.refine_sigma_nref,
                refine_proposal=args.refine_proposal,
                refine_smart_prob=args.refine_smart_prob,
                refine_smart_beta=args.refine_smart_beta,
                refine_smart_drift=args.refine_smart_drift,
                refine_smart_noise=args.refine_smart_noise,
                refine_overlap_lambda=args.refine_overlap_lambda,
                refine_allow_collisions=args.refine_allow_collisions,
                refine_objective=args.refine_objective,
                hc_nmax=args.hc_nmax,
                hc_passes=args.hc_passes,
                hc_step_xy=args.hc_step_xy,
                hc_step_deg=args.hc_step_deg,
                ga_nmax=args.ga_nmax,
                ga_pop=args.ga_pop,
                ga_gens=args.ga_gens,
                ga_elite_frac=args.ga_elite_frac,
                ga_crossover_prob=args.ga_crossover_prob,
                ga_mut_sigma_xy=args.ga_mut_sigma_xy,
                ga_mut_sigma_deg=args.ga_mut_sigma_deg,
                ga_directed_prob=args.ga_directed_prob,
                ga_directed_step_xy=args.ga_directed_step_xy,
                ga_directed_k=args.ga_directed_k,
                ga_repair_iters=args.ga_repair_iters,
                ga_hc_passes=args.ga_hc_passes,
                ga_hc_step_xy=args.ga_hc_step_xy,
                ga_hc_step_deg=args.ga_hc_step_deg,
                guided_model=args.guided_model,
                guided_prob=args.guided_prob,
                guided_pmax=args.guided_pmax,
                guided_prob_end=args.guided_prob_end,
                guided_pmax_end=args.guided_pmax_end,
            )

            mother = np.array(mother, dtype=float)
            mother[:, 2] = np.mod(mother[:, 2], 360.0)
            mother = _radial_reorder(points, mother)

            for n in range(1, args.nmax + 1):
                poses = shift_poses_to_origin(points, mother[:n])
                poses = np.array(poses, dtype=float)
                poses[:, 2] = np.mod(poses[:, 2], 360.0)
                for i, (x, y, deg) in enumerate(poses):
                    writer.writerow([f"{n:03d}_{i}", _format_val(x), _format_val(y), _format_val(deg)])
        else:
            for n in range(1, args.nmax + 1):
                poses = solve_n(
                    n,
                    seed=args.seed + n,
                    lattice_pattern=args.lattice_pattern,
                    lattice_margin=args.lattice_margin,
                    lattice_rotate_deg=args.lattice_rotate,
                    lattice_rotate_degs=lattice_rotate_degs,
                    sa_nmax=args.sa_nmax,
                    sa_batch_size=args.sa_batch,
                    sa_steps=args.sa_steps,
                    sa_trans_sigma=args.sa_trans_sigma,
                    sa_rot_sigma=args.sa_rot_sigma,
                    sa_rot_prob=args.sa_rot_prob,
                    sa_rot_prob_end=args.sa_rot_prob_end,
                    sa_cooling=args.sa_cooling,
                    sa_cooling_power=args.sa_cooling_power,
                    sa_trans_sigma_nexp=args.sa_trans_nexp,
                    sa_rot_sigma_nexp=args.sa_rot_nexp,
                    sa_sigma_nref=args.sa_sigma_nref,
                    sa_proposal=args.sa_proposal,
                    sa_smart_prob=args.sa_smart_prob,
                    sa_smart_beta=args.sa_smart_beta,
                    sa_smart_drift=args.sa_smart_drift,
                    sa_smart_noise=args.sa_smart_noise,
                    sa_overlap_lambda=args.sa_overlap_lambda,
                    sa_allow_collisions=args.sa_allow_collisions,
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
                    refine_rot_prob_end=args.refine_rot_prob_end,
                    refine_cooling=args.refine_cooling,
                    refine_cooling_power=args.refine_cooling_power,
                    refine_trans_sigma_nexp=args.refine_trans_nexp,
                    refine_rot_sigma_nexp=args.refine_rot_nexp,
                    refine_sigma_nref=args.refine_sigma_nref,
                    refine_proposal=args.refine_proposal,
                    refine_smart_prob=args.refine_smart_prob,
                    refine_smart_beta=args.refine_smart_beta,
                    refine_smart_drift=args.refine_smart_drift,
                    refine_smart_noise=args.refine_smart_noise,
                    refine_overlap_lambda=args.refine_overlap_lambda,
                    refine_allow_collisions=args.refine_allow_collisions,
                    refine_objective=args.refine_objective,
                    hc_nmax=args.hc_nmax,
                    hc_passes=args.hc_passes,
                    hc_step_xy=args.hc_step_xy,
                    hc_step_deg=args.hc_step_deg,
                    ga_nmax=args.ga_nmax,
                    ga_pop=args.ga_pop,
                    ga_gens=args.ga_gens,
                    ga_elite_frac=args.ga_elite_frac,
                    ga_crossover_prob=args.ga_crossover_prob,
                    ga_mut_sigma_xy=args.ga_mut_sigma_xy,
                    ga_mut_sigma_deg=args.ga_mut_sigma_deg,
                    ga_directed_prob=args.ga_directed_prob,
                    ga_directed_step_xy=args.ga_directed_step_xy,
                    ga_directed_k=args.ga_directed_k,
                    ga_repair_iters=args.ga_repair_iters,
                    ga_hc_passes=args.ga_hc_passes,
                    ga_hc_step_xy=args.ga_hc_step_xy,
                    ga_hc_step_deg=args.ga_hc_step_deg,
                    guided_model=args.guided_model,
                    guided_prob=args.guided_prob,
                    guided_pmax=args.guided_pmax,
                    guided_prob_end=args.guided_prob_end,
                    guided_pmax_end=args.guided_pmax_end,
                )

                poses = np.array(poses, dtype=float)
                poses[:, 2] = np.mod(poses[:, 2], 360.0)

                for i, (x, y, deg) in enumerate(poses):
                    writer.writerow([f"{n:03d}_{i}", _format_val(x), _format_val(y), _format_val(deg)])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
