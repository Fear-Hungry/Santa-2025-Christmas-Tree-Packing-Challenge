#!/usr/bin/env python3

# %% Setup
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
import itertools
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import importlib
import inspect
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Resolve repo root (works whether cwd is repo root or notebooks/)
CWD = Path.cwd()
if (CWD / "src").exists():
    ROOT = CWD
elif (CWD.parent / "src").exists():
    ROOT = CWD.parent
else:
    ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from geom_np import (  # noqa: E402
    packing_score,
    polygon_bbox,
    polygon_radius,
    prefix_score,
    shift_poses_to_origin,
    transform_polygon,
)
import l2o as l2o_mod  # noqa: E402
from optimizer import run_sa_batch  # noqa: E402
import scripts.train_l2o as train_l2o_mod  # noqa: E402
from tree_data import TREE_POINTS  # noqa: E402

train_l2o_mod = importlib.reload(train_l2o_mod)
l2o_mod = importlib.reload(l2o_mod)
L2OConfig = l2o_mod.L2OConfig
load_params_npz = l2o_mod.load_params_npz
save_params_npz = l2o_mod.save_params_npz
optimize_with_l2o = l2o_mod.optimize_with_l2o


def train_model_safe(**kwargs):
    sig = inspect.signature(train_l2o_mod.train_model)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    missing = sorted(set(kwargs) - set(allowed))
    if missing:
        print(f"[warn] train_model ignorou parametros nao suportados: {missing}")
    return train_l2o_mod.train_model(**allowed)


# %% Initial layouts
def grid_initial(n: int, spacing: float) -> np.ndarray:
    cols = int(np.ceil(np.sqrt(n)))
    poses = np.zeros((n, 3), dtype=float)
    for i in range(n):
        row = i // cols
        col = i % cols
        poses[i] = (col * spacing, row * spacing, 0.0)
    return poses


def random_initial(n: int, spacing: float, rng: np.random.Generator, rand_scale: float) -> np.ndarray:
    scale = spacing * math.sqrt(max(n, 1)) * rand_scale
    xy = rng.uniform(-scale, scale, size=(n, 2))
    theta = rng.uniform(0.0, 360.0, size=(n, 1))
    return np.concatenate([xy, theta], axis=1)


def make_initial(
    points: np.ndarray,
    n: int,
    spacing: float,
    seed: int,
    init_mode: str,
    rand_scale: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if init_mode == "grid":
        poses = grid_initial(n, spacing)
    elif init_mode == "random":
        poses = random_initial(n, spacing, rng, rand_scale)
    else:
        poses = grid_initial(n, spacing) if seed % 2 == 0 else random_initial(n, spacing, rng, rand_scale)
    return shift_poses_to_origin(points, poses)


# %% Scoring/plots/utilities
def prefix_packing_score_np(points: np.ndarray, poses: np.ndarray) -> float:
    if poses.shape[0] == 0:
        return 0.0
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    s_values: List[float] = []
    for pose in poses:
        bbox = polygon_bbox(transform_polygon(points, pose))
        min_x = min(min_x, float(bbox[0]))
        min_y = min(min_y, float(bbox[1]))
        max_x = max(max_x, float(bbox[2]))
        max_y = max(max_y, float(bbox[3]))
        width = max_x - min_x
        height = max_y - min_y
        s_values.append(max(width, height))
    return float(prefix_score(s_values))


def plot_packing(poses: np.ndarray, title: str, out_path: Path | None = None) -> None:
    points = np.array(TREE_POINTS, dtype=float)
    plt.figure(figsize=(6, 6))
    for pose in poses:
        poly = transform_polygon(points, pose)
        p = np.vstack([poly, poly[0]])
        plt.plot(p[:, 0], p[:, 1], "g-")
    plt.axis("equal")
    plt.title(title)
    if out_path is not None:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def run_cmd(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}")


def l2o_config_from_meta(meta: Dict[str, object], *, reward: str, deterministic: bool) -> L2OConfig:
    def _get_int(key: str, default: int) -> int:
        val = meta.get(key, default)
        return int(val) if hasattr(val, "__int__") else default

    def _get_float(key: str, default: float) -> float:
        val = meta.get(key, default)
        if isinstance(val, (float, np.floating)):
            return float(val)
        if isinstance(val, (int, np.integer)):
            return float(val)
        if isinstance(val, np.ndarray) and val.shape == ():
            return float(val.item())
        return default

    def _get_bool(key: str, default: bool) -> bool:
        val = meta.get(key, default)
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        if isinstance(val, (int, np.integer)):
            return bool(int(val))
        if isinstance(val, np.ndarray) and val.shape == ():
            return bool(val.item())
        return default

    policy = str(meta.get("policy", "mlp"))
    knn_k = _get_int("knn_k", 4)
    hidden = _get_int("hidden", 32)
    mlp_depth = _get_int("mlp_depth", 1)
    gnn_steps = _get_int("gnn_steps", 1)
    gnn_attention = _get_bool("gnn_attention", False)
    action_scale = _get_float("action_scale", 1.0)
    feature_mode = str(meta.get("feature_mode", "raw"))

    return L2OConfig(
        hidden_size=hidden,
        policy=policy,
        knn_k=knn_k,
        reward=reward,
        mlp_depth=mlp_depth,
        gnn_steps=gnn_steps,
        gnn_attention=gnn_attention,
        feature_mode=feature_mode,
        action_scale=action_scale,
        action_noise=not deterministic,
    )


# %% Evaluation helpers
def evaluate_solver(
    name: str,
    solve_fn: Callable[[int, int], np.ndarray | Tuple[np.ndarray, Dict[str, str]]],
    n_list: Iterable[int],
    seeds: Iterable[int],
    points: np.ndarray,
    split: str,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for n in n_list:
        for seed in seeds:
            result = solve_fn(n, seed)
            info: Dict[str, object] = {}
            if isinstance(result, tuple):
                poses, info = result
            else:
                poses = result
            prefix = prefix_packing_score_np(points, poses)
            pack = packing_score(points, poses)
            row = {
                "split": split,
                "model": name,
                "n": int(n),
                "seed": int(seed),
                "prefix_score": float(prefix),
                "packing_score": float(pack),
            }
            if isinstance(info, dict) and "selected" in info:
                row["selected"] = str(info["selected"])
            rows.append(row)
    return rows


def summarize_results(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, float]]] = {}
    for row in rows:
        key = (row["split"], row["model"], int(row["n"]))
        grouped.setdefault(key, []).append(row)

    summary: List[Dict[str, float]] = []
    for (split, model, n), items in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        prefix_vals = np.array([r["prefix_score"] for r in items], dtype=float)
        pack_vals = np.array([r["packing_score"] for r in items], dtype=float)
        summary.append(
            {
                "split": split,
                "model": model,
                "n": int(n),
                "samples": int(prefix_vals.size),
                "prefix_mean": float(prefix_vals.mean()),
                "prefix_std": float(prefix_vals.std()),
                "packing_mean": float(pack_vals.mean()),
                "packing_std": float(pack_vals.std()),
            }
        )
    return summary


def summarize_overall(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault((row["split"], row["model"]), []).append(row)

    overall: List[Dict[str, float]] = []
    for (split, model), items in sorted(grouped.items()):
        prefix_vals = np.array([r["prefix_score"] for r in items], dtype=float)
        pack_vals = np.array([r["packing_score"] for r in items], dtype=float)
        overall.append(
            {
                "split": split,
                "model": model,
                "samples": int(prefix_vals.size),
                "prefix_mean": float(prefix_vals.mean()),
                "prefix_std": float(prefix_vals.std()),
                "packing_mean": float(pack_vals.mean()),
                "packing_std": float(pack_vals.std()),
            }
        )
    return overall


def challenge_score_from_results(rows: List[Dict[str, float]], model: str, split: str) -> float:
    grouped: Dict[int, List[float]] = {}
    for row in rows:
        if row.get("model") != model or row.get("split") != split:
            continue
        grouped.setdefault(int(row["n"]), []).append(float(row["packing_score"]))
    total = 0.0
    for n in sorted(grouped):
        mean_s = float(np.mean(grouped[n]))
        total += (mean_s * mean_s) / n
    return total


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fieldnames: List[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_eval_artifacts(
    run_dir: Path,
    rows: List[Dict[str, float]],
    summary: List[Dict[str, float]],
    overall: List[Dict[str, float]],
    meta: Dict[str, object],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_dir / "metrics.csv", rows)
    write_csv(run_dir / "per_n.csv", summary)
    write_csv(run_dir / "overall.csv", overall)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    lines = [
        "# L2O evaluation summary",
        "",
        "Lower is better. Prefix score matches the leaderboard aggregate.",
        "",
        "## Overall (mean across n and seeds)",
    ]
    for row in overall:
        lines.append(
            f"- [{row['split']}] {row['model']}: prefix={row['prefix_mean']:.4f} +/- {row['prefix_std']:.4f}, "
            f"packing={row['packing_mean']:.4f} +/- {row['packing_std']:.4f}"
        )
    (run_dir / "summary.md").write_text("\n".join(lines))


def plot_eval_curves(summary: List[Dict[str, float]], out_path: Path | None = None) -> None:
    if not summary:
        return
    grouped: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    for row in summary:
        grouped.setdefault((row["split"], row["model"]), []).append(row)

    plt.figure(figsize=(7, 4))
    for (split, model), items in grouped.items():
        items = sorted(items, key=lambda r: r["n"])
        ns = [r["n"] for r in items]
        means = [r["prefix_mean"] for r in items]
        stds = [r["prefix_std"] for r in items]
        plt.plot(ns, means, marker="o", label=f"{model} ({split})")
        plt.fill_between(ns, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2)
    plt.xlabel("n")
    plt.ylabel("prefix score")
    plt.title("Prefix score vs n")
    plt.legend()
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


# %% Configuracoes
# === Configuracoes ===
TRAIN_N_LIST = [8, 10, 12]
VAL_N_LIST = [6, 9, 11]
TRAIN_STEPS = 400
ROLLOUT_STEPS = 50
BATCH = 64
REWARD = "prefix"  # "packing" ou "prefix"
HIDDEN_SIZE = 32
ACTION_SCALE = 0.05
FEATURE_MODE = "raw"  # raw | bbox_norm
TRAIN_INIT_MODE = "all"  # grid | random | mix | lattice | all
TRAIN_LATTICE_PATTERN = "hex"
TRAIN_LATTICE_MARGIN = 0.02
TRAIN_LATTICE_ROTATE = 0.0
TRAIN_CURRICULUM = False
TRAIN_CURRICULUM_START_MAX = None
TRAIN_CURRICULUM_END_MAX = None
TRAIN_CURRICULUM_STEPS = None

BASELINE_MODE = "ema"  # "batch" (baseline por batch) | "ema" (media movel)
BASELINE_DECAY = 0.9

MLP_DEPTH = 2
GNN_STEPS = 2
GNN_ATTENTION = False

TRAIN_EVAL_SEEDS = [0, 1, 2]
VAL_EVAL_SEEDS = [3, 4, 5]
EVAL_STEPS = 50
INIT_MODE = "grid"  # grid | random | mix
RAND_SCALE = 0.3

SA_STEPS = 300
SA_TRANS_SIGMA = 0.2
SA_ROT_SIGMA = 15.0
SA_ROT_PROB = 0.3
SA_OBJECTIVE = REWARD

RUN_BC_PIPELINE = True
BC_POLICY = "gnn"
BC_KNN_K = 4
BC_RUNS_PER_N = 3
BC_STEPS = 200
BC_TRAIN_STEPS = 200
BC_SEED = 0
BC_INIT_MODE = "all"  # grid | random | mix | lattice | all
BC_RAND_SCALE = 0.3
BC_LATTICE_PATTERN = "hex"
BC_LATTICE_MARGIN = 0.02
BC_LATTICE_ROTATE = 0.0
BC_CURRICULUM = False
BC_CURRICULUM_START_MAX = None
BC_CURRICULUM_END_MAX = None
BC_CURRICULUM_STEPS = None
BC_DATASET_PATH = None  # sobrescreva para reutilizar dataset
BC_POLICY_PATH = None  # sobrescreva para reutilizar policy

RUN_META_TRAIN = True
META_INIT_MODEL_PATH = None
META_TRAIN_STEPS = 30
META_ES_POP = 6
META_SA_STEPS = 150

RUN_HEATMAP_TRAIN = True
HEATMAP_MODEL_PATH = None
HEATMAP_TRAIN_STEPS = 30
HEATMAP_ES_POP = 6
HEATMAP_STEPS = 200

RUN_ENSEMBLE = True
ENSEMBLE_SCORE = "prefix"  # criterio de selecao no ensemble
L2O_REFINE_GRID = False
L2O_REFINE_SA = False
REFINE_STEPS = 100

# %% Treino das politicas (setup)
RUN_DIR = ROOT / "runs" / f"l2o_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

points = np.array(TREE_POINTS, dtype=float)
spacing = 2.0 * polygon_radius(points) * 1.2
VIS_N = TRAIN_N_LIST[0]
init = shift_poses_to_origin(points, grid_initial(VIS_N, spacing))

# %% Treino MLP
mlp_params, mlp_loss = train_model_safe(
    seed=1,
    n_list=TRAIN_N_LIST,
    batch=BATCH,
    train_steps=TRAIN_STEPS,
    steps=ROLLOUT_STEPS,
    hidden_size=HIDDEN_SIZE,
    policy="mlp",
    reward=REWARD,
    action_scale=ACTION_SCALE,
    mlp_depth=MLP_DEPTH,
    gnn_steps=GNN_STEPS,
    gnn_attention=GNN_ATTENTION,
    init_mode=TRAIN_INIT_MODE,
    rand_scale=RAND_SCALE,
    lattice_pattern=TRAIN_LATTICE_PATTERN,
    lattice_margin=TRAIN_LATTICE_MARGIN,
    lattice_rotate=TRAIN_LATTICE_ROTATE,
    baseline_mode=BASELINE_MODE,
    baseline_decay=BASELINE_DECAY,
    curriculum=TRAIN_CURRICULUM,
    curriculum_start_max=TRAIN_CURRICULUM_START_MAX,
    curriculum_end_max=TRAIN_CURRICULUM_END_MAX,
    curriculum_steps=TRAIN_CURRICULUM_STEPS,
    feature_mode=FEATURE_MODE,
    verbose_freq=10,
)

# %% Treino GNN
gnn_params, gnn_loss = train_model_safe(
    seed=2,
    n_list=TRAIN_N_LIST,
    batch=BATCH,
    train_steps=TRAIN_STEPS,
    steps=ROLLOUT_STEPS,
    hidden_size=HIDDEN_SIZE,
    policy="gnn",
    reward=REWARD,
    action_scale=ACTION_SCALE,
    knn_k=4,
    mlp_depth=MLP_DEPTH,
    gnn_steps=GNN_STEPS,
    gnn_attention=GNN_ATTENTION,
    init_mode=TRAIN_INIT_MODE,
    rand_scale=RAND_SCALE,
    lattice_pattern=TRAIN_LATTICE_PATTERN,
    lattice_margin=TRAIN_LATTICE_MARGIN,
    lattice_rotate=TRAIN_LATTICE_ROTATE,
    baseline_mode=BASELINE_MODE,
    baseline_decay=BASELINE_DECAY,
    curriculum=TRAIN_CURRICULUM,
    curriculum_start_max=TRAIN_CURRICULUM_START_MAX,
    curriculum_end_max=TRAIN_CURRICULUM_END_MAX,
    curriculum_steps=TRAIN_CURRICULUM_STEPS,
    feature_mode=FEATURE_MODE,
    verbose_freq=10,
)

# %% Plot losses
plt.figure(figsize=(6, 4))
plt.plot(mlp_loss, label="MLP")
plt.plot(gnn_loss, label="GNN")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("L2O Loss")
plt.legend()
plt.tight_layout()
plt.savefig(RUN_DIR / "loss_curve.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Visualizacao de packing (MLP)
key = jax.random.PRNGKey(0)
config = L2OConfig(
    hidden_size=HIDDEN_SIZE,
    policy="mlp",
    reward=REWARD,
    mlp_depth=MLP_DEPTH,
    gnn_steps=GNN_STEPS,
    gnn_attention=GNN_ATTENTION,
    action_scale=ACTION_SCALE,
    action_noise=False,
)
mlp_poses = optimize_with_l2o(key, mlp_params, jnp.array(init), ROLLOUT_STEPS, config)
plot_packing(np.array(mlp_poses), "MLP packing", RUN_DIR / "mlp_packing.png")

# %% Visualizacao de packing (GNN)
key = jax.random.PRNGKey(1)
config = L2OConfig(
    hidden_size=HIDDEN_SIZE,
    policy="gnn",
    reward=REWARD,
    knn_k=4,
    mlp_depth=MLP_DEPTH,
    gnn_steps=GNN_STEPS,
    gnn_attention=GNN_ATTENTION,
    action_scale=ACTION_SCALE,
    action_noise=False,
)
gnn_poses = optimize_with_l2o(key, gnn_params, jnp.array(init), ROLLOUT_STEPS, config)
plot_packing(np.array(gnn_poses), "GNN packing", RUN_DIR / "gnn_packing.png")

# %% Pipelines opcionais (BC / meta / heatmap)
bc_policy_path: Path | None = Path(BC_POLICY_PATH) if BC_POLICY_PATH else None

# %% Pipeline opcional: BC (imitation learning)
if RUN_BC_PIPELINE:
    bc_dataset = Path(BC_DATASET_PATH) if BC_DATASET_PATH else RUN_DIR / "bc_dataset.npz"
    bc_policy_path = Path(BC_POLICY_PATH) if BC_POLICY_PATH else RUN_DIR / "bc_policy.npz"
    run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "collect_sa_dataset.py"),
            "--n-list",
            ",".join(str(n) for n in TRAIN_N_LIST),
            "--runs-per-n",
            str(BC_RUNS_PER_N),
            "--steps",
            str(BC_STEPS),
            "--seed",
            str(BC_SEED),
            "--init",
            BC_INIT_MODE,
            "--rand-scale",
            str(BC_RAND_SCALE),
            "--lattice-pattern",
            BC_LATTICE_PATTERN,
            "--lattice-margin",
            str(BC_LATTICE_MARGIN),
            "--lattice-rotate",
            str(BC_LATTICE_ROTATE),
            "--out",
            str(bc_dataset),
        ]
    )
    train_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_l2o_bc.py"),
        "--dataset",
        str(bc_dataset),
        "--policy",
        BC_POLICY,
        "--knn-k",
        str(BC_KNN_K),
        "--train-steps",
        str(BC_TRAIN_STEPS),
        "--seed",
        str(BC_SEED + 1),
        "--reward",
        REWARD,
        "--hidden",
        str(HIDDEN_SIZE),
        "--mlp-depth",
        str(MLP_DEPTH),
        "--gnn-steps",
        str(GNN_STEPS),
        "--feature-mode",
        FEATURE_MODE,
        "--out",
        str(bc_policy_path),
    ]
    if BC_CURRICULUM:
        train_cmd.append("--curriculum")
    if BC_CURRICULUM_START_MAX is not None:
        train_cmd += ["--curriculum-start-max", str(int(BC_CURRICULUM_START_MAX))]
    if BC_CURRICULUM_END_MAX is not None:
        train_cmd += ["--curriculum-end-max", str(int(BC_CURRICULUM_END_MAX))]
    if BC_CURRICULUM_STEPS is not None:
        train_cmd += ["--curriculum-steps", str(int(BC_CURRICULUM_STEPS))]
    if GNN_ATTENTION:
        train_cmd.append("--gnn-attention")
    run_cmd(train_cmd)

# %% Pipeline opcional: meta-init
meta_init_path: Path | None = Path(META_INIT_MODEL_PATH) if META_INIT_MODEL_PATH else None
if RUN_META_TRAIN:
    meta_init_path = Path(META_INIT_MODEL_PATH) if META_INIT_MODEL_PATH else RUN_DIR / "meta_init.npz"
    run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_meta_init.py"),
            "--n-list",
            ",".join(str(n) for n in TRAIN_N_LIST),
            "--train-steps",
            str(META_TRAIN_STEPS),
            "--es-pop",
            str(META_ES_POP),
            "--sa-steps",
            str(META_SA_STEPS),
            "--out",
            str(meta_init_path),
        ]
    )

# %% Pipeline opcional: heatmap
heatmap_path: Path | None = Path(HEATMAP_MODEL_PATH) if HEATMAP_MODEL_PATH else None
if RUN_HEATMAP_TRAIN:
    heatmap_path = Path(HEATMAP_MODEL_PATH) if HEATMAP_MODEL_PATH else RUN_DIR / "heatmap_meta.npz"
    run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_heatmap_meta.py"),
            "--n-list",
            ",".join(str(n) for n in TRAIN_N_LIST),
            "--train-steps",
            str(HEATMAP_TRAIN_STEPS),
            "--es-pop",
            str(HEATMAP_ES_POP),
            "--heatmap-steps",
            str(HEATMAP_STEPS),
            "--policy",
            "gnn",
            "--knn-k",
            str(BC_KNN_K),
            "--out",
            str(heatmap_path),
        ]
    )

# %% Solvers e configuracao de avaliacao
if set(TRAIN_N_LIST) & set(VAL_N_LIST):
    raise ValueError("TRAIN_N_LIST and VAL_N_LIST must be disjoint to avoid leakage.")

# %% Modelos opcionais carregados de disco
bc_params = None
bc_config = None
if bc_policy_path is not None and bc_policy_path.exists():
    bc_params, bc_meta = load_params_npz(bc_policy_path)
    bc_config = l2o_config_from_meta(bc_meta, reward=REWARD, deterministic=True)

# %% Solvers
initial_cache: Dict[Tuple[int, int], np.ndarray] = {}


def get_initial(n: int, seed: int) -> np.ndarray:
    key = (n, seed)
    if key not in initial_cache:
        initial_cache[key] = make_initial(points, n, spacing, seed, INIT_MODE, RAND_SCALE)
    return initial_cache[key]


def solve_grid(n: int, seed: int) -> np.ndarray:
    return get_initial(n, seed)


def solve_sa(n: int, seed: int) -> np.ndarray:
    init_pose = get_initial(n, seed)
    init_batch = jnp.array(init_pose)[None, :, :]
    key = jax.random.PRNGKey(seed)
    best_poses, _ = run_sa_batch(
        key,
        SA_STEPS,
        n,
        init_batch,
        trans_sigma=SA_TRANS_SIGMA,
        rot_sigma=SA_ROT_SIGMA,
        rot_prob=SA_ROT_PROB,
        objective=SA_OBJECTIVE,
    )
    return np.array(best_poses[0])


def solve_l2o(params, cfg: L2OConfig) -> Callable[[int, int], np.ndarray]:
    def _solve(n: int, seed: int) -> np.ndarray:
        init_pose = get_initial(n, seed)
        key = jax.random.PRNGKey(seed)
        poses = optimize_with_l2o(key, params, jnp.array(init_pose), EVAL_STEPS, cfg)
        return np.array(poses)

    return _solve


def solve_l2o_refine(base_solver: Callable[[int, int], np.ndarray], params, cfg: L2OConfig) -> Callable[[int, int], np.ndarray]:
    def _solve(n: int, seed: int) -> np.ndarray:
        base_pose = base_solver(n, seed)
        key = jax.random.PRNGKey(seed)
        poses = optimize_with_l2o(key, params, jnp.array(base_pose), REFINE_STEPS, cfg)
        return np.array(poses)

    return _solve


def solve_meta_init_sa(n: int, seed: int) -> np.ndarray:
    if meta_init_path is None or not meta_init_path.exists():
        return solve_sa(n, seed)
    try:
        from meta_init import MetaInitConfig, apply_meta_init, load_meta_params  # noqa: E402
    except Exception:
        return solve_sa(n, seed)
    params, meta = load_meta_params(meta_init_path)
    config = MetaInitConfig(
        hidden_size=int(meta.get("hidden", 32)) if hasattr(meta.get("hidden", 32), "__int__") else 32,
        delta_xy=float(meta.get("delta_xy", 0.2)),
        delta_theta=float(meta.get("delta_theta", 10.0)),
    )
    init_pose = get_initial(n, seed)
    init_pose = np.array(apply_meta_init(params, jnp.array(init_pose), config))
    init_batch = jnp.array(init_pose)[None, :, :]
    key = jax.random.PRNGKey(seed)
    best_poses, _ = run_sa_batch(
        key,
        SA_STEPS,
        n,
        init_batch,
        trans_sigma=SA_TRANS_SIGMA,
        rot_sigma=SA_ROT_SIGMA,
        rot_prob=SA_ROT_PROB,
        objective=SA_OBJECTIVE,
    )
    return np.array(best_poses[0])


def solve_heatmap(n: int, seed: int) -> np.ndarray:
    if heatmap_path is None or not heatmap_path.exists():
        return solve_grid(n, seed)
    try:
        from heatmap_meta import HeatmapConfig, heatmap_search, load_params  # noqa: E402
    except Exception:
        return solve_grid(n, seed)
    params, meta = load_params(heatmap_path)
    config = HeatmapConfig(
        hidden_size=int(meta.get("hidden", 32)) if hasattr(meta.get("hidden", 32), "__int__") else 32,
        policy=str(meta.get("policy", "gnn")),
        knn_k=int(meta.get("knn_k", 4)) if hasattr(meta.get("knn_k", 4), "__int__") else 4,
        heatmap_lr=float(meta.get("heatmap_lr", 0.1)),
        trans_sigma=float(meta.get("trans_sigma", 0.2)),
        rot_sigma=float(meta.get("rot_sigma", 10.0)),
    )
    base = get_initial(n, seed)
    rng = np.random.default_rng(seed)
    poses, _ = heatmap_search(params, base, config, HEATMAP_STEPS, rng)
    return np.array(poses)


def solve_ensemble(candidates: Dict[str, Callable[[int, int], np.ndarray]]) -> Callable[[int, int], Tuple[np.ndarray, Dict[str, str]]]:
    def _solve(n: int, seed: int) -> Tuple[np.ndarray, Dict[str, str]]:
        best_score = float("inf")
        best_pose: np.ndarray | None = None
        best_name = "none"
        for name, fn in candidates.items():
            poses = fn(n, seed)
            if ENSEMBLE_SCORE == "packing":
                score = packing_score(points, poses)
            else:
                score = prefix_packing_score_np(points, poses)
            if score < best_score:
                best_score = float(score)
                best_pose = poses
                best_name = name
        if best_pose is None:
            best_pose = candidates[next(iter(candidates))](n, seed)
        return best_pose, {"selected": best_name}

    return _solve


# %% Rodar avaliacao (configs)
results: List[Dict[str, float]] = []
l2o_mlp_cfg = L2OConfig(
    hidden_size=HIDDEN_SIZE,
    policy="mlp",
    reward=REWARD,
    mlp_depth=MLP_DEPTH,
    gnn_steps=GNN_STEPS,
    gnn_attention=GNN_ATTENTION,
    action_scale=ACTION_SCALE,
    action_noise=False,
    feature_mode=FEATURE_MODE,
)
l2o_gnn_cfg = L2OConfig(
    hidden_size=HIDDEN_SIZE,
    policy="gnn",
    reward=REWARD,
    knn_k=4,
    mlp_depth=MLP_DEPTH,
    gnn_steps=GNN_STEPS,
    gnn_attention=GNN_ATTENTION,
    action_scale=ACTION_SCALE,
    action_noise=False,
    feature_mode=FEATURE_MODE,
)

# %% Avaliacao: baselines
results += evaluate_solver("grid", solve_grid, TRAIN_N_LIST, TRAIN_EVAL_SEEDS, points, split="train")
results += evaluate_solver("grid", solve_grid, VAL_N_LIST, VAL_EVAL_SEEDS, points, split="val")
results += evaluate_solver("sa", solve_sa, TRAIN_N_LIST, TRAIN_EVAL_SEEDS, points, split="train")
results += evaluate_solver("sa", solve_sa, VAL_N_LIST, VAL_EVAL_SEEDS, points, split="val")

# %% Avaliacao: L2O
results += evaluate_solver(
    "l2o_mlp",
    solve_l2o(mlp_params, l2o_mlp_cfg),
    TRAIN_N_LIST,
    TRAIN_EVAL_SEEDS,
    points,
    split="train",
)
results += evaluate_solver(
    "l2o_gnn",
    solve_l2o(gnn_params, l2o_gnn_cfg),
    TRAIN_N_LIST,
    TRAIN_EVAL_SEEDS,
    points,
    split="train",
)
results += evaluate_solver(
    "l2o_mlp",
    solve_l2o(mlp_params, l2o_mlp_cfg),
    VAL_N_LIST,
    VAL_EVAL_SEEDS,
    points,
    split="val",
)
results += evaluate_solver(
    "l2o_gnn",
    solve_l2o(gnn_params, l2o_gnn_cfg),
    VAL_N_LIST,
    VAL_EVAL_SEEDS,
    points,
    split="val",
)

# %% Avaliacao: modelos opcionais
if bc_params is not None and bc_config is not None:
    results += evaluate_solver(
        "l2o_bc",
        solve_l2o(bc_params, bc_config),
        TRAIN_N_LIST,
        TRAIN_EVAL_SEEDS,
        points,
        split="train",
    )
    results += evaluate_solver(
        "l2o_bc",
        solve_l2o(bc_params, bc_config),
        VAL_N_LIST,
        VAL_EVAL_SEEDS,
        points,
        split="val",
    )

if meta_init_path is not None and meta_init_path.exists():
    results += evaluate_solver(
        "sa_meta_init",
        solve_meta_init_sa,
        TRAIN_N_LIST,
        TRAIN_EVAL_SEEDS,
        points,
        split="train",
    )
    results += evaluate_solver(
        "sa_meta_init",
        solve_meta_init_sa,
        VAL_N_LIST,
        VAL_EVAL_SEEDS,
        points,
        split="val",
    )

if heatmap_path is not None and heatmap_path.exists():
    results += evaluate_solver(
        "heatmap",
        solve_heatmap,
        TRAIN_N_LIST,
        TRAIN_EVAL_SEEDS,
        points,
        split="train",
    )
    results += evaluate_solver(
        "heatmap",
        solve_heatmap,
        VAL_N_LIST,
        VAL_EVAL_SEEDS,
        points,
        split="val",
    )

if L2O_REFINE_GRID:
    results += evaluate_solver(
        "l2o_refine_grid",
        solve_l2o_refine(solve_grid, mlp_params, l2o_mlp_cfg),
        TRAIN_N_LIST,
        TRAIN_EVAL_SEEDS,
        points,
        split="train",
    )
    results += evaluate_solver(
        "l2o_refine_grid",
        solve_l2o_refine(solve_grid, mlp_params, l2o_mlp_cfg),
        VAL_N_LIST,
        VAL_EVAL_SEEDS,
        points,
        split="val",
    )

if L2O_REFINE_SA:
    results += evaluate_solver(
        "l2o_refine_sa",
        solve_l2o_refine(solve_sa, gnn_params, l2o_gnn_cfg),
        TRAIN_N_LIST,
        TRAIN_EVAL_SEEDS,
        points,
        split="train",
    )
    results += evaluate_solver(
        "l2o_refine_sa",
        solve_l2o_refine(solve_sa, gnn_params, l2o_gnn_cfg),
        VAL_N_LIST,
        VAL_EVAL_SEEDS,
        points,
        split="val",
    )

# %% Avaliacao: ensemble
if RUN_ENSEMBLE:
    ensemble_candidates = {
        "grid": solve_grid,
        "sa": solve_sa,
        "l2o_mlp": solve_l2o(mlp_params, l2o_mlp_cfg),
        "l2o_gnn": solve_l2o(gnn_params, l2o_gnn_cfg),
    }
    if bc_params is not None and bc_config is not None:
        ensemble_candidates["l2o_bc"] = solve_l2o(bc_params, bc_config)
    if meta_init_path is not None and meta_init_path.exists():
        ensemble_candidates["sa_meta_init"] = solve_meta_init_sa
    if heatmap_path is not None and heatmap_path.exists():
        ensemble_candidates["heatmap"] = solve_heatmap
    results += evaluate_solver(
        "ensemble",
        solve_ensemble(ensemble_candidates),
        TRAIN_N_LIST,
        TRAIN_EVAL_SEEDS,
        points,
        split="train",
    )
    results += evaluate_solver(
        "ensemble",
        solve_ensemble(ensemble_candidates),
        VAL_N_LIST,
        VAL_EVAL_SEEDS,
        points,
        split="val",
    )

# %% Resumos e artifacts
per_n = summarize_results(results)
overall = summarize_overall(results)

meta = {
    "reward": REWARD,
    "train_n_list": TRAIN_N_LIST,
    "val_n_list": VAL_N_LIST,
    "train_eval_seeds": TRAIN_EVAL_SEEDS,
    "val_eval_seeds": VAL_EVAL_SEEDS,
    "train_steps": TRAIN_STEPS,
    "rollout_steps": ROLLOUT_STEPS,
    "batch": BATCH,
    "eval_steps": EVAL_STEPS,
    "hidden_size": HIDDEN_SIZE,
    "mlp_depth": MLP_DEPTH,
    "gnn_steps": GNN_STEPS,
    "gnn_attention": GNN_ATTENTION,
    "action_scale": ACTION_SCALE,
    "feature_mode": FEATURE_MODE,
    "baseline_mode": BASELINE_MODE,
    "baseline_decay": BASELINE_DECAY,
    "init_mode": INIT_MODE,
    "rand_scale": RAND_SCALE,
    "sa_steps": SA_STEPS,
    "sa_trans_sigma": SA_TRANS_SIGMA,
    "sa_rot_sigma": SA_ROT_SIGMA,
    "sa_rot_prob": SA_ROT_PROB,
    "sa_objective": SA_OBJECTIVE,
    "run_bc_pipeline": RUN_BC_PIPELINE,
    "bc_policy": BC_POLICY,
    "bc_runs_per_n": BC_RUNS_PER_N,
    "bc_steps": BC_STEPS,
    "bc_train_steps": BC_TRAIN_STEPS,
    "bc_policy_path": str(bc_policy_path) if bc_policy_path else None,
    "run_meta_train": RUN_META_TRAIN,
    "meta_init_path": str(meta_init_path) if meta_init_path else None,
    "run_heatmap_train": RUN_HEATMAP_TRAIN,
    "heatmap_path": str(heatmap_path) if heatmap_path else None,
    "run_ensemble": RUN_ENSEMBLE,
    "ensemble_score": ENSEMBLE_SCORE,
    "l2o_refine_grid": L2O_REFINE_GRID,
    "l2o_refine_sa": L2O_REFINE_SA,
    "refine_steps": REFINE_STEPS,
}

save_eval_artifacts(RUN_DIR, results, per_n, overall, meta)
plot_eval_curves(per_n, RUN_DIR / "eval_curve.png")

# %% Score GNN (proposta do desafio)
gnn_train_score = challenge_score_from_results(results, "l2o_gnn", "train")
gnn_val_score = challenge_score_from_results(results, "l2o_gnn", "val")
(RUN_DIR / "gnn_score.txt").write_text(
    f"gnn_train_score={gnn_train_score:.6f}\n"
    f"gnn_val_score={gnn_val_score:.6f}\n"
)

print("GNN score (challenge-style):")
print(f"  train={gnn_train_score:.6f}")
print(f"  val={gnn_val_score:.6f}")
print("Eval artifacts saved to", RUN_DIR)

# %% Gerar submission.csv (Kaggle)
SUBMISSION_NMAX = 200
SUBMISSION_SEED = 1
SUBMISSION_OVERLAP_CHECK = True  # para score final, mantenha True

RUN_SUBMISSION_SWEEP = True  # True = gera/score varias receitas + seeds
SWEEP_NMAX = 50  # use 200 para score final
SWEEP_SEEDS = [1, 2]
SWEEP_SCORE_OVERLAP_CHECK = False  # durante sweep rapido, pode ser False; no final use True
SWEEP_BUILD_ENSEMBLE = True

# Salvar as politicas treinadas neste notebook (para usar no generate_submission/guided SA)
MLP_POLICY_PATH = RUN_DIR / "l2o_mlp.npz"
GNN_POLICY_PATH = RUN_DIR / "l2o_gnn.npz"
save_params_npz(
    MLP_POLICY_PATH,
    mlp_params,
    meta={
        "policy": "mlp",
        "hidden": HIDDEN_SIZE,
        "mlp_depth": MLP_DEPTH,
        "feature_mode": FEATURE_MODE,
        "reward": REWARD,
        "action_scale": ACTION_SCALE,
    },
)
save_params_npz(
    GNN_POLICY_PATH,
    gnn_params,
    meta={
        "policy": "gnn",
        "hidden": HIDDEN_SIZE,
        "knn_k": 4,
        "mlp_depth": MLP_DEPTH,
        "gnn_steps": GNN_STEPS,
        "gnn_attention": GNN_ATTENTION,
        "feature_mode": FEATURE_MODE,
        "reward": REWARD,
        "action_scale": ACTION_SCALE,
    },
)


def run_cmd_capture(cmd: List[str]) -> str:
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}:\n{result.stdout}")
    return result.stdout


def score_csv(csv_path: Path, *, nmax: int, check_overlap: bool) -> Dict[str, object]:
    cmd = [sys.executable, str(ROOT / "scripts" / "score_submission.py"), str(csv_path), "--nmax", str(nmax)]
    if not check_overlap:
        cmd.append("--no-overlap")
    out = run_cmd_capture(cmd).strip()
    return json.loads(out) if out else {}


def generate_submission(
    out_csv: Path,
    *,
    seed: int,
    nmax: int,
    args: Dict[str, object],
) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "generate_submission.py"),
        "--out",
        str(out_csv),
        "--seed",
        str(seed),
        "--nmax",
        str(nmax),
    ]
    for key, value in args.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        cmd += [flag, str(value)]
    run_cmd(cmd)


def _best_per_puzzle_ensemble(out_csv: Path, candidates: Dict[str, Path], *, nmax: int) -> Dict[str, object]:
    try:
        from scoring import load_submission  # noqa: E402
    except Exception as exc:
        raise RuntimeError("Failed to import scoring.load_submission") from exc

    points = np.array(TREE_POINTS, dtype=float)
    loaded = {name: load_submission(path, nmax=nmax) for name, path in candidates.items()}

    selected: Dict[int, str] = {}
    best_poses: Dict[int, np.ndarray] = {}
    for n in range(1, nmax + 1):
        best_s = float("inf")
        best_name = None
        best_pose = None
        for name, puzzles in loaded.items():
            poses = puzzles.get(n)
            if poses is None or poses.shape[0] != n:
                continue
            s = float(packing_score(points, poses))
            if s < best_s:
                best_s = s
                best_name = name
                best_pose = poses
        if best_name is None or best_pose is None:
            raise ValueError(f"No complete candidates for puzzle {n}")
        selected[n] = best_name
        best_poses[n] = np.array(best_pose, dtype=float)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "deg"])
        for n in range(1, nmax + 1):
            poses = best_poses[n]
            poses[:, 2] = np.mod(poses[:, 2], 360.0)
            for i, (x, y, deg) in enumerate(poses):
                writer.writerow([f"{n:03d}_{i}", f"s{float(x):.17f}", f"s{float(y):.17f}", f"s{float(deg):.17f}"])

    return {"selected_by_puzzle": {str(k): v for k, v in selected.items()}}


# === Modelos disponiveis (paths) ===
META_INIT_MODEL = str(meta_init_path) if (meta_init_path is not None and meta_init_path.exists()) else None
HEATMAP_MODEL = str(heatmap_path) if (heatmap_path is not None and heatmap_path.exists()) else None

# L2O models trained in this run (plus optional BC policy).
CANDIDATE_L2O_MODELS: Dict[str, Path] = {
    "reinforce_gnn": GNN_POLICY_PATH,
    "reinforce_mlp": MLP_POLICY_PATH,
}
if bc_policy_path is not None and bc_policy_path.exists():
    CANDIDATE_L2O_MODELS["bc"] = bc_policy_path
CANDIDATE_GUIDED_MODELS: Dict[str, Path] = dict(CANDIDATE_L2O_MODELS)


def _stable_hash_dict(data: Dict[str, object]) -> str:
    blob = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:10]


def _build_recipe_pool() -> tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]], Dict[str, object]]:
    """Gera uma pool de receitas cobrindo todas as features do generate_submission.py.

    Mantem as receitas como um dict de flags (somente chaves suportadas pelo script).
    Metadados (familia/modelo) ficam em um dict separado para auditoria.
    """

    # Base: desliga tudo e deixa lattice como fallback garantido.
    base: Dict[str, object] = {
        "sa_nmax": 0,
        "sa_batch": 64,
        "sa_steps": 400,
        "sa_trans_sigma": 0.2,
        "sa_rot_sigma": 15.0,
        "sa_rot_prob": 0.3,
        "sa_objective": "packing",
        "meta_init_model": None,
        "heatmap_model": None,
        "heatmap_nmax": 0,
        "heatmap_steps": 200,
        "l2o_model": None,
        "l2o_init": "grid",
        "l2o_nmax": 0,
        "l2o_steps": 200,
        "l2o_trans_sigma": 0.2,
        "l2o_rot_sigma": 10.0,
        "l2o_deterministic": True,
        "lattice_pattern": "hex",
        "lattice_margin": 0.02,
        "lattice_rotate": 0.0,
        "refine_nmin": 0,
        "refine_batch": 16,
        "refine_steps": 0,
        "refine_trans_sigma": 0.2,
        "refine_rot_sigma": 15.0,
        "refine_rot_prob": 0.3,
        "refine_objective": "packing",
        "guided_model": None,
        "guided_prob": 1.0,
        "guided_pmax": 0.05,
    }

    # ===== Experiment grids (ajuste aqui) =====
    # Lattice sweep (base/fallback de tudo).
    lattice_variants_all: List[Dict[str, object]] = []
    for pattern, margin, rot in itertools.product(
        ["hex", "square"],
        [0.0, 0.01, 0.02],
        [0.0, 15.0, 30.0],
    ):
        lattice_variants_all.append({"lattice_pattern": pattern, "lattice_margin": float(margin), "lattice_rotate": float(rot)})

    # SA (n pequeno): custo principal do solver -> vale experimentar.
    sa_presets: Dict[str, Dict[str, object]] = {
        "sa30": {"sa_nmax": 30, "sa_batch": 64, "sa_steps": 400, "sa_trans_sigma": 0.2, "sa_rot_sigma": 15.0, "sa_rot_prob": 0.3},
        "sa50": {"sa_nmax": 50, "sa_batch": 64, "sa_steps": 500, "sa_trans_sigma": 0.2, "sa_rot_sigma": 15.0, "sa_rot_prob": 0.3},
        "sa80": {"sa_nmax": 80, "sa_batch": 96, "sa_steps": 600, "sa_trans_sigma": 0.2, "sa_rot_sigma": 18.0, "sa_rot_prob": 0.35},
    }

    # Refine (n alto): warm-start lattice/l2o e refina via SA.
    refine_presets: Dict[str, Dict[str, object]] = {
        "ref80_200": {
            "refine_nmin": 80,
            "refine_batch": 16,
            "refine_steps": 200,
            "refine_trans_sigma": 0.2,
            "refine_rot_sigma": 15.0,
            "refine_rot_prob": 0.3,
        },
        "ref80_400": {
            "refine_nmin": 80,
            "refine_batch": 24,
            "refine_steps": 400,
            "refine_trans_sigma": 0.2,
            "refine_rot_sigma": 15.0,
            "refine_rot_prob": 0.3,
        },
        "ref120_300": {
            "refine_nmin": 120,
            "refine_batch": 24,
            "refine_steps": 300,
            "refine_trans_sigma": 0.2,
            "refine_rot_sigma": 15.0,
            "refine_rot_prob": 0.3,
        },
    }

    # Guided SA knobs (policy como proposal quando confiante).
    guided_presets: Dict[str, Dict[str, object]] = {
        "guided_p005": {"guided_prob": 1.0, "guided_pmax": 0.05},
        "guided_p02": {"guided_prob": 1.0, "guided_pmax": 0.02},
        "guided_mix": {"guided_prob": 0.5, "guided_pmax": 0.05},
    }

    # L2O knobs (n pequeno): policy pode substituir SA inicial.
    l2o_presets: Dict[str, Dict[str, object]] = {
        "l2o10": {"l2o_init": "lattice", "l2o_nmax": 10, "l2o_steps": 200, "l2o_trans_sigma": 0.2, "l2o_rot_sigma": 10.0, "l2o_deterministic": True},
        "l2o20": {"l2o_init": "lattice", "l2o_nmax": 20, "l2o_steps": 250, "l2o_trans_sigma": 0.2, "l2o_rot_sigma": 10.0, "l2o_deterministic": True},
    }

    # Heatmap knobs (n muito pequeno): meta-optimizer alternativo ao SA/L2O.
    heatmap_presets: Dict[str, Dict[str, object]] = {
        "heat10": {"heatmap_nmax": 10, "heatmap_steps": 200},
        "heat20": {"heatmap_nmax": 20, "heatmap_steps": 250},
    }

    # Limites para nao explodir combinacoes (aumente/disable para explorar mais).
    MAX_RECIPES_PER_FAMILY = 20
    MAX_LATTICE_VARIANTS = 10

    def limit_variants(variants: List[Dict[str, object]], max_n: int | None) -> List[Dict[str, object]]:
        if max_n is None or len(variants) <= max_n:
            return variants
        scored = [(json.dumps(v, sort_keys=True), v) for v in variants]
        scored.sort(key=lambda x: hashlib.sha1(x[0].encode("utf-8")).hexdigest())
        return [v for _s, v in scored[:max_n]]

    lattice_variants = limit_variants(lattice_variants_all, MAX_LATTICE_VARIANTS)

    recipes: Dict[str, Dict[str, object]] = {}
    meta: Dict[str, Dict[str, object]] = {}

    def add_recipe(family: str, recipe: Dict[str, object], *, meta_extra: Dict[str, object] | None = None) -> None:
        full = dict(base)
        full.update(recipe)
        rid = _stable_hash_dict(full)
        name = f"{family}_{rid}"
        if name in recipes:
            return
        recipes[name] = full
        meta[name] = {"family": family}
        if meta_extra:
            meta[name].update(meta_extra)

    # ---- Baselines ----
    for lat in lattice_variants:
        add_recipe("lattice", lat, meta_extra={"lattice": lat})

    # ---- SA / SA+refine ----
    for lat in lattice_variants:
        for sa_name, sa_cfg in sa_presets.items():
            add_recipe("sa", {**lat, **sa_cfg}, meta_extra={"lattice": lat, "sa": sa_name})
            for ref_name, ref_cfg in refine_presets.items():
                add_recipe("sa_refine", {**lat, **sa_cfg, **ref_cfg}, meta_extra={"lattice": lat, "sa": sa_name, "refine": ref_name})
                if META_INIT_MODEL is not None:
                    add_recipe(
                        "sa_refine_meta",
                        {**lat, **sa_cfg, **ref_cfg, "meta_init_model": META_INIT_MODEL},
                        meta_extra={"lattice": lat, "sa": sa_name, "refine": ref_name, "meta_init": True},
                    )

    # ---- Guided SA (em cima do melhor preset base) ----
    guided_lattice = lattice_variants[:2] if lattice_variants else [{"lattice_pattern": "hex", "lattice_margin": 0.02, "lattice_rotate": 0.0}]
    guided_sa = sa_presets.get("sa50", next(iter(sa_presets.values())))
    guided_ref = refine_presets.get("ref80_200", next(iter(refine_presets.values())))
    for model_name, model_path in CANDIDATE_GUIDED_MODELS.items():
        for lat in guided_lattice:
            for g_name, g_cfg in guided_presets.items():
                add_recipe(
                    "guided_refine",
                    {**lat, **guided_sa, **guided_ref, **g_cfg, "guided_model": str(model_path)},
                    meta_extra={"guided_model": model_name, "guided": g_name, "lattice": lat},
                )
                if META_INIT_MODEL is not None:
                    add_recipe(
                        "guided_refine_meta",
                        {**lat, **guided_sa, **guided_ref, **g_cfg, "guided_model": str(model_path), "meta_init_model": META_INIT_MODEL},
                        meta_extra={"guided_model": model_name, "guided": g_name, "lattice": lat, "meta_init": True},
                    )

    # ---- L2O (n pequeno) + SA/refine ----
    l2o_lattice = lattice_variants[:2] if lattice_variants else [{"lattice_pattern": "hex", "lattice_margin": 0.02, "lattice_rotate": 0.0}]
    l2o_sa = sa_presets.get("sa50", next(iter(sa_presets.values())))
    l2o_ref = refine_presets.get("ref80_200", next(iter(refine_presets.values())))
    for model_name, model_path in CANDIDATE_L2O_MODELS.items():
        for lat in l2o_lattice:
            for l2o_name, l2o_cfg in l2o_presets.items():
                add_recipe(
                    "l2o_refine",
                    {**lat, **l2o_sa, **l2o_ref, **l2o_cfg, "l2o_model": str(model_path)},
                    meta_extra={"l2o_model": model_name, "l2o": l2o_name, "lattice": lat},
                )
                if META_INIT_MODEL is not None:
                    add_recipe(
                        "l2o_refine_meta",
                        {**lat, **l2o_sa, **l2o_ref, **l2o_cfg, "l2o_model": str(model_path), "meta_init_model": META_INIT_MODEL},
                        meta_extra={"l2o_model": model_name, "l2o": l2o_name, "lattice": lat, "meta_init": True},
                    )

    # ---- Heatmap variants ----
    if HEATMAP_MODEL is not None:
        heat_lattice = lattice_variants[:2] if lattice_variants else [{"lattice_pattern": "hex", "lattice_margin": 0.02, "lattice_rotate": 0.0}]
        for lat in heat_lattice:
            for h_name, h_cfg in heatmap_presets.items():
                add_recipe(
                    "heatmap",
                    {**lat, **h_cfg, "heatmap_model": HEATMAP_MODEL},
                    meta_extra={"heatmap": h_name, "lattice": lat},
                )
                # heatmap + SA/refine
                add_recipe(
                    "heatmap_sa_refine",
                    {**lat, **h_cfg, "heatmap_model": HEATMAP_MODEL, **guided_sa, **guided_ref},
                    meta_extra={"heatmap": h_name, "lattice": lat, "sa": "sa50", "refine": "ref80_200"},
                )
                # heatmap + l2o (com o melhor modelo disponivel) + refine
                best_l2o_name = next(iter(CANDIDATE_L2O_MODELS))
                best_l2o_path = CANDIDATE_L2O_MODELS[best_l2o_name]
                best_l2o_cfg = l2o_presets.get("l2o10", next(iter(l2o_presets.values())))
                add_recipe(
                    "heatmap_l2o_refine",
                    {**lat, **h_cfg, "heatmap_model": HEATMAP_MODEL, **best_l2o_cfg, "l2o_model": str(best_l2o_path), **guided_sa, **guided_ref},
                    meta_extra={"heatmap": h_name, "l2o_model": best_l2o_name, "l2o": "l2o10", "lattice": lat},
                )

    # ---- Cap recipes per family ----
    if MAX_RECIPES_PER_FAMILY is not None:
        by_family: Dict[str, List[str]] = {}
        for name in recipes:
            fam = str(meta.get(name, {}).get("family", "misc"))
            by_family.setdefault(fam, []).append(name)
        keep: List[str] = []
        for fam, names in by_family.items():
            keep.extend(sorted(names)[:MAX_RECIPES_PER_FAMILY])
        recipes = {k: recipes[k] for k in keep}
        meta = {k: meta[k] for k in keep}

    settings: Dict[str, object] = {
        "max_recipes_per_family": MAX_RECIPES_PER_FAMILY,
        "max_lattice_variants": MAX_LATTICE_VARIANTS,
        "lattice_variants_total": len(lattice_variants_all),
        "lattice_variants_used": len(lattice_variants),
        "sa_presets": sorted(sa_presets.keys()),
        "refine_presets": sorted(refine_presets.keys()),
        "guided_presets": sorted(guided_presets.keys()),
        "l2o_presets": sorted(l2o_presets.keys()),
        "heatmap_presets": sorted(heatmap_presets.keys()),
    }

    return recipes, meta, settings


# === Receitas (flags do scripts/generate_submission.py) ===
RECIPES, RECIPES_META, RECIPES_SETTINGS = _build_recipe_pool()
RECIPES_SETTINGS.update(
    {
        "meta_init_model": META_INIT_MODEL,
        "heatmap_model": HEATMAP_MODEL,
        "candidate_l2o_models": {k: str(v) for k, v in CANDIDATE_L2O_MODELS.items()},
        "candidate_guided_models": {k: str(v) for k, v in CANDIDATE_GUIDED_MODELS.items()},
    }
)
(RUN_DIR / "recipes.json").write_text(json.dumps(RECIPES, indent=2, sort_keys=True, default=str))
(RUN_DIR / "recipes_meta.json").write_text(json.dumps(RECIPES_META, indent=2, sort_keys=True, default=str))
(RUN_DIR / "recipes_settings.json").write_text(json.dumps(RECIPES_SETTINGS, indent=2, sort_keys=True, default=str))

# Remove receitas que dependem de modelos inexistentes
ACTIVE_RECIPES: Dict[str, Dict[str, object]] = {}
ACTIVE_META: Dict[str, Dict[str, object]] = {}
for name, recipe in RECIPES.items():
    needs = [
        ("l2o_model", recipe.get("l2o_model")),
        ("guided_model", recipe.get("guided_model")),
        ("meta_init_model", recipe.get("meta_init_model")),
        ("heatmap_model", recipe.get("heatmap_model")),
    ]
    missing = [k for k, v in needs if v is not None and not Path(str(v)).exists()]
    if missing:
        print(f"[skip] receita '{name}' (modelos ausentes: {missing})")
        continue
    ACTIVE_RECIPES[name] = recipe
    if name in RECIPES_META:
        ACTIVE_META[name] = RECIPES_META[name]
(RUN_DIR / "active_recipes.json").write_text(json.dumps(ACTIVE_RECIPES, indent=2, sort_keys=True, default=str))
(RUN_DIR / "active_recipes_meta.json").write_text(json.dumps(ACTIVE_META, indent=2, sort_keys=True, default=str))

SUB_DIR = RUN_DIR / "submissions"
SUB_DIR.mkdir(parents=True, exist_ok=True)

if RUN_SUBMISSION_SWEEP:
    # Sweep em 2 estagios:
    # 1) ranking rapido em n pequeno (p/ cortar combinacoes)
    # 2) rerun somente top-K em nmax=200 (com overlap_check) + opcional ensemble por puzzle
    SWEEP_TOPK = 20  # quantos candidatos do estagio 1 vao para o estagio 2
    TWO_STAGE_SWEEP = True
    (RUN_DIR / "submission_sweep_meta.json").write_text(
        json.dumps(
            {
                "two_stage": TWO_STAGE_SWEEP,
                "stage1": {
                    "nmax": int(SWEEP_NMAX),
                    "seeds": [int(s) for s in SWEEP_SEEDS],
                    "overlap_check": bool(SWEEP_SCORE_OVERLAP_CHECK),
                },
                "stage2": {
                    "nmax": int(SUBMISSION_NMAX),
                    "overlap_check": bool(SUBMISSION_OVERLAP_CHECK),
                    "topk": int(SWEEP_TOPK),
                },
            },
            indent=2,
        )
    )

    stage1_rows: List[Dict[str, object]] = []
    stage1_paths: Dict[str, Path] = {}
    for recipe_name, recipe in ACTIVE_RECIPES.items():
        for seed in SWEEP_SEEDS:
            tag = f"{recipe_name}_seed{seed}"
            out_csv = SUB_DIR / f"stage1_{tag}.csv"
            generate_submission(out_csv, seed=seed, nmax=SWEEP_NMAX, args=recipe)
            score = score_csv(out_csv, nmax=SWEEP_NMAX, check_overlap=SWEEP_SCORE_OVERLAP_CHECK)
            stage1_rows.append(
                {
                    "tag": tag,
                    "stage": 1,
                    "recipe": recipe_name,
                    "seed": int(seed),
                    "nmax": int(SWEEP_NMAX),
                    "score": score.get("score"),
                    "s_max": score.get("s_max"),
                    "overlap_check": score.get("overlap_check"),
                }
            )
            stage1_paths[tag] = out_csv

    stage1_rows = sorted(stage1_rows, key=lambda r: (float(r.get("score") or float("inf")), str(r["tag"])))
    write_csv(RUN_DIR / "submission_sweep_stage1.csv", stage1_rows)

    if not TWO_STAGE_SWEEP:
        # Comportamento antigo (1 estagio): promove o melhor do sweep rapido.
        best = stage1_rows[0] if stage1_rows else None
        if best is not None:
            best_path = stage1_paths[str(best["tag"])]
            shutil.copyfile(best_path, RUN_DIR / "submission_best.csv")
            (RUN_DIR / "submission_best.txt").write_text(json.dumps(best, indent=2))
            print("Best (sweep stage1):", best)
            print("Saved:", RUN_DIR / "submission_best.csv")

        if SWEEP_BUILD_ENSEMBLE and stage1_paths:
            ens_csv = RUN_DIR / "submission_ensemble.csv"
            ens_meta = _best_per_puzzle_ensemble(ens_csv, stage1_paths, nmax=SWEEP_NMAX)
            (RUN_DIR / "submission_ensemble_meta.json").write_text(json.dumps(ens_meta, indent=2))
            ens_score = score_csv(ens_csv, nmax=SWEEP_NMAX, check_overlap=SWEEP_SCORE_OVERLAP_CHECK)
            (RUN_DIR / "submission_ensemble_score.json").write_text(json.dumps(ens_score, indent=2))
            print("Ensemble score (stage1):", ens_score.get("score"))
            print("Saved:", ens_csv)
    else:
        selected = stage1_rows[: int(SWEEP_TOPK)] if stage1_rows else []
        (RUN_DIR / "submission_sweep_selected.json").write_text(json.dumps(selected, indent=2, default=str))
        print(f"Stage1: {len(stage1_rows)} candidates; promoting top {len(selected)} to stage2")

        stage2_rows: List[Dict[str, object]] = []
        stage2_paths: Dict[str, Path] = {}
        for row in selected:
            recipe_name = str(row["recipe"])
            seed = int(row["seed"])
            tag = str(row["tag"])
            recipe = ACTIVE_RECIPES[recipe_name]

            out_csv = SUB_DIR / f"stage2_{tag}.csv"
            generate_submission(out_csv, seed=seed, nmax=SUBMISSION_NMAX, args=recipe)
            score = score_csv(out_csv, nmax=SUBMISSION_NMAX, check_overlap=SUBMISSION_OVERLAP_CHECK)
            stage2_rows.append(
                {
                    "tag": tag,
                    "stage": 2,
                    "recipe": recipe_name,
                    "seed": seed,
                    "nmax": int(SUBMISSION_NMAX),
                    "score": score.get("score"),
                    "s_max": score.get("s_max"),
                    "overlap_check": score.get("overlap_check"),
                    "stage1_score": row.get("score"),
                }
            )
            stage2_paths[tag] = out_csv

        stage2_rows = sorted(stage2_rows, key=lambda r: (float(r.get("score") or float("inf")), str(r["tag"])))
        write_csv(RUN_DIR / "submission_sweep_stage2.csv", stage2_rows)

        best2 = stage2_rows[0] if stage2_rows else None
        best_score = float(best2.get("score")) if best2 is not None and best2.get("score") is not None else float("inf")
        best_csv: Path | None = None
        best_meta: Dict[str, object] | None = None
        if best2 is not None:
            best_csv = stage2_paths[str(best2["tag"])]
            best_meta = dict(best2)
            shutil.copyfile(best_csv, RUN_DIR / "submission_best.csv")
            (RUN_DIR / "submission_best.txt").write_text(json.dumps(best2, indent=2))
            print("Best (stage2):", best2)
            print("Saved:", RUN_DIR / "submission_best.csv")

        if SWEEP_BUILD_ENSEMBLE and stage2_paths:
            ens_csv = RUN_DIR / "submission_ensemble.csv"
            ens_meta = _best_per_puzzle_ensemble(ens_csv, stage2_paths, nmax=SUBMISSION_NMAX)
            (RUN_DIR / "submission_ensemble_meta.json").write_text(json.dumps(ens_meta, indent=2))
            ens_score = score_csv(ens_csv, nmax=SUBMISSION_NMAX, check_overlap=SUBMISSION_OVERLAP_CHECK)
            (RUN_DIR / "submission_ensemble_score.json").write_text(json.dumps(ens_score, indent=2))
            print("Ensemble score (stage2):", ens_score.get("score"))
            print("Saved:", ens_csv)
            try:
                ens_total = float(ens_score.get("score")) if ens_score.get("score") is not None else float("inf")
            except Exception:
                ens_total = float("inf")
            if ens_total < best_score:
                shutil.copyfile(ens_csv, RUN_DIR / "submission_best.csv")
                (RUN_DIR / "submission_best.txt").write_text(
                    json.dumps(
                        {
                            "tag": "ensemble",
                            "stage": 2,
                            "score": ens_score.get("score"),
                            "s_max": ens_score.get("s_max"),
                            "overlap_check": ens_score.get("overlap_check"),
                            "selected_by_puzzle": ens_meta.get("selected_by_puzzle"),
                            "candidates": list(stage2_paths.keys()),
                        },
                        indent=2,
                    )
                )
                print("Best updated to ensemble.")
else:
    # Rodada unica (use nmax=200 + overlap_check=True para score final)
    def _pick_single_recipe() -> str:
        preferred_families = [
            "guided_refine_meta",
            "guided_refine",
            "l2o_refine_meta",
            "l2o_refine",
            "sa_refine_meta",
            "sa_refine",
            "sa",
            "lattice",
        ]
        for fam in preferred_families:
            cands = [k for k, m in ACTIVE_META.items() if str(m.get("family")) == fam]
            if cands:
                return sorted(cands)[0]
        return sorted(ACTIVE_RECIPES)[0]

    SINGLE_RECIPE = _pick_single_recipe()
    out_csv = RUN_DIR / "submission.csv"
    generate_submission(out_csv, seed=SUBMISSION_SEED, nmax=SUBMISSION_NMAX, args=ACTIVE_RECIPES[SINGLE_RECIPE])
    score = score_csv(out_csv, nmax=SUBMISSION_NMAX, check_overlap=SUBMISSION_OVERLAP_CHECK)
    (RUN_DIR / "submission_score.json").write_text(json.dumps(score, indent=2))
    print("Submission saved to", out_csv)
    print("Score:", score.get("score"))
