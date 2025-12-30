#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

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

from geom_np import polygon_radius, shift_poses_to_origin, transform_polygon  # noqa: E402
from l2o import L2OConfig, optimize_with_l2o  # noqa: E402
from scripts.train_l2o import train_model  # noqa: E402
from tree_data import TREE_POINTS  # noqa: E402


def grid_initial(n: int, spacing: float) -> np.ndarray:
    cols = int(np.ceil(np.sqrt(n)))
    poses = np.zeros((n, 3), dtype=float)
    for i in range(n):
        row = i // cols
        col = i % cols
        poses[i] = (col * spacing, row * spacing, 0.0)
    return poses


def plot_packing(poses: np.ndarray, title: str) -> None:
    points = np.array(TREE_POINTS, dtype=float)
    plt.figure(figsize=(6, 6))
    for pose in poses:
        poly = transform_polygon(points, pose)
        p = np.vstack([poly, poly[0]])
        plt.plot(p[:, 0], p[:, 1], "g-")
    plt.axis("equal")
    plt.title(title)
    plt.show()


# === Configuracoes rapidas ===
N = 10
TRAIN_STEPS = 50
ROLLOUT_STEPS = 200
BATCH = 64
REWARD = "prefix"  # "packing" ou "prefix"

points = np.array(TREE_POINTS, dtype=float)
spacing = 2.0 * polygon_radius(points) * 1.2
init = shift_poses_to_origin(points, grid_initial(N, spacing))

# === Treino MLP ===
mlp_params, mlp_loss = train_model(
    seed=1,
    n_list=[N],
    batch=BATCH,
    train_steps=TRAIN_STEPS,
    steps=ROLLOUT_STEPS,
    policy="mlp",
    reward=REWARD,
    verbose_freq=10,
)

# === Treino GNN ===
gnn_params, gnn_loss = train_model(
    seed=2,
    n_list=[N],
    batch=BATCH,
    train_steps=TRAIN_STEPS,
    steps=ROLLOUT_STEPS,
    policy="gnn",
    reward=REWARD,
    knn_k=4,
    verbose_freq=10,
)

# === Plot losses ===
plt.figure(figsize=(6, 4))
plt.plot(mlp_loss, label="MLP")
plt.plot(gnn_loss, label="GNN")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("L2O Loss")
plt.legend()
plt.show()

# === Visualizacao de packing ===
key = jax.random.PRNGKey(0)
config = L2OConfig(policy="mlp", reward=REWARD, action_noise=False)
mlp_poses = optimize_with_l2o(key, mlp_params, jnp.array(init), ROLLOUT_STEPS, config)
plot_packing(np.array(mlp_poses), "MLP packing")

key = jax.random.PRNGKey(1)
config = L2OConfig(policy="gnn", reward=REWARD, knn_k=4, action_noise=False)
gnn_poses = optimize_with_l2o(key, gnn_params, jnp.array(init), ROLLOUT_STEPS, config)
plot_packing(np.array(gnn_poses), "GNN packing")
