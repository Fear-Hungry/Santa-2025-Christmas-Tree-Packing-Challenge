from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from collisions import check_any_collisions
from packing import packing_score, prefix_packing_score
from tree import get_tree_polygon


@dataclass(frozen=True)
class L2OConfig:
    hidden_size: int = 32
    policy: str = "mlp"  # "mlp" or "gnn"
    knn_k: int = 4
    mlp_depth: int = 1
    gnn_steps: int = 1
    gnn_attention: bool = False
    feature_mode: str = "raw"  # "raw" or "bbox_norm"
    reward: str = "packing"  # "packing" or "prefix"
    trans_sigma: float = 0.2
    rot_sigma: float = 10.0
    action_scale: float = 1.0
    overlap_penalty: float = 50.0
    action_noise: bool = True


Params = Dict[str, jnp.ndarray]


def init_params(
    key: jax.Array,
    hidden_size: int = 32,
    policy: str = "mlp",
    mlp_depth: int = 1,
    gnn_attention: bool = False,
    feature_mode: str = "raw",
) -> Params:
    input_dim = _feature_dim(feature_mode)
    if policy == "gnn":
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)
        w_in = jax.random.normal(key1, (input_dim, hidden_size)) * 0.1
        b_in = jnp.zeros((hidden_size,))
        w_msg = jax.random.normal(key2, (hidden_size, hidden_size)) * 0.1
        b_msg = jnp.zeros((hidden_size,))
        w_out = jax.random.normal(key3, (hidden_size, 4)) * 0.1
        b_out = jnp.zeros((4,))
        params = {"w_in": w_in, "b_in": b_in, "w_msg": w_msg, "b_msg": b_msg, "w_out": w_out, "b_out": b_out}
        if gnn_attention:
            w_q = jax.random.normal(key4, (hidden_size, hidden_size)) * 0.1
            w_k = jax.random.normal(key5, (hidden_size, hidden_size)) * 0.1
            params["w_q"] = w_q
            params["w_k"] = w_k
        return params
    key1, key2, key3 = jax.random.split(key, 3)
    w1 = jax.random.normal(key1, (input_dim, hidden_size)) * 0.1
    b1 = jnp.zeros((hidden_size,))
    w2 = jax.random.normal(key2, (hidden_size, 4)) * 0.1
    b2 = jnp.zeros((4,))
    params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    extra_layers = max(int(mlp_depth), 1) - 1
    for idx in range(extra_layers):
        key3, sub = jax.random.split(key3)
        params[f"w_hidden_{idx}"] = jax.random.normal(sub, (hidden_size, hidden_size)) * 0.1
        params[f"b_hidden_{idx}"] = jnp.zeros((hidden_size,))
    return params


def _feature_dim(feature_mode: str) -> int:
    if feature_mode == "raw":
        return 4
    if feature_mode == "bbox_norm":
        return 6
    raise ValueError(f"Unknown feature_mode='{feature_mode}'")


def _features(poses: jax.Array, feature_mode: str) -> jax.Array:
    xy = poses[:, 0:2]
    theta = jnp.deg2rad(poses[:, 2:3])
    if feature_mode == "raw":
        return jnp.concatenate([xy, jnp.sin(theta), jnp.cos(theta)], axis=1)
    if feature_mode != "bbox_norm":
        raise ValueError(f"Unknown feature_mode='{feature_mode}'")

    min_xy = jnp.min(xy, axis=0)
    max_xy = jnp.max(xy, axis=0)
    center = (min_xy + max_xy) * 0.5
    max_side = jnp.maximum(jnp.max(max_xy - min_xy), 1e-6)
    xy_norm = (xy - center) / max_side
    dist_center = jnp.linalg.norm(xy - center, axis=1, keepdims=True) / max_side

    if xy.shape[0] <= 1:
        nn_dist = jnp.zeros((xy.shape[0], 1), dtype=poses.dtype)
    else:
        dists = jnp.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=-1)
        dists = dists + jnp.eye(xy.shape[0]) * 1e9
        nn_dist = jnp.sqrt(jnp.min(dists, axis=1, keepdims=True)) / max_side

    return jnp.concatenate([xy_norm, jnp.sin(theta), jnp.cos(theta), dist_center, nn_dist], axis=1)


def _mlp_apply(
    params: Params,
    poses: jax.Array,
    depth: int,
    feature_mode: str,
) -> Tuple[jax.Array, jax.Array]:
    feats = _features(poses, feature_mode)
    h = jnp.tanh(feats @ params["w1"] + params["b1"])
    extra_layers = max(int(depth), 1) - 1
    for idx in range(extra_layers):
        w_key = f"w_hidden_{idx}"
        b_key = f"b_hidden_{idx}"
        if w_key not in params or b_key not in params:
            break
        h = jnp.tanh(h @ params[w_key] + params[b_key])
    out = h @ params["w2"] + params["b2"]
    logits = out[:, 0]
    mean = out[:, 1:4]
    return logits, mean


def _knn_indices(xy: jax.Array, k: int) -> jax.Array:
    n = xy.shape[0]
    dists = jnp.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=-1)
    dists = dists + jnp.eye(n) * 1e9
    return jnp.argsort(dists, axis=1)[:, :k]


def _gnn_apply(
    params: Params,
    poses: jax.Array,
    knn_k: int,
    steps: int,
    attention: bool,
    feature_mode: str,
) -> Tuple[jax.Array, jax.Array]:
    feats = _features(poses, feature_mode)
    h0 = jnp.tanh(feats @ params["w_in"] + params["b_in"])
    h = h0
    idx = _knn_indices(poses[:, :2], knn_k)
    use_attn = attention and "w_q" in params and "w_k" in params
    for _ in range(max(int(steps), 1)):
        neigh = jnp.take(h, idx, axis=0)
        if use_attn:
            q = h @ params["w_q"]
            k = h @ params["w_k"]
            k_neigh = jnp.take(k, idx, axis=0)
            scale = jnp.sqrt(k.shape[-1]).astype(h.dtype)
            scores = jnp.sum(q[:, None, :] * k_neigh, axis=-1) / (scale + 1e-9)
            weights = jax.nn.softmax(scores, axis=1)
            agg = jnp.sum(neigh * weights[:, :, None], axis=1)
        else:
            agg = jnp.mean(neigh, axis=1)
        h = jnp.tanh(h + agg @ params["w_msg"] + params["b_msg"])
    out = h @ params["w_out"] + params["b_out"]
    logits = out[:, 0]
    mean = out[:, 1:4]
    return logits, mean


def policy_apply(params: Params, poses: jax.Array, config: L2OConfig) -> Tuple[jax.Array, jax.Array]:
    if config.policy == "gnn":
        logits, mean = _gnn_apply(
            params,
            poses,
            config.knn_k,
            config.gnn_steps,
            config.gnn_attention,
            config.feature_mode,
        )
    else:
        logits, mean = _mlp_apply(params, poses, config.mlp_depth, config.feature_mode)
    if config.action_scale != 1.0:
        mean = mean * config.action_scale
    return logits, mean


def _gaussian_logprob(x: jax.Array, mean: jax.Array, scale: jax.Array) -> jax.Array:
    var = scale ** 2
    return -0.5 * jnp.sum(((x - mean) ** 2) / var + jnp.log(2.0 * math.pi * var))


def _sample_action(
    key: jax.Array,
    logits: jax.Array,
    mean: jax.Array,
    trans_sigma: float,
    rot_sigma: float,
    action_noise: bool,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    idx = jax.random.categorical(key, logits)
    key, sub = jax.random.split(key)
    scales = jnp.array([trans_sigma, trans_sigma, rot_sigma])
    if action_noise:
        noise = jax.random.normal(sub, (3,))
        delta = mean[idx] + noise * scales
    else:
        delta = mean[idx]
    logp_idx = jax.nn.log_softmax(logits)[idx]
    logp_delta = _gaussian_logprob(delta, mean[idx], scales)
    return idx, delta, logp_idx + logp_delta


def _apply_delta(poses: jax.Array, idx: jax.Array, delta: jax.Array) -> jax.Array:
    return poses.at[idx].add(delta)


def _reward_fn(poses: jax.Array, overlap_penalty: float, reward: str) -> jax.Array:
    base_poly = get_tree_polygon()
    collision = check_any_collisions(poses, base_poly)
    score = prefix_packing_score(poses) if reward == "prefix" else packing_score(poses)
    return -score - overlap_penalty * collision.astype(jnp.float32)


def rollout(
    key: jax.Array,
    params: Params,
    poses: jax.Array,
    steps: int,
    config: L2OConfig,
) -> Tuple[jax.Array, jax.Array]:
    logp_total = 0.0
    for _ in range(steps):
        logits, mean = policy_apply(params, poses, config)
        key, sub = jax.random.split(key)
        idx, delta, logp = _sample_action(
            sub,
            logits,
            mean,
            config.trans_sigma,
            config.rot_sigma,
            config.action_noise,
        )
        poses = _apply_delta(poses, idx, delta)
        poses = poses.at[:, 2].set(jnp.mod(poses[:, 2], 360.0))
        logp_total = logp_total + logp
    return poses, logp_total


def optimize_with_l2o(
    key: jax.Array,
    params: Params,
    poses: jax.Array,
    steps: int,
    config: L2OConfig,
) -> jax.Array:
    final_poses, _ = rollout(key, params, poses, steps, config)
    return final_poses


def loss_fn(
    params: Params,
    key: jax.Array,
    poses_batch: jax.Array,
    steps: int,
    config: L2OConfig,
) -> jax.Array:
    def one_rollout(k, p):
        final_poses, logp = rollout(k, params, p, steps, config)
        reward = _reward_fn(final_poses, config.overlap_penalty, config.reward)
        return reward, logp

    keys = jax.random.split(key, poses_batch.shape[0])
    reward, logp = jax.vmap(one_rollout)(keys, poses_batch)
    baseline = jax.lax.stop_gradient(jnp.mean(reward))
    loss = -jnp.mean((reward - baseline) * logp)
    return loss


def loss_with_baseline(
    params: Params,
    key: jax.Array,
    poses_batch: jax.Array,
    steps: int,
    config: L2OConfig,
    baseline: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    def one_rollout(k, p):
        final_poses, logp = rollout(k, params, p, steps, config)
        reward = _reward_fn(final_poses, config.overlap_penalty, config.reward)
        return reward, logp

    keys = jax.random.split(key, poses_batch.shape[0])
    reward, logp = jax.vmap(one_rollout)(keys, poses_batch)
    advantage = reward - baseline
    loss = -jnp.mean(advantage * logp)
    return loss, jnp.mean(reward)


def behavior_cloning_loss(
    params: Params,
    poses_batch: jax.Array,
    idx_batch: jax.Array,
    delta_batch: jax.Array,
    config: L2OConfig,
) -> jax.Array:
    scales = jnp.array([config.trans_sigma, config.trans_sigma, config.rot_sigma])

    def one_sample(poses, idx, delta):
        logits, mean = policy_apply(params, poses, config)
        logp_idx = jax.nn.log_softmax(logits)[idx]
        logp_delta = _gaussian_logprob(delta, mean[idx], scales)
        return logp_idx + logp_delta

    logp = jax.vmap(one_sample)(poses_batch, idx_batch, delta_batch)
    return -jnp.mean(logp)


def behavior_cloning_loss_weighted(
    params: Params,
    poses_batch: jax.Array,
    idx_batch: jax.Array,
    delta_batch: jax.Array,
    weights: jax.Array,
    config: L2OConfig,
) -> jax.Array:
    scales = jnp.array([config.trans_sigma, config.trans_sigma, config.rot_sigma])

    def one_sample(poses, idx, delta):
        logits, mean = policy_apply(params, poses, config)
        logp_idx = jax.nn.log_softmax(logits)[idx]
        logp_delta = _gaussian_logprob(delta, mean[idx], scales)
        return logp_idx + logp_delta

    logp = jax.vmap(one_sample)(poses_batch, idx_batch, delta_batch)
    w = jnp.maximum(weights, 0.0)
    denom = jnp.sum(w) + 1e-9
    return -jnp.sum(w * logp) / denom


def _flatten_params(params: Params, prefix: str = "") -> Dict[str, jnp.ndarray]:
    flat: Dict[str, jnp.ndarray] = {}
    for key, value in params.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_params(value, name))
        else:
            flat[name] = value
    return flat


def _unflatten_params(flat: Dict[str, jnp.ndarray]) -> Params:
    root: Dict[str, object] = {}
    for key, value in flat.items():
        parts = key.split("/")
        cur = root
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value
    return root  # type: ignore[return-value]


def save_params_npz(path, params: Params, meta: Dict[str, object] | None = None) -> None:
    import numpy as np

    payload = {k: np.array(v) for k, v in _flatten_params(params).items()}
    if meta:
        for key, value in meta.items():
            payload[f"meta/{key}"] = np.array(value)
    np.savez(path, **payload)


def load_params_npz(path) -> Tuple[Params, Dict[str, object]]:
    import numpy as np

    data = np.load(path)
    params_raw: Dict[str, jnp.ndarray] = {}
    meta: Dict[str, object] = {}
    for key in data.files:
        if key.startswith("meta/"):
            meta[key.split("/", 1)[1]] = data[key].item() if data[key].shape == () else data[key]
        else:
            params_raw[key] = jnp.array(data[key])
    return _unflatten_params(params_raw), meta
