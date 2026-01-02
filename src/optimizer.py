import jax
import jax.numpy as jnp
from functools import partial
from packing import packing_score, prefix_packing_score
from tree import get_tree_polygon
from collisions import check_any_collisions, check_collision_for_index
from l2o import policy_apply
from tree_bounds import TREE_RADIUS2, aabb_for_poses

# Constants
N_TREES = 25  # Starting small for testing, actual problem has thousands? No, usually 25-100 for this type? 
# Wait, the problem is packing 2500 identical trees? Or variable? 
# The snippet showed 'S200' objective, usually means 200 items?
# Let's assume N is passed in or configurable.

def check_collisions(poses, base_poly):
    """
    Checks if any pair of trees intersect.
    
    Args:
        poses: (N, 3)
        base_poly: (V, 2)
        
    Returns:
        True if any collision exists, False otherwise.
    """
    return check_any_collisions(poses, base_poly)

@partial(
    jax.jit,
    static_argnames=["n_steps", "n_trees", "objective", "cooling", "proposal", "allow_collisions"],
)
def run_sa_batch(
    random_key,
    n_steps,
    n_trees,
    initial_poses,
    t_start=1.0,
    t_end=0.001,
    trans_sigma=0.1,
    rot_sigma=15.0,
    rot_prob=0.3,
    rot_prob_end=-1.0,
    swap_prob=0.0,
    swap_prob_end=-1.0,
    cooling="geom",
    cooling_power=1.0,
    trans_sigma_nexp=0.0,
    rot_sigma_nexp=0.0,
    sigma_nref=50.0,
    proposal="random",
    smart_prob=1.0,
    smart_beta=8.0,
    smart_drift=1.0,
    smart_noise=0.25,
    overlap_lambda=0.0,
    allow_collisions=False,
    objective="packing",
):
    """
    Runs a batch of SA chains.
    
    Args:
        random_key: JAX PRNGKey
        n_steps: Number of SA steps
        n_trees: Number of trees per packing
        initial_poses: (Batch, N, 3) or None to init?
        
    Returns:
        Final poses (Batch, N, 3) and scores (Batch,)
    """
    
    base_poly = get_tree_polygon()
    score_fn = prefix_packing_score if objective == "prefix" else packing_score
    
    n_ratio = jnp.asarray(float(n_trees), dtype=initial_poses.dtype) / jnp.asarray(sigma_nref, dtype=initial_poses.dtype)
    trans_sigma_eff = jnp.asarray(trans_sigma, dtype=initial_poses.dtype) * (n_ratio ** jnp.asarray(trans_sigma_nexp, dtype=initial_poses.dtype))
    rot_sigma_eff = jnp.asarray(rot_sigma, dtype=initial_poses.dtype) * (n_ratio ** jnp.asarray(rot_sigma_nexp, dtype=initial_poses.dtype))
    overlap_lambda_t = jnp.asarray(overlap_lambda, dtype=initial_poses.dtype)
    thr2 = 4.0 * TREE_RADIUS2

    def _pair_penalty_for_index(poses_xy: jax.Array, idx: jax.Array) -> jax.Array:
        center_k = poses_xy[idx]
        d = poses_xy - center_k
        dist2 = jnp.sum(d * d, axis=1)
        pen = jnp.maximum(thr2 - dist2, 0.0)
        mask = (jnp.arange(n_trees) != idx).astype(pen.dtype)
        return jnp.sum((pen * pen) * mask)

    def _total_penalty(poses_xy: jax.Array) -> jax.Array:
        d = poses_xy[:, None, :] - poses_xy[None, :, :]
        dist2 = jnp.sum(d * d, axis=-1)
        pen = jnp.maximum(thr2 - dist2, 0.0)
        pen2 = pen * pen
        mask = jnp.triu(jnp.ones_like(pen2), k=1)
        return jnp.sum(pen2 * mask)

    def _select_bbox_inward(subkey: jax.Array, poses_one: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        bboxes = aabb_for_poses(poses_one, padded=False)
        min_x = jnp.min(bboxes[:, 0])
        min_y = jnp.min(bboxes[:, 1])
        max_x = jnp.max(bboxes[:, 2])
        max_y = jnp.max(bboxes[:, 3])
        width = max_x - min_x
        height = max_y - min_y
        use_x = width >= height

        slack_x = jnp.minimum(bboxes[:, 0] - min_x, max_x - bboxes[:, 2])
        slack_y = jnp.minimum(bboxes[:, 1] - min_y, max_y - bboxes[:, 3])
        slack = jnp.where(use_x, slack_x, slack_y)
        slack = jnp.maximum(slack, 0.0)
        scale = jnp.maximum(jnp.where(use_x, width, height), 1e-6)
        logits = -(slack / scale) * jnp.asarray(smart_beta, dtype=poses_one.dtype)
        idx = jax.random.categorical(subkey, logits)
        center = jnp.array([(min_x + max_x) * 0.5, (min_y + max_y) * 0.5], dtype=poses_one.dtype)
        return idx, center, use_x

    def step_fn(state, i):
        key, poses, current_score, current_penalty, current_colliding, best_poses, best_score, temp = state
        
        # Annealing schedule
        frac = i / n_steps
        anneal = frac ** jnp.asarray(cooling_power, dtype=initial_poses.dtype)
        if cooling in {"geom", "geometric", "exp", "exponential"}:
            current_temp = t_start * (t_end / t_start) ** anneal
        elif cooling == "linear":
            current_temp = t_start + (t_end - t_start) * anneal
        elif cooling == "log":
            # Log schedule with exact endpoints: T(0)=t_start, T(n_steps-1)=t_end.
            denom = jnp.log1p(jnp.asarray(float(max(n_steps - 1, 1)), dtype=initial_poses.dtype))
            alpha = (t_start / t_end - 1.0) / (denom + 1e-12)
            current_temp = t_start / (1.0 + alpha * jnp.log1p(i.astype(initial_poses.dtype)))
        else:
            current_temp = t_start * (t_end / t_start) ** anneal

        rot_prob_final = jnp.where(rot_prob_end >= 0.0, rot_prob_end, rot_prob)
        current_rot_prob = rot_prob + (rot_prob_final - rot_prob) * anneal
        current_rot_prob = jnp.clip(current_rot_prob, 0.0, 1.0)

        swap_prob_final = jnp.where(swap_prob_end >= 0.0, swap_prob_end, swap_prob)
        current_swap_prob = swap_prob + (swap_prob_final - swap_prob) * anneal
        current_swap_prob = jnp.clip(current_swap_prob, 0.0, 1.0)
        
        # 1. Propose Move
        key, subkey_select, subkey_k, subkey_k2, subkey_swap, subkey_rot_choice, subkey_trans, subkey_rot, subkey_gate = jax.random.split(key, 9)
        
        batch_size = poses.shape[0]
        batch_idx = jnp.arange(batch_size)

        k_rand = jax.random.randint(subkey_k, (batch_size,), 0, n_trees)

        # Swap is a permutation move (useful for objective='prefix'); it does not change
        # the set of poses (only their order), so it cannot introduce new collisions.
        swap_choice = jax.random.uniform(subkey_swap, (batch_size,))
        if n_trees > 1:
            do_swap = swap_choice < current_swap_prob
        else:
            do_swap = jnp.zeros((batch_size,), dtype=bool)
        rot_choice = jax.random.uniform(subkey_rot_choice, (batch_size,))

        eps_xy = jax.random.normal(subkey_trans, (batch_size, 2))
        eps_theta = jax.random.normal(subkey_rot, (batch_size,))

        if proposal in {"bbox", "bbox_inward", "inward", "smart", "mixed"}:
            keys_pick = jax.random.split(subkey_select, batch_size)
            k_smart, centers, use_x = jax.vmap(_select_bbox_inward)(keys_pick, poses)
            xy_k = poses[batch_idx, k_smart, 0:2]
            direction = centers - xy_k
            axis_direction = jnp.where(use_x[:, None], jnp.stack([direction[:, 0], jnp.zeros_like(direction[:, 0])], axis=1), jnp.stack([jnp.zeros_like(direction[:, 1]), direction[:, 1]], axis=1))
            norm = jnp.linalg.norm(axis_direction, axis=1, keepdims=True)
            unit = axis_direction / (norm + 1e-12)
            drift = unit * (trans_sigma_eff * current_temp * jnp.asarray(smart_drift, dtype=poses.dtype))
            noise = eps_xy * (trans_sigma_eff * current_temp * jnp.asarray(smart_noise, dtype=poses.dtype))
            dxy_smart = drift + noise
        else:
            k_smart = k_rand
            dxy_smart = eps_xy * (trans_sigma_eff * current_temp)

        dxy_rand = eps_xy * (trans_sigma_eff * current_temp)

        if proposal in {"bbox", "bbox_inward", "inward", "smart"}:
            k = k_smart
            dxy = dxy_smart
        elif proposal == "mixed":
            gate = jax.random.uniform(subkey_gate, (batch_size,)) < jnp.asarray(smart_prob, dtype=poses.dtype)
            k = jnp.where(gate, k_smart, k_rand)
            dxy = jnp.where(gate[:, None], dxy_smart, dxy_rand)
        else:
            k = k_rand
            dxy = dxy_rand

        dtheta = eps_theta * (rot_sigma_eff * current_temp)

        rot_mask = rot_choice < current_rot_prob
        delta_xy = dxy * (~rot_mask)[:, None]
        delta_theta = dtheta * rot_mask
        delta = jnp.concatenate([delta_xy, delta_theta[:, None]], axis=1)

        candidate_poses_single = poses.at[batch_idx, k].add(delta)
        candidate_poses_single = candidate_poses_single.at[:, :, 2].set(jnp.mod(candidate_poses_single[:, :, 2], 360.0))

        # Swap move (k1, k2) with k2 != k1 (when n_trees > 1).
        k1 = k_rand
        if n_trees > 1:
            k2_raw = jax.random.randint(subkey_k2, (batch_size,), 0, n_trees - 1)
            k2 = jnp.where(k2_raw >= k1, k2_raw + 1, k2_raw)
        else:
            k2 = k1
        pose1 = poses[batch_idx, k1]
        pose2 = poses[batch_idx, k2]
        candidate_poses_swap = poses.at[batch_idx, k1].set(pose2)
        candidate_poses_swap = candidate_poses_swap.at[batch_idx, k2].set(pose1)

        candidate_poses = jnp.where(do_swap[:, None, None], candidate_poses_swap, candidate_poses_single)
        
        # 2. Check Constraints
        # Only the moved tree can introduce a new overlap, so we check one-vs-all.
        is_colliding_single = jax.vmap(lambda p, idx: check_collision_for_index(p, base_poly, idx))(candidate_poses, k)
        is_colliding = jnp.where(do_swap, current_colliding, is_colliding_single)
        
        # 3. Calculate Score (+ optional overlap penalty)
        candidate_score = jax.vmap(score_fn)(candidate_poses)

        def _update_penalty() -> jax.Array:
            old_k = jax.vmap(lambda p, idx: _pair_penalty_for_index(p[:, :2], idx))(poses, k)
            new_k = jax.vmap(lambda p, idx: _pair_penalty_for_index(p[:, :2], idx))(candidate_poses, k)
            return current_penalty + (new_k - old_k) * (~do_swap).astype(current_penalty.dtype)

        candidate_penalty = jax.lax.cond(overlap_lambda_t > 0.0, _update_penalty, lambda: current_penalty)
        current_energy = current_score + overlap_lambda_t * current_penalty
        candidate_energy = candidate_score + overlap_lambda_t * candidate_penalty
        
        # 4. Metropolis Criterion
        delta = candidate_energy - current_energy
        
        key, subkey_accept = jax.random.split(key)
        r = jax.random.uniform(subkey_accept, (batch_size,))
        
        # Acceptance condition:
        # If !colliding AND (delta < 0 OR r < exp(-delta/T))
        should_accept = (delta < 0) | (r < jnp.exp(-delta / current_temp))
        if not allow_collisions:
            should_accept = should_accept & (~is_colliding)
        
        # Update state where accepted
        new_poses = jnp.where(should_accept[:, None, None], candidate_poses, poses)
        new_score = jnp.where(should_accept, candidate_score, current_score)
        new_penalty = jnp.where(should_accept, candidate_penalty, current_penalty)
        new_colliding = jnp.where(should_accept, is_colliding, current_colliding)
        
        # Update best
        improved = (new_score < best_score) & (~new_colliding)
        new_best_poses = jnp.where(improved[:, None, None], new_poses, best_poses)
        new_best_score = jnp.where(improved, new_score, best_score)
        
        return (key, new_poses, new_score, new_penalty, new_colliding, new_best_poses, new_best_score, current_temp), (new_score, is_colliding)

    # Init State
    batch_size = initial_poses.shape[0]
    initial_scores = jax.vmap(score_fn)(initial_poses)
    initial_penalty = jax.lax.cond(
        overlap_lambda_t > 0.0,
        lambda: jax.vmap(lambda p: _total_penalty(p[:, :2]))(initial_poses),
        lambda: jnp.zeros((batch_size,), dtype=initial_poses.dtype),
    )
    initial_colliding = jnp.zeros((batch_size,), dtype=bool)
    
    init_state = (random_key, initial_poses, initial_scores, initial_penalty, initial_colliding, initial_poses, initial_scores, t_start)
    
    final_state, history = jax.lax.scan(step_fn, init_state, jnp.arange(n_steps))
    
    _, _, _, _, _, best_poses, best_score, _ = final_state
    
    return best_poses, best_score


@partial(
    jax.jit,
    static_argnames=["n_steps", "n_trees", "objective", "cooling", "allow_collisions"],
)
def run_sa_blocks_batch(
    random_key,
    n_steps,
    n_trees,
    initial_poses,
    blocks,
    blocks_mask,
    t_start=1.0,
    t_end=0.001,
    trans_sigma=0.1,
    rot_sigma=15.0,
    rot_prob=0.3,
    rot_prob_end=-1.0,
    cooling="geom",
    cooling_power=1.0,
    trans_sigma_nexp=0.0,
    rot_sigma_nexp=0.0,
    sigma_nref=50.0,
    overlap_lambda=0.0,
    allow_collisions=False,
    objective="packing",
):
    """Simulated annealing over *rigid blocks* of trees.

    This applies the same translation/rotation delta to every tree index inside
    the selected block. It is intended as a "meta-model" to reduce degrees of
    freedom (e.g., block_size=2..4) before refining individual trees.

    Args:
        blocks: (B, K) int32 array of tree indices.
        blocks_mask: (B, K) bool array; False entries are ignored (padding).
    """

    base_poly = get_tree_polygon()
    score_fn = prefix_packing_score if objective == "prefix" else packing_score

    n_ratio = jnp.asarray(float(n_trees), dtype=initial_poses.dtype) / jnp.asarray(sigma_nref, dtype=initial_poses.dtype)
    trans_sigma_eff = jnp.asarray(trans_sigma, dtype=initial_poses.dtype) * (n_ratio ** jnp.asarray(trans_sigma_nexp, dtype=initial_poses.dtype))
    rot_sigma_eff = jnp.asarray(rot_sigma, dtype=initial_poses.dtype) * (n_ratio ** jnp.asarray(rot_sigma_nexp, dtype=initial_poses.dtype))
    overlap_lambda_t = jnp.asarray(overlap_lambda, dtype=initial_poses.dtype)
    thr2 = 4.0 * TREE_RADIUS2

    def _pair_penalty_for_index(poses_xy: jax.Array, idx: jax.Array) -> jax.Array:
        center_k = poses_xy[idx]
        d = poses_xy - center_k
        dist2 = jnp.sum(d * d, axis=1)
        pen = jnp.maximum(thr2 - dist2, 0.0)
        mask = (jnp.arange(n_trees) != idx).astype(pen.dtype)
        return jnp.sum((pen * pen) * mask)

    def _total_penalty(poses_xy: jax.Array) -> jax.Array:
        d = poses_xy[:, None, :] - poses_xy[None, :, :]
        dist2 = jnp.sum(d * d, axis=-1)
        pen = jnp.maximum(thr2 - dist2, 0.0)
        pen2 = pen * pen
        mask = jnp.triu(jnp.ones_like(pen2), k=1)
        return jnp.sum(pen2 * mask)

    def _rotate_about(xy: jax.Array, *, center: jax.Array, delta_deg: jax.Array) -> jax.Array:
        rad = jnp.deg2rad(delta_deg)[:, None, None]
        c = jnp.cos(rad)
        s = jnp.sin(rad)
        v = xy - center[:, None, :]
        x = v[:, :, 0:1]
        y = v[:, :, 1:2]
        xr = x * c - y * s
        yr = x * s + y * c
        return center[:, None, :] + jnp.concatenate([xr, yr], axis=2)

    def step_fn(state, i):
        key, poses, current_score, current_penalty, current_colliding, best_poses, best_score, temp = state

        # Annealing schedule (match run_sa_batch)
        frac = i / n_steps
        anneal = frac ** jnp.asarray(cooling_power, dtype=initial_poses.dtype)
        if cooling in {"geom", "geometric", "exp", "exponential"}:
            current_temp = t_start * (t_end / t_start) ** anneal
        elif cooling == "linear":
            current_temp = t_start + (t_end - t_start) * anneal
        elif cooling == "log":
            denom = jnp.log1p(jnp.asarray(float(max(n_steps - 1, 1)), dtype=initial_poses.dtype))
            alpha = (t_start / t_end - 1.0) / (denom + 1e-12)
            current_temp = t_start / (1.0 + alpha * jnp.log1p(i.astype(initial_poses.dtype)))
        else:
            current_temp = t_start * (t_end / t_start) ** anneal

        rot_prob_final = jnp.where(rot_prob_end >= 0.0, rot_prob_end, rot_prob)
        current_rot_prob = rot_prob + (rot_prob_final - rot_prob) * anneal
        current_rot_prob = jnp.clip(current_rot_prob, 0.0, 1.0)

        key, subkey_block, subkey_move, subkey_trans, subkey_rot = jax.random.split(key, 5)
        batch_size = poses.shape[0]
        batch_idx = jnp.arange(batch_size)

        n_blocks = blocks.shape[0]
        b = jax.random.randint(subkey_block, (batch_size,), 0, n_blocks)
        blk_idx = blocks[b]  # (batch, K)
        blk_mask = blocks_mask[b]  # (batch, K)

        move_choice = jax.random.uniform(subkey_move, (batch_size,))
        rot_mask = move_choice < current_rot_prob

        eps_xy = jax.random.normal(subkey_trans, (batch_size, 2))
        eps_theta = jax.random.normal(subkey_rot, (batch_size,))
        dxy = eps_xy * (trans_sigma_eff * current_temp)
        dtheta = eps_theta * (rot_sigma_eff * current_temp)

        # Gather current block poses.
        poses_blk = poses[batch_idx[:, None], blk_idx]  # (batch, K, 3)
        xy = poses_blk[:, :, 0:2]
        theta = poses_blk[:, :, 2]
        mask_f = blk_mask.astype(poses.dtype)
        denom = jnp.maximum(jnp.sum(mask_f, axis=1, keepdims=True), 1.0)
        center = jnp.sum(xy * mask_f[:, :, None], axis=1) / denom  # (batch, 2)

        xy_trans = xy + dxy[:, None, :]
        xy_rot = _rotate_about(xy, center=center, delta_deg=dtheta)
        theta_rot = theta + dtheta[:, None]

        cand_xy = jnp.where(rot_mask[:, None, None], xy_rot, xy_trans)
        cand_theta = jnp.where(rot_mask[:, None], theta_rot, theta)
        cand_theta = jnp.mod(cand_theta, 360.0)

        # Scatter updated block poses back.
        candidate_poses = poses
        k_size = blk_idx.shape[1]
        for t in range(k_size):
            idx_t = blk_idx[:, t]
            m_t = blk_mask[:, t]
            safe_idx = jnp.where(m_t, idx_t, 0)
            new_pose = jnp.concatenate([cand_xy[:, t, :], cand_theta[:, t : t + 1]], axis=1)
            cur_pose = candidate_poses[batch_idx, safe_idx]
            out_pose = jnp.where(m_t[:, None], new_pose, cur_pose)
            candidate_poses = candidate_poses.at[batch_idx, safe_idx].set(out_pose)

        # 2. Check Constraints: moved indices only (K one-vs-all checks).
        is_colliding = jnp.zeros((batch_size,), dtype=bool)
        for t in range(k_size):
            idx_t = blk_idx[:, t]
            m_t = blk_mask[:, t]
            safe_idx = jnp.where(m_t, idx_t, 0)
            coll_t = jax.vmap(lambda p, idx: check_collision_for_index(p, base_poly, idx))(candidate_poses, safe_idx)
            is_colliding = is_colliding | (coll_t & m_t)

        # 3. Calculate Score (+ optional overlap penalty)
        candidate_score = jax.vmap(score_fn)(candidate_poses)

        def _update_penalty() -> jax.Array:
            old_xy = poses[:, :, 0:2]
            new_xy = candidate_poses[:, :, 0:2]
            delta_pen = jnp.zeros((batch_size,), dtype=poses.dtype)
            for t in range(k_size):
                idx_t = blk_idx[:, t]
                m_t = blk_mask[:, t]
                safe_idx = jnp.where(m_t, idx_t, 0)
                old_k = jax.vmap(lambda pxy, idx: _pair_penalty_for_index(pxy, idx))(old_xy, safe_idx)
                new_k = jax.vmap(lambda pxy, idx: _pair_penalty_for_index(pxy, idx))(new_xy, safe_idx)
                delta_pen = delta_pen + (new_k - old_k) * m_t.astype(poses.dtype)
            return current_penalty + delta_pen

        candidate_penalty = jax.lax.cond(overlap_lambda_t > 0.0, _update_penalty, lambda: current_penalty)
        current_energy = current_score + overlap_lambda_t * current_penalty
        candidate_energy = candidate_score + overlap_lambda_t * candidate_penalty

        # 4. Metropolis Criterion
        delta = candidate_energy - current_energy
        key, subkey_accept = jax.random.split(key)
        r = jax.random.uniform(subkey_accept, (batch_size,))
        should_accept = (delta < 0) | (r < jnp.exp(-delta / current_temp))
        if not allow_collisions:
            should_accept = should_accept & (~is_colliding)

        new_poses = jnp.where(should_accept[:, None, None], candidate_poses, poses)
        new_score = jnp.where(should_accept, candidate_score, current_score)
        new_penalty = jnp.where(should_accept, candidate_penalty, current_penalty)
        new_colliding = jnp.where(should_accept, is_colliding, current_colliding)

        improved = (new_score < best_score) & (~new_colliding)
        new_best_poses = jnp.where(improved[:, None, None], new_poses, best_poses)
        new_best_score = jnp.where(improved, new_score, best_score)

        return (key, new_poses, new_score, new_penalty, new_colliding, new_best_poses, new_best_score, current_temp), (
            new_score,
            is_colliding,
        )

    batch_size = initial_poses.shape[0]
    initial_scores = jax.vmap(score_fn)(initial_poses)
    initial_penalty = jax.lax.cond(
        overlap_lambda_t > 0.0,
        lambda: jax.vmap(lambda p: _total_penalty(p[:, :2]))(initial_poses),
        lambda: jnp.zeros((batch_size,), dtype=initial_poses.dtype),
    )
    initial_colliding = jnp.zeros((batch_size,), dtype=bool)
    init_state = (
        random_key,
        initial_poses,
        initial_scores,
        initial_penalty,
        initial_colliding,
        initial_poses,
        initial_scores,
        t_start,
    )
    final_state, history = jax.lax.scan(step_fn, init_state, jnp.arange(n_steps))
    _, _, _, _, _, best_poses, best_score, _ = final_state
    return best_poses, best_score


@partial(
    jax.jit,
    static_argnames=["n_steps", "n_trees", "objective", "policy_config", "cooling", "proposal", "allow_collisions"],
)
def run_sa_batch_guided(
    random_key,
    n_steps,
    n_trees,
    initial_poses,
    policy_params,
    policy_config,
    t_start=1.0,
    t_end=0.001,
    trans_sigma=0.1,
    rot_sigma=15.0,
    rot_prob=0.3,
    rot_prob_end=-1.0,
    swap_prob=0.0,
    swap_prob_end=-1.0,
    cooling="geom",
    cooling_power=1.0,
    trans_sigma_nexp=0.0,
    rot_sigma_nexp=0.0,
    sigma_nref=50.0,
    proposal="random",
    smart_prob=1.0,
    smart_beta=8.0,
    smart_drift=1.0,
    smart_noise=0.25,
    overlap_lambda=0.0,
    allow_collisions=False,
    objective="packing",
    policy_prob=1.0,
    policy_pmax=0.05,
    policy_prob_end=-1.0,
    policy_pmax_end=-1.0,
):
    """Runs SA where the proposal is a hybrid: learned policy OR heuristic fallback.

    Proposal selection (per batch element):
      - Compute policy logits over trees.
      - If max softmax prob < `policy_pmax`, fallback to heuristic proposal.
      - Else use policy proposal with probability `policy_prob`, otherwise heuristic.

    The heuristic is the same baseline used by `run_sa_batch` (random k + gaussian move).
    """

    base_poly = get_tree_polygon()
    score_fn = prefix_packing_score if objective == "prefix" else packing_score

    n_ratio = jnp.asarray(float(n_trees), dtype=initial_poses.dtype) / jnp.asarray(sigma_nref, dtype=initial_poses.dtype)
    trans_sigma_eff = jnp.asarray(trans_sigma, dtype=initial_poses.dtype) * (n_ratio ** jnp.asarray(trans_sigma_nexp, dtype=initial_poses.dtype))
    rot_sigma_eff = jnp.asarray(rot_sigma, dtype=initial_poses.dtype) * (n_ratio ** jnp.asarray(rot_sigma_nexp, dtype=initial_poses.dtype))
    overlap_lambda_t = jnp.asarray(overlap_lambda, dtype=initial_poses.dtype)
    thr2 = 4.0 * TREE_RADIUS2

    def _pair_penalty_for_index(poses_xy: jax.Array, idx: jax.Array) -> jax.Array:
        center_k = poses_xy[idx]
        d = poses_xy - center_k
        dist2 = jnp.sum(d * d, axis=1)
        pen = jnp.maximum(thr2 - dist2, 0.0)
        mask = (jnp.arange(n_trees) != idx).astype(pen.dtype)
        return jnp.sum((pen * pen) * mask)

    def _total_penalty(poses_xy: jax.Array) -> jax.Array:
        d = poses_xy[:, None, :] - poses_xy[None, :, :]
        dist2 = jnp.sum(d * d, axis=-1)
        pen = jnp.maximum(thr2 - dist2, 0.0)
        pen2 = pen * pen
        mask = jnp.triu(jnp.ones_like(pen2), k=1)
        return jnp.sum(pen2 * mask)

    def _select_bbox_inward(subkey: jax.Array, poses_one: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        bboxes = aabb_for_poses(poses_one, padded=False)
        min_x = jnp.min(bboxes[:, 0])
        min_y = jnp.min(bboxes[:, 1])
        max_x = jnp.max(bboxes[:, 2])
        max_y = jnp.max(bboxes[:, 3])
        width = max_x - min_x
        height = max_y - min_y
        use_x = width >= height

        slack_x = jnp.minimum(bboxes[:, 0] - min_x, max_x - bboxes[:, 2])
        slack_y = jnp.minimum(bboxes[:, 1] - min_y, max_y - bboxes[:, 3])
        slack = jnp.where(use_x, slack_x, slack_y)
        slack = jnp.maximum(slack, 0.0)
        scale = jnp.maximum(jnp.where(use_x, width, height), 1e-6)
        logits = -(slack / scale) * jnp.asarray(smart_beta, dtype=poses_one.dtype)
        idx = jax.random.categorical(subkey, logits)
        center = jnp.array([(min_x + max_x) * 0.5, (min_y + max_y) * 0.5], dtype=poses_one.dtype)
        return idx, center, use_x

    def step_fn(state, i):
        key, poses, current_score, current_penalty, current_colliding, best_poses, best_score = state

        frac = i / n_steps
        anneal = frac ** jnp.asarray(cooling_power, dtype=initial_poses.dtype)
        if cooling in {"geom", "geometric", "exp", "exponential"}:
            current_temp = t_start * (t_end / t_start) ** anneal
        elif cooling == "linear":
            current_temp = t_start + (t_end - t_start) * anneal
        elif cooling == "log":
            denom = jnp.log1p(jnp.asarray(float(max(n_steps - 1, 1)), dtype=initial_poses.dtype))
            alpha = (t_start / t_end - 1.0) / (denom + 1e-12)
            current_temp = t_start / (1.0 + alpha * jnp.log1p(i.astype(initial_poses.dtype)))
        else:
            current_temp = t_start * (t_end / t_start) ** anneal

        rot_prob_final = jnp.where(rot_prob_end >= 0.0, rot_prob_end, rot_prob)
        current_rot_prob = rot_prob + (rot_prob_final - rot_prob) * anneal
        current_rot_prob = jnp.clip(current_rot_prob, 0.0, 1.0)

        swap_prob_final = jnp.where(swap_prob_end >= 0.0, swap_prob_end, swap_prob)
        current_swap_prob = swap_prob + (swap_prob_final - swap_prob) * anneal
        current_swap_prob = jnp.clip(current_swap_prob, 0.0, 1.0)

        policy_prob_final = jnp.where(policy_prob_end >= 0.0, policy_prob_end, policy_prob)
        current_policy_prob = policy_prob + (policy_prob_final - policy_prob) * anneal
        current_policy_prob = jnp.clip(current_policy_prob, 0.0, 1.0)

        pmax_final = jnp.where(policy_pmax_end >= 0.0, policy_pmax_end, policy_pmax)
        current_policy_pmax = policy_pmax + (pmax_final - policy_pmax) * anneal
        current_policy_pmax = jnp.clip(current_policy_pmax, 0.0, 1.0)

        batch_size = poses.shape[0]
        batch_idx = jnp.arange(batch_size)

        # --- Heuristic proposal (baseline random SA move)
        key, subkey_select, subkey_k, subkey_k2, subkey_swap, subkey_rot_choice, subkey_trans, subkey_rot, subkey_gate = jax.random.split(key, 9)
        k_rand = jax.random.randint(subkey_k, (batch_size,), 0, n_trees)
        swap_choice = jax.random.uniform(subkey_swap, (batch_size,))
        if n_trees > 1:
            do_swap = swap_choice < current_swap_prob
        else:
            do_swap = jnp.zeros((batch_size,), dtype=bool)
        rot_choice = jax.random.uniform(subkey_rot_choice, (batch_size,))

        eps_xy = jax.random.normal(subkey_trans, (batch_size, 2))
        eps_theta = jax.random.normal(subkey_rot, (batch_size,))

        if proposal in {"bbox", "bbox_inward", "inward", "smart", "mixed"}:
            keys_pick = jax.random.split(subkey_select, batch_size)
            k_smart, centers, use_x = jax.vmap(_select_bbox_inward)(keys_pick, poses)
            xy_k = poses[batch_idx, k_smart, 0:2]
            direction = centers - xy_k
            axis_direction = jnp.where(use_x[:, None], jnp.stack([direction[:, 0], jnp.zeros_like(direction[:, 0])], axis=1), jnp.stack([jnp.zeros_like(direction[:, 1]), direction[:, 1]], axis=1))
            norm = jnp.linalg.norm(axis_direction, axis=1, keepdims=True)
            unit = axis_direction / (norm + 1e-12)
            drift = unit * (trans_sigma_eff * current_temp * jnp.asarray(smart_drift, dtype=poses.dtype))
            noise = eps_xy * (trans_sigma_eff * current_temp * jnp.asarray(smart_noise, dtype=poses.dtype))
            dxy_smart = drift + noise
        else:
            k_smart = k_rand
            dxy_smart = eps_xy * (trans_sigma_eff * current_temp)

        dxy_rand = eps_xy * (trans_sigma_eff * current_temp)

        if proposal in {"bbox", "bbox_inward", "inward", "smart"}:
            k_h = k_smart
            dxy_h = dxy_smart
        elif proposal == "mixed":
            gate = jax.random.uniform(subkey_gate, (batch_size,)) < jnp.asarray(smart_prob, dtype=poses.dtype)
            k_h = jnp.where(gate, k_smart, k_rand)
            dxy_h = jnp.where(gate[:, None], dxy_smart, dxy_rand)
        else:
            k_h = k_rand
            dxy_h = dxy_rand

        dtheta_h = eps_theta * (rot_sigma_eff * current_temp)

        rot_mask = rot_choice < current_rot_prob
        delta_xy = dxy_h * (~rot_mask)[:, None]
        delta_theta = dtheta_h * rot_mask
        delta_h = jnp.concatenate([delta_xy, delta_theta[:, None]], axis=1)

        # --- Policy proposal
        logits_b, mean_b = jax.vmap(lambda p: policy_apply(policy_params, p, policy_config))(poses)
        probs_b = jax.nn.softmax(logits_b, axis=1)
        pmax = jnp.max(probs_b, axis=1)

        key, subkey_gate = jax.random.split(key)
        u = jax.random.uniform(subkey_gate, (batch_size,))
        use_policy = (pmax >= current_policy_pmax) & (u < current_policy_prob)

        key, subkey_pick = jax.random.split(key)
        keys_pick = jax.random.split(subkey_pick, batch_size)
        k_p = jax.vmap(lambda kk, logit: jax.random.categorical(kk, logit))(keys_pick, logits_b)

        key, subkey_eps = jax.random.split(key)
        eps = jax.random.normal(subkey_eps, (batch_size, 3))
        mean_sel = mean_b[batch_idx, k_p]
        scales = jnp.array([trans_sigma_eff, trans_sigma_eff, rot_sigma_eff]) * current_temp
        delta_p = mean_sel * current_temp + eps * scales

        # --- Mix
        k = jnp.where(use_policy, k_p, k_h)
        delta = jnp.where(use_policy[:, None], delta_p, delta_h)

        candidate_poses_single = poses.at[batch_idx, k].add(delta)
        candidate_poses_single = candidate_poses_single.at[:, :, 2].set(jnp.mod(candidate_poses_single[:, :, 2], 360.0))

        # Swap move (useful for objective='prefix'): permutes the order without changing geometry.
        k1 = k_rand
        if n_trees > 1:
            k2_raw = jax.random.randint(subkey_k2, (batch_size,), 0, n_trees - 1)
            k2 = jnp.where(k2_raw >= k1, k2_raw + 1, k2_raw)
        else:
            k2 = k1
        pose1 = poses[batch_idx, k1]
        pose2 = poses[batch_idx, k2]
        candidate_poses_swap = poses.at[batch_idx, k1].set(pose2)
        candidate_poses_swap = candidate_poses_swap.at[batch_idx, k2].set(pose1)

        candidate_poses = jnp.where(do_swap[:, None, None], candidate_poses_swap, candidate_poses_single)

        # --- Constraints: only moved tree can introduce overlap
        is_colliding_single = jax.vmap(lambda p, idx: check_collision_for_index(p, base_poly, idx))(candidate_poses, k)
        is_colliding = jnp.where(do_swap, current_colliding, is_colliding_single)

        # --- Score (+ optional overlap penalty) + Metropolis
        candidate_score = jax.vmap(score_fn)(candidate_poses)
        def _update_penalty() -> jax.Array:
            old_k = jax.vmap(lambda p, idx: _pair_penalty_for_index(p[:, :2], idx))(poses, k)
            new_k = jax.vmap(lambda p, idx: _pair_penalty_for_index(p[:, :2], idx))(candidate_poses, k)
            return current_penalty + (new_k - old_k) * (~do_swap).astype(current_penalty.dtype)

        candidate_penalty = jax.lax.cond(overlap_lambda_t > 0.0, _update_penalty, lambda: current_penalty)
        current_energy = current_score + overlap_lambda_t * current_penalty
        candidate_energy = candidate_score + overlap_lambda_t * candidate_penalty
        dscore = candidate_energy - current_energy

        key, subkey_accept = jax.random.split(key)
        r = jax.random.uniform(subkey_accept, (batch_size,))
        should_accept = (dscore < 0) | (r < jnp.exp(-dscore / current_temp))
        if not allow_collisions:
            should_accept = should_accept & (~is_colliding)

        new_poses = jnp.where(should_accept[:, None, None], candidate_poses, poses)
        new_score = jnp.where(should_accept, candidate_score, current_score)
        new_penalty = jnp.where(should_accept, candidate_penalty, current_penalty)
        new_colliding = jnp.where(should_accept, is_colliding, current_colliding)

        improved = (new_score < best_score) & (~new_colliding)
        new_best_poses = jnp.where(improved[:, None, None], new_poses, best_poses)
        new_best_score = jnp.where(improved, new_score, best_score)

        return (key, new_poses, new_score, new_penalty, new_colliding, new_best_poses, new_best_score), (new_score, is_colliding, use_policy, pmax)

    initial_scores = jax.vmap(score_fn)(initial_poses)
    batch_size = initial_poses.shape[0]
    initial_penalty = jax.lax.cond(
        overlap_lambda_t > 0.0,
        lambda: jax.vmap(lambda p: _total_penalty(p[:, :2]))(initial_poses),
        lambda: jnp.zeros((batch_size,), dtype=initial_poses.dtype),
    )
    initial_colliding = jnp.zeros((batch_size,), dtype=bool)
    init_state = (random_key, initial_poses, initial_scores, initial_penalty, initial_colliding, initial_poses, initial_scores)
    final_state, _history = jax.lax.scan(step_fn, init_state, jnp.arange(n_steps))
    _, _, _, _, _, best_poses, best_score = final_state
    return best_poses, best_score
