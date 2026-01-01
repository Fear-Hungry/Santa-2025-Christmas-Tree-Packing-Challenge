import jax
import jax.numpy as jnp
from functools import partial
from packing import packing_score, prefix_packing_score
from tree import get_tree_polygon
from collisions import check_any_collisions, check_collision_for_index
from l2o import policy_apply

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

@partial(jax.jit, static_argnames=['n_steps', 'n_trees', 'objective'])
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
    
    def step_fn(state, i):
        key, poses, current_score, best_poses, best_score, temp = state
        
        # Annealing schedule
        # Linear or geometric? Geometric is common.
        # t = t_start * (t_end / t_start) ** (i / n_steps_float)
        frac = i / n_steps
        current_temp = t_start * (t_end / t_start) ** frac
        
        # 1. Propose Move
        key, subkey = jax.random.split(key)
        
        # For each batch element, pick ONE tree to perturb? 
        # Or perturb all slightly? Usually perturb one or a few.
        # Let's perturb all slightly for now (diffusion) or pick one.
        # Vectorized logic is easier if we do the same operation on all batches.
        
        # Perturb one tree per batch:
        # Pick random index per batch?
        
        # Simplified: Add small gaussian noise to all poses.
        # This is more like Langevin Dynamics.
        # For SA, usually we want a discrete move.
        
        # Let's pick a random tree index `k` for each batch.
        batch_size = poses.shape[0]
        k = jax.random.randint(subkey, (batch_size,), 0, n_trees)
        
        key, subkey_move = jax.random.split(key)
        move_choice = jax.random.uniform(subkey_move, (batch_size,))

        key, subkey_trans = jax.random.split(key)
        dxy = jax.random.normal(subkey_trans, (batch_size, 2)) * trans_sigma * current_temp

        key, subkey_rot = jax.random.split(key)
        dtheta = jax.random.normal(subkey_rot, (batch_size,)) * rot_sigma * current_temp

        rot_mask = move_choice < rot_prob
        delta_xy = dxy * (~rot_mask)[:, None]
        delta_theta = dtheta * rot_mask
        delta = jnp.concatenate([delta_xy, delta_theta[:, None]], axis=1)

        candidate_poses = poses.at[jnp.arange(batch_size), k].add(delta)
        candidate_poses = candidate_poses.at[:, :, 2].set(jnp.mod(candidate_poses[:, :, 2], 360.0))
        
        # 2. Check Constraints
        # Only the moved tree can introduce a new overlap, so we check one-vs-all.
        is_colliding = jax.vmap(lambda p, idx: check_collision_for_index(p, base_poly, idx))(candidate_poses, k)
        
        # 3. Calculate Score
        candidate_score = jax.vmap(score_fn)(candidate_poses)
        
        # 4. Metropolis Criterion
        delta = candidate_score - current_score
        
        key, subkey_accept = jax.random.split(key)
        r = jax.random.uniform(subkey_accept, (batch_size,))
        
        # Acceptance condition:
        # If !colliding AND (delta < 0 OR r < exp(-delta/T))
        should_accept = (~is_colliding) & ((delta < 0) | (r < jnp.exp(-delta / current_temp)))
        
        # Update state where accepted
        new_poses = jnp.where(should_accept[:, None, None], candidate_poses, poses)
        new_score = jnp.where(should_accept, candidate_score, current_score)
        
        # Update best
        improved = new_score < best_score
        new_best_poses = jnp.where(improved[:, None, None], new_poses, best_poses)
        new_best_score = jnp.minimum(new_score, best_score)
        
        return (key, new_poses, new_score, new_best_poses, new_best_score, current_temp), (new_score, is_colliding)

    # Init State
    batch_size = initial_poses.shape[0]
    initial_scores = jax.vmap(score_fn)(initial_poses)
    
    init_state = (random_key, initial_poses, initial_scores, initial_poses, initial_scores, t_start)
    
    final_state, history = jax.lax.scan(step_fn, init_state, jnp.arange(n_steps))
    
    _, _, _, best_poses, best_score, _ = final_state
    
    return best_poses, best_score


@partial(jax.jit, static_argnames=["n_steps", "n_trees", "objective", "policy_config"])
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
    objective="packing",
    policy_prob=1.0,
    policy_pmax=0.05,
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

    def step_fn(state, i):
        key, poses, current_score, best_poses, best_score = state

        frac = i / n_steps
        current_temp = t_start * (t_end / t_start) ** frac

        batch_size = poses.shape[0]
        batch_idx = jnp.arange(batch_size)

        # --- Heuristic proposal (baseline random SA move)
        key, subkey_k, subkey_move, subkey_trans, subkey_rot = jax.random.split(key, 5)
        k_h = jax.random.randint(subkey_k, (batch_size,), 0, n_trees)
        move_choice = jax.random.uniform(subkey_move, (batch_size,))

        dxy = jax.random.normal(subkey_trans, (batch_size, 2)) * trans_sigma * current_temp
        dtheta = jax.random.normal(subkey_rot, (batch_size,)) * rot_sigma * current_temp

        rot_mask = move_choice < rot_prob
        delta_xy = dxy * (~rot_mask)[:, None]
        delta_theta = dtheta * rot_mask
        delta_h = jnp.concatenate([delta_xy, delta_theta[:, None]], axis=1)

        # --- Policy proposal
        logits_b, mean_b = jax.vmap(lambda p: policy_apply(policy_params, p, policy_config))(poses)
        probs_b = jax.nn.softmax(logits_b, axis=1)
        pmax = jnp.max(probs_b, axis=1)

        key, subkey_gate = jax.random.split(key)
        u = jax.random.uniform(subkey_gate, (batch_size,))
        use_policy = (pmax >= policy_pmax) & (u < policy_prob)

        key, subkey_pick = jax.random.split(key)
        keys_pick = jax.random.split(subkey_pick, batch_size)
        k_p = jax.vmap(lambda kk, logit: jax.random.categorical(kk, logit))(keys_pick, logits_b)

        key, subkey_eps = jax.random.split(key)
        eps = jax.random.normal(subkey_eps, (batch_size, 3))
        mean_sel = mean_b[batch_idx, k_p]
        scales = jnp.array([trans_sigma, trans_sigma, rot_sigma]) * current_temp
        delta_p = mean_sel * current_temp + eps * scales

        # --- Mix
        k = jnp.where(use_policy, k_p, k_h)
        delta = jnp.where(use_policy[:, None], delta_p, delta_h)

        candidate_poses = poses.at[batch_idx, k].add(delta)
        candidate_poses = candidate_poses.at[:, :, 2].set(jnp.mod(candidate_poses[:, :, 2], 360.0))

        # --- Constraints: only moved tree can introduce overlap
        is_colliding = jax.vmap(lambda p, idx: check_collision_for_index(p, base_poly, idx))(candidate_poses, k)

        # --- Score + Metropolis
        candidate_score = jax.vmap(score_fn)(candidate_poses)
        dscore = candidate_score - current_score

        key, subkey_accept = jax.random.split(key)
        r = jax.random.uniform(subkey_accept, (batch_size,))
        should_accept = (~is_colliding) & ((dscore < 0) | (r < jnp.exp(-dscore / current_temp)))

        new_poses = jnp.where(should_accept[:, None, None], candidate_poses, poses)
        new_score = jnp.where(should_accept, candidate_score, current_score)

        improved = new_score < best_score
        new_best_poses = jnp.where(improved[:, None, None], new_poses, best_poses)
        new_best_score = jnp.minimum(new_score, best_score)

        return (key, new_poses, new_score, new_best_poses, new_best_score), (new_score, is_colliding, use_policy, pmax)

    initial_scores = jax.vmap(score_fn)(initial_poses)
    init_state = (random_key, initial_poses, initial_scores, initial_poses, initial_scores)
    final_state, _history = jax.lax.scan(step_fn, init_state, jnp.arange(n_steps))
    _, _, _, best_poses, best_score = final_state
    return best_poses, best_score
