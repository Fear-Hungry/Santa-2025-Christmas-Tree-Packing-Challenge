import jax
import jax.numpy as jnp
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt

from optimizer import run_sa_batch
from geometry import transform_polygon
from tree import get_tree_polygon

def plot_packing(poses, score, filename="packing.png"):
    poly = get_tree_polygon()
    
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    for pose in poses:
        t_poly = transform_polygon(poly, pose)
        # Close the polygon for plotting
        p = np.vstack([t_poly, t_poly[0]])
        plt.plot(p[:, 0], p[:, 1], 'g-')
        
    plt.axis('equal')
    plt.title(f"Packing Score: {score:.4f}")
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trees', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--trans_sigma', type=float, default=0.1)
    parser.add_argument('--rot_sigma', type=float, default=15.0)
    parser.add_argument('--rot_prob', type=float, default=0.3)
    parser.add_argument('--rot_prob_end', type=float, default=-1.0, help="Final rotation move prob (-1 keeps constant)")
    parser.add_argument('--cooling', type=str, default="geom", choices=["geom", "linear", "log"])
    parser.add_argument('--cooling_power', type=float, default=1.0, help=">=1 slows early cooling")
    parser.add_argument('--trans_nexp', type=float, default=0.0, help="Scale trans_sigma by (n/nref)^nexp")
    parser.add_argument('--rot_nexp', type=float, default=0.0, help="Scale rot_sigma by (n/nref)^nexp")
    parser.add_argument('--sigma_nref', type=float, default=50.0)
    parser.add_argument('--objective', type=str, default="packing", choices=["packing", "prefix"])
    parser.add_argument(
        "--proposal",
        type=str,
        default="random",
        choices=["random", "bbox_inward", "bbox", "inward", "smart", "mixed"],
        help="SA proposal mode. 'bbox_inward/smart' targets boundary trees; 'mixed' blends with random.",
    )
    parser.add_argument("--smart_prob", type=float, default=1.0, help="For proposal=mixed: probability of smart move.")
    parser.add_argument("--smart_beta", type=float, default=8.0, help="Edge focus strength (higher=more boundary-biased).")
    parser.add_argument("--smart_drift", type=float, default=1.0, help="Inward drift multiplier (translation moves).")
    parser.add_argument("--smart_noise", type=float, default=0.25, help="Noise multiplier for smart inward moves.")
    parser.add_argument("--overlap_lambda", type=float, default=0.0, help="Energy penalty weight for circle overlap (0 disables).")
    parser.add_argument("--allow_collisions", action="store_true", help="Allow accepting colliding states (best kept feasible).")
    args = parser.parse_args()
    
    print(f"Running JAX packing optimization with:")
    print(f"  Trees: {args.n_trees}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Steps: {args.n_steps}")
    print(f"  Device: {jax.devices()[0]}")
    
    # Initialize Poses (Grid)
    # Simple grid layout to ensure no initial collisions (approximately)
    side = int(np.ceil(np.sqrt(args.n_trees)))
    spacing = 2.0 
    
    initial_poses_single = []
    for i in range(args.n_trees):
        row = i // side
        col = i % side
        initial_poses_single.append([col * spacing, row * spacing, 0.0])
        
    initial_poses_single = jnp.array(initial_poses_single)
    
    # Replicate for batch
    initial_poses = jnp.tile(initial_poses_single, (args.batch_size, 1, 1))
    
    key = jax.random.PRNGKey(42)
    
    start_time = time.time()
    
    # JIT compile first
    print("Compiling...")
    # We call the function once? Or let JAX handle it on first call.
    # The run_sa_batch is jitted.
    
    best_poses_batch, best_scores = run_sa_batch(
        key,
        args.n_steps,
        args.n_trees,
        initial_poses,
        trans_sigma=args.trans_sigma,
        rot_sigma=args.rot_sigma,
        rot_prob=args.rot_prob,
        rot_prob_end=args.rot_prob_end,
        cooling=args.cooling,
        cooling_power=args.cooling_power,
        trans_sigma_nexp=args.trans_nexp,
        rot_sigma_nexp=args.rot_nexp,
        sigma_nref=args.sigma_nref,
        proposal=args.proposal,
        smart_prob=args.smart_prob,
        smart_beta=args.smart_beta,
        smart_drift=args.smart_drift,
        smart_noise=args.smart_noise,
        overlap_lambda=args.overlap_lambda,
        allow_collisions=args.allow_collisions,
        objective=args.objective,
    )
    # Block until ready
    best_scores.block_until_ready()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Done in {elapsed:.2f}s")
    print(f"Steps/sec: {args.n_steps * args.batch_size / elapsed:.2f} (batch steps)")
    
    best_idx = jnp.argmin(best_scores)
    best_score = best_scores[best_idx]
    
    print(f"Best Score: {best_score}")
    
    # Plot best
    plot_packing(best_poses_batch[best_idx], best_score, "best_packing.png")

if __name__ == "__main__":
    main()
