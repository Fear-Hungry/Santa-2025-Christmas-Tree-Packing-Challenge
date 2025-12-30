import jax
import jax.numpy as jnp
from geometry import transform_polygon, polygon_bbox
from tree import get_tree_polygon


def compute_packing_bbox(poses):
    """
    Computes the bounding box of the entire packing.

    Args:
        poses: (N, 3) array of [x, y, theta]

    Returns:
        (4,) array [min_x, min_y, max_x, max_y]
    """
    base_poly = get_tree_polygon()

    # Transform all polygons
    # vmap over poses (N, 3) -> (N, 15, 2)
    transformed_polys = jax.vmap(lambda p: transform_polygon(base_poly, p))(poses)

    # Get bbox of each polygon
    # (N, 4)
    bboxes = jax.vmap(polygon_bbox)(transformed_polys)

    # Global bbox
    min_x = jnp.min(bboxes[:, 0])
    min_y = jnp.min(bboxes[:, 1])
    max_x = jnp.max(bboxes[:, 2])
    max_y = jnp.max(bboxes[:, 3])

    return jnp.array([min_x, min_y, max_x, max_y])


def packing_score(poses):
    """
    Objective function: Max side length of the bounding square.
    Minimize this.
    """
    bbox = compute_packing_bbox(poses)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return jnp.maximum(width, height)


def prefix_score(s_values):
    """
    Prefix-style score: sum_{n=1..N} s_n^2 / n.
    """
    s_values = jnp.array(s_values)
    n = jnp.arange(1, s_values.shape[0] + 1)
    return jnp.sum((s_values ** 2) / n)
