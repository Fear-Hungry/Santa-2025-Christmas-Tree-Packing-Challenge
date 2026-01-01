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


def prefix_packing_score(poses):
    """
    Prefix-style objective over a single ordered packing:
    sum_{n=1..N} s_n^2 / n where s_n is the bbox max side for prefix [0..n).
    """
    base_poly = get_tree_polygon()
    transformed_polys = jax.vmap(lambda p: transform_polygon(base_poly, p))(poses)
    bboxes = jax.vmap(polygon_bbox)(transformed_polys)

    def scan_fn(carry, bbox):
        min_x, min_y, max_x, max_y = carry
        min_x = jnp.minimum(min_x, bbox[0])
        min_y = jnp.minimum(min_y, bbox[1])
        max_x = jnp.maximum(max_x, bbox[2])
        max_y = jnp.maximum(max_y, bbox[3])
        new_carry = (min_x, min_y, max_x, max_y)
        return new_carry, jnp.array([min_x, min_y, max_x, max_y])

    if bboxes.shape[0] == 0:
        return jnp.array(0.0)
    if bboxes.shape[0] == 1:
        width = bboxes[0, 2] - bboxes[0, 0]
        height = bboxes[0, 3] - bboxes[0, 1]
        return jnp.maximum(width, height) ** 2

    init = (bboxes[0, 0], bboxes[0, 1], bboxes[0, 2], bboxes[0, 3])
    _, prefix = jax.lax.scan(scan_fn, init, bboxes[1:])
    prefix_bboxes = jnp.vstack([bboxes[0], prefix])
    widths = prefix_bboxes[:, 2] - prefix_bboxes[:, 0]
    heights = prefix_bboxes[:, 3] - prefix_bboxes[:, 1]
    s_values = jnp.maximum(widths, heights)
    return prefix_score(s_values)
