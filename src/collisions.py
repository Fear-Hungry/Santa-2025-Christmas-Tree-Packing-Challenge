from __future__ import annotations

import jax
import jax.numpy as jnp

from geometry import transform_polygon
from physics import polygons_intersect


def check_any_collisions(poses: jax.Array, base_poly: jax.Array) -> jax.Array:
    """Return True if any pair of polygons intersects.

    Notes:
    - This is O(N^2) and intended mainly for debugging / small N.
    - For SA steps that move a single index, prefer `check_collision_for_index`.
    """

    n = poses.shape[0]
    polys = jax.vmap(lambda p: transform_polygon(base_poly, p))(poses)

    def check_pair(i, j):
        return jax.lax.cond(
            i < j,
            lambda: polygons_intersect(polys[i], polys[j]),
            lambda: jnp.array(False),
        )

    matrix = jax.vmap(lambda i: jax.vmap(lambda j: check_pair(i, j))(jnp.arange(n)))(jnp.arange(n))
    return jnp.any(matrix)


def check_collision_for_index(poses: jax.Array, base_poly: jax.Array, idx: jax.Array) -> jax.Array:
    """Return True if polygon `idx` intersects any other polygon.

    This is O(N) (one-vs-all), which is much faster than checking all pairs when
    only a single tree is moved.
    """

    n = poses.shape[0]
    polys = jax.vmap(lambda p: transform_polygon(base_poly, p))(poses)
    poly_k = polys[idx]
    hits = jax.vmap(lambda poly: polygons_intersect(poly_k, poly))(polys)
    mask = jnp.arange(n) != idx
    return jnp.any(hits & mask)

