import jax.numpy as jnp

from tree_data import TREE_POINTS

def get_tree_polygon() -> jnp.ndarray:
    """
    Returns the static 15-point polygon defining the Christmas Tree.
    Vertices are in counter-clockwise order.
    
    Returns:
        (15, 2) array of vertices.
    """
    # Coordinates taken from src/include/santa2025/tree_polygon.hpp
    return jnp.array(TREE_POINTS)
