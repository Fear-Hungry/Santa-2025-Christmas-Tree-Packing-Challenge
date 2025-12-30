import jax
import jax.numpy as jnp
from typing import Tuple

# A Polygon is represented as a (N, 2) array of points.
# A Point is a (2,) array.

def rotate_point(p: jnp.ndarray, theta: float) -> jnp.ndarray:
    """Rotates a point p around the origin by theta (radians)."""
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    rotation_matrix = jnp.array([[c, -s], [s, c]])
    return jnp.dot(rotation_matrix, p)

def rotate_polygon(poly: jnp.ndarray, theta: float) -> jnp.ndarray:
    """Rotates a polygon around the origin by theta (radians)."""
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    # Rotation matrix shape (2, 2)
    rotation_matrix = jnp.array([[c, -s], [s, c]])
    # poly shape (N, 2), result (N, 2)
    return jnp.dot(poly, rotation_matrix.T)

def translate_polygon(poly: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """Translates a polygon by (dx, dy)."""
    translation = jnp.array([dx, dy])
    return poly + translation

def transform_polygon(poly: jnp.ndarray, pose: jnp.ndarray) -> jnp.ndarray:
    """
    Transforms a polygon by a pose (x, y, theta_degrees).
    
    Args:
        poly: (N, 2) array of vertices
        pose: (3,) array [x, y, theta_degrees]
    
    Returns:
        Transformed polygon (N, 2)
    """
    x, y, theta_deg = pose
    theta_rad = jnp.deg2rad(theta_deg)
    
    rotated = rotate_polygon(poly, theta_rad)
    translated = translate_polygon(rotated, x, y)
    return translated

def polygon_bbox(poly: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the axis-aligned bounding box of a polygon.
    
    Returns:
        (4,) array [min_x, min_y, max_x, max_y]
    """
    min_vals = jnp.min(poly, axis=0) # (2,)
    max_vals = jnp.max(poly, axis=0) # (2,)
    return jnp.concatenate([min_vals, max_vals])
