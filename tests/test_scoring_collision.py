import numpy as np

from santa_packing.scoring import polygons_intersect, polygons_intersect_strict

SQUARE = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ],
    dtype=float,
)


def test_overlap() -> None:
    a = SQUARE
    b = SQUARE + np.array([0.5, 0.5], dtype=float)
    assert polygons_intersect_strict(a, b)
    assert polygons_intersect(a, b)


def test_touch_counts_as_intersection() -> None:
    a = SQUARE
    b = SQUARE + np.array([1.0, 0.0], dtype=float)
    assert polygons_intersect(a, b)


def test_separated() -> None:
    a = SQUARE
    b = SQUARE + np.array([2.0, 0.0], dtype=float)
    assert not polygons_intersect_strict(a, b)
    assert not polygons_intersect(a, b)
