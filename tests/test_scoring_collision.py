import unittest

import numpy as np

from santa_packing.scoring import polygons_intersect, polygons_intersect_strict


class TestScoringCollision(unittest.TestCase):
    def setUp(self) -> None:
        self.square = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=float,
        )

    def test_overlap(self) -> None:
        a = self.square
        b = self.square + np.array([0.5, 0.5], dtype=float)
        self.assertTrue(polygons_intersect_strict(a, b))
        self.assertTrue(polygons_intersect(a, b))

    def test_touch_counts_as_intersection(self) -> None:
        a = self.square
        b = self.square + np.array([1.0, 0.0], dtype=float)
        self.assertTrue(polygons_intersect(a, b))

    def test_separated(self) -> None:
        a = self.square
        b = self.square + np.array([2.0, 0.0], dtype=float)
        self.assertFalse(polygons_intersect_strict(a, b))
        self.assertFalse(polygons_intersect(a, b))

