import unittest
import numpy as np
from gpbr.direct.common.source import SourcePoints2D, source_points_2d
from gpbr.direct.common.boundary import Point2D, StarlikeCurve

class TestSourcePoints2D(unittest.TestCase):
    def setUp(self):
        # Create mock StarlikeCurve objects
        self.collocation = type('collocation', (object,), {'n': 4})
        self.curve1 = StarlikeCurve(self.collocation, [Point2D(i, i) for i in range(1, 5)])
        self.curve2 = StarlikeCurve(self.collocation, [Point2D(i, i) for i in range(5, 9)])
        self.eta1 = 0.5
        self.eta2 = 0.8

    def test_getitem(self):
        source_points = SourcePoints2D(8, self.eta1, self.eta2, self.curve1, self.curve2)
        self.assertTrue(np.array_equal(source_points[0], self.curve2[0]))
        self.assertTrue(np.array_equal(source_points[4], self.curve1[0]))

    def test_get_point(self):
        source_points = SourcePoints2D(8, self.eta1, self.eta2, self.curve1, self.curve2)
        self.assertTrue(np.array_equal(source_points.get_point(0), self.curve2.points[0]))
        self.assertTrue(np.array_equal(source_points.get_point(4), self.curve1.points[0]))

    def test_as_boundary(self):
        source_points = SourcePoints2D(8, self.eta1, self.eta2, self.curve1, self.curve2)
        combined_boundary = source_points.as_boundary()
        expected_points = np.concatenate((self.curve1.points, self.curve2.points))
        self.assertTrue(np.array_equal(combined_boundary.points, expected_points))

    def test_source_points_2d(self):
        source_points = source_points_2d(self.eta1, self.eta2, self.curve1, self.curve2)
        self.assertEqual(source_points.M, 8)
        self.assertEqual(source_points.eta1, self.eta1)
        self.assertEqual(source_points.eta2, self.eta2)
        self.assertTrue(np.array_equal(source_points.gart1.points, [p*self.eta1 for p in self.curve1.points]))
        self.assertTrue(np.array_equal(source_points.gart2.points, [p*self.eta2 for p in self.curve2.points]))

if __name__ == '__main__':
    unittest.main()