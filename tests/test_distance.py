import unittest
import numpy as np
from gpbr.direct.common.distance import point_distance, boundary_pointwise_distance
from gpbr.direct.common.boundary import Point2D, Point3D, StarlikeCurve, StarlikeSurface

class TestDistanceFunctions(unittest.TestCase):

    def test_point_distance_2d(self):
        p1 = Point2D(1, 2)
        p2 = Point2D(4, 6)
        self.assertAlmostEqual(point_distance(p1, p2), 5.0)

    def test_point_distance_2d_default(self):
        p1 = Point2D(3, 4)
        self.assertAlmostEqual(point_distance(p1), 5.0)

    def test_point_distance_3d(self):
        p1 = Point3D(1, 2, 3)
        p2 = Point3D(4, 6, 8)
        self.assertAlmostEqual(point_distance(p1, p2), 7.0710678118654755)

    def test_point_distance_3d_default(self):
        p1 = Point3D(3, 4, 5)
        self.assertAlmostEqual(point_distance(p1), 7.0710678118654755)

    def test_boundary_pointwise_distance_curve(self):
        class MockStarlikeCurve(StarlikeCurve):
            def raw_points(self):
                return np.array([1, 2]), np.array([3, 4])

        starlike1 = MockStarlikeCurve(None, None, None, None)
        starlike2 = MockStarlikeCurve(None, None, None, None)
        np.testing.assert_almost_equal(boundary_pointwise_distance(starlike1, starlike2), np.array([0, 0]))

    def test_boundary_pointwise_distance_surface(self):
        class MockStarlikeSurface(StarlikeSurface):
            def raw_points(self):
                return np.array([1, 2]), np.array([3, 4]), np.array([5, 6])

        starlike1 = MockStarlikeSurface(None, None)
        starlike2 = MockStarlikeSurface(None, None)
        np.testing.assert_almost_equal(boundary_pointwise_distance(starlike1, starlike2), np.array([0, 0, 0]))

if __name__ == '__main__':
    unittest.main()