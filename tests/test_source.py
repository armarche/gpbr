import unittest
import numpy as np
from gpbr.direct.common.collocation import collocation_points_2d
from gpbr.direct.common.source import SourcePoints2D, SourcePoints3D
from gpbr.direct.common.boundary import StarlikeCurve

class TestSourcePoints2D(unittest.TestCase):
    def setUp(self):
        # Create mock StarlikeCurve objects
        self.collocation = type('collocation', (object,), {'n': 4})
        self.collocation = collocation_points_2d(4, startpoint=False)
        # self.curve1 = StarlikeCurve(self.collocation, [Point2D(i, i) for i in range(1, 5)])
        # self.curve2 = StarlikeCurve(self.collocation, [Point2D(i, i) for i in range(5, 9)])
        self.curve1 = StarlikeCurve.from_radial(self.collocation, lambda s: 1.0)
        self.curve2 = StarlikeCurve.from_radial(self.collocation, lambda s: 2.0)
        self.eta1 = 0.5
        self.eta2 = 0.8

    def test_getitem(self):
        source_points = SourcePoints2D(8, self.eta1, self.eta2, self.curve1, self.curve2)
        self.assertTrue(np.array_equal(source_points[0], self.eta2*self.curve2[0]))
        self.assertTrue(np.array_equal(source_points[4], self.eta1*self.curve1[0]))

    def test_get_point(self):
        source_points = SourcePoints2D(8, self.eta1, self.eta2, self.curve1, self.curve2)
        self.assertTrue(np.array_equal(source_points[0], self.eta2*self.curve2.point_list[0]))
        self.assertTrue(np.array_equal(source_points[4], self.eta1*self.curve1.point_list[0]))

    def test_source_points_2d(self):
        # source_points = source_points_2d(self.eta1, self.eta2, self.curve1, self.curve2)
        source_points = SourcePoints2D(8, self.eta1, self.eta2, self.curve1, self.curve2)
        self.assertEqual(source_points.M, 8)
        self.assertEqual(source_points.eta1, self.eta1)
        self.assertEqual(source_points.eta2, self.eta2)

        np.testing.assert_almost_equal(
            source_points.points(),
            np.concatenate(
                (
                    self.eta2*self.curve2.points_np.reshape(2,-1),
                    self.eta1*self.curve1.points_np.reshape(2,-1)
                ), axis=1
            ))


class DummyStarlikeSurface:
    """A minimal mock for StarlikeSurface for testing SourcePoints3D."""
    def __init__(self, mesh_np):
        self.mesh_np = mesh_np

    def __getitem__(self, idx):
        # Return the idx-th column as a Point3D (or np.ndarray)
        return self.mesh_np[:, idx]

class TestSourcePoints3D(unittest.TestCase):
    def setUp(self):
        # Create two dummy surfaces with 2 points each (so M=4)
        self.mesh1 = np.array([[1, 2], [3, 4], [5, 6]])  # shape (3, 2)
        self.mesh2 = np.array([[7, 8], [9, 10], [11, 12]])  # shape (3, 2)
        self.surf1 = DummyStarlikeSurface(self.mesh1)
        self.surf2 = DummyStarlikeSurface(self.mesh2)
        self.eta1 = 0.5
        self.eta2 = 0.8

    def test_getitem(self):
        source_points = SourcePoints3D(4, self.eta1, self.eta2, self.surf1, self.surf2)
        # First half from surf2, scaled by eta2
        np.testing.assert_array_equal(source_points[0], self.eta2 * self.mesh2[:, 0])
        np.testing.assert_array_equal(source_points[1], self.eta2 * self.mesh2[:, 1])
        # Second half from surf1, scaled by eta1
        np.testing.assert_array_equal(source_points[2], self.eta1 * self.mesh1[:, 0])
        np.testing.assert_array_equal(source_points[3], self.eta1 * self.mesh1[:, 1])

    def test_mesh(self):
        source_points = SourcePoints3D(4, self.eta1, self.eta2, self.surf1, self.surf2)
        expected_mesh = np.concatenate(
            (
                self.eta2 * self.mesh2.reshape(3, -1),
                self.eta1 * self.mesh1.reshape(3, -1)
            ),
            axis=1
        )
        np.testing.assert_almost_equal(source_points.mesh(), expected_mesh)


if __name__ == '__main__':
    unittest.main()