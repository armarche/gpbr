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
        # Create two simple StarlikeCurve objects with points_np arrays
        points1 = np.array([[0, 1], [0, 0]])  # (0,0), (1,0)
        points2 = np.array([[0, 0], [0, 1]])  # (0,0), (0,1)
        starlike1 = StarlikeCurve(
            rf=None,
            drf=None,
            collocation=None,
            point_list=None,
            normal_list=None,
            points_np=points1,
            normals_np=None,
        )
        starlike2 = StarlikeCurve(
            rf=None,
            drf=None,
            collocation=None,
            point_list=None,
            normal_list=None,
            points_np=points2,
            normals_np=None,
        )
        # distances: between (0,0)-(0,0) = 0, (1,0)-(0,1) = sqrt(2)
        np.testing.assert_almost_equal(
            boundary_pointwise_distance(starlike1, starlike2),
            np.array([0.0, np.sqrt(2)])
        )

    def test_boundary_pointwise_distance_surface(self):
        # Create two simple StarlikeSurface objects with mesh_np arrays
        mesh1 = np.array([[0, 1], [0, 0], [0, 0]])  # (0,0,0), (1,0,0)
        mesh2 = np.array([[0, 0], [0, 1], [0, 0]])  # (0,0,0), (0,1,0)
        starlike1 = StarlikeSurface(
            rf=None,
            drf_phi=None,
            drf_theta=None,
            collocation=None,
            point_list=None,
            normal_list=None,
            mesh_np=mesh1,
            normals_mesh_np=None,
        )
        starlike2 = StarlikeSurface(
            rf=None,
            drf_phi=None,
            drf_theta=None,
            collocation=None,
            point_list=None,
            normal_list=None,
            mesh_np=mesh2,
            normals_mesh_np=None,
        )
        # distances: between (0,0,0)-(0,0,0) = 0, (1,0,0)-(0,1,0) = sqrt(2)
        np.testing.assert_almost_equal(
            boundary_pointwise_distance(starlike1, starlike2),
            np.array([0.0, np.sqrt(2)])
        )

if __name__ == '__main__':
    unittest.main()