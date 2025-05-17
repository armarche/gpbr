import unittest
import numpy as np
from gpbr.direct.common.boundary import Point2D, Point3D, StarlikeCurve, StarlikeSurface
from gpbr.direct.common.collocation import CollocationData2D, CollocationData3D

class TestBoundary(unittest.TestCase):

    def test_point2d_operations(self):
        p1 = Point2D(1, 2)
        p2 = Point2D(3, 4)
        self.assertEqual(p1 * 2, Point2D(2, 4))
        self.assertEqual(2 * p1, Point2D(2, 4))
        self.assertEqual(p1 - p2, Point2D(-2, -2))
        self.assertEqual(p1 + p2, Point2D(4, 6))

    def test_point3d_operations(self):
        p1 = Point3D(1, 2, 3)
        p2 = Point3D(4, 5, 6)
        self.assertEqual(p1 * 2, Point3D(2, 4, 6))
        self.assertEqual(2 * p1, Point3D(2, 4, 6))
        self.assertEqual(p1 - p2, Point3D(-3, -3, -3))
        self.assertEqual(p1 + p2, Point3D(5, 7, 9))

    def test_starlike_curve(self):
        n = 100
        collocation = CollocationData2D(n=n, theta=np.linspace(0, 2 * np.pi, n))
        curve = StarlikeCurve.from_radial(collocation, lambda s: 1.0)
        x, y = curve.raw_points()
        np.testing.assert_almost_equal(x, np.cos(collocation.theta))
        np.testing.assert_almost_equal(y, np.sin(collocation.theta))

        # Test normals
        curve_with_normals = StarlikeCurve.from_radial_with_derivative(
            collocation, lambda s: 1.0, lambda s: 0.0
        )
        for i, theta in enumerate(collocation.theta):
            normal = curve_with_normals.normal(theta)
            expected_normal = Point2D(np.cos(theta), np.sin(theta))
            self.assertAlmostEqual(normal.x, expected_normal.x, places=6)
            self.assertAlmostEqual(normal.y, expected_normal.y, places=6)

        # Test __call__
        point = curve(0.0)
        self.assertAlmostEqual(point.x, 1.0)
        self.assertAlmostEqual(point.y, 0.0)

    def test_starlike_surface(self):
        n_theta = 50
        n_phi = 50
        theta = np.array([(np.pi*i)/(n_theta+1) for i in range(1, n_theta+1)], dtype=np.float64) ## TODO: Check why we need to add 1 in (np.pi*i)/(n_theta+1)
        phi = np.array([(2*np.pi*i)/n_phi for i in range(1, n_phi+1)], dtype=np.float64)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        collocation = CollocationData3D(
            n_theta=n_theta,
            n_phi=n_phi,
            theta=theta,
            phi=phi,
            theta_grid=theta_grid,
            phi_grid=phi_grid,
        )
        surface = StarlikeSurface.from_radial(
            collocation, lambda theta, phi: np.ones_like(theta)
        )
        x, y, z = surface.raw_points()
        np.testing.assert_almost_equal(
            x, (np.sin(theta_grid) * np.cos(phi_grid)).ravel()
        )
        np.testing.assert_almost_equal(
            y, (np.sin(theta_grid) * np.sin(phi_grid)).ravel()
        )
        np.testing.assert_almost_equal(z, np.cos(theta_grid).ravel())

        # Test normals
        surface_with_normals = StarlikeSurface.from_radial_with_derivative(
            collocation, lambda theta, phi: np.ones_like(theta)*1.6, lambda theta, phi: np.zeros_like(phi), lambda theta, phi: np.zeros_like(theta)
        )
        for i, (theta, phi) in enumerate(
            zip(collocation.theta_grid.ravel(), collocation.phi_grid.ravel())
        ):
            normal = surface_with_normals.normal(theta, phi)
            expected_normal = Point3D(
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            )
            self.assertAlmostEqual(normal.x, expected_normal.x, places=6)
            self.assertAlmostEqual(normal.y, expected_normal.y, places=6)
            self.assertAlmostEqual(normal.z, expected_normal.z, places=6)

        # Test __call__
        point = surface(0.0, 0.0)
        self.assertAlmostEqual(point.x, 0.0)
        self.assertAlmostEqual(point.y, 0.0)
        self.assertAlmostEqual(point.z, 1.0)


if __name__ == "__main__":
    unittest.main()