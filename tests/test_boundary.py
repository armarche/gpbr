import unittest
import numpy as np
from gpbr.direct.common.boundary import Point2D, Point3D, StarlikeCurve, StarlikeSurface, starlike_circle_base, starlike_curve, starlike_sphere_base, starlike_surface
from gpbr.direct.common.collocation import CollocationData2D, CollocationData3D

class TestBoundary(unittest.TestCase):

    def test_point2d_operations(self):
        p1 = Point2D(1, 2)
        p2 = Point2D(3, 4)
        self.assertEqual(p1 * 2, Point2D(2, 4))
        self.assertEqual(2 * p1, Point2D(2, 4))
        self.assertEqual(p1 - p2, Point2D(-2, -2))

    def test_point3d_operations(self):
        p1 = Point3D(1, 2, 3)
        p2 = Point3D(4, 5, 6)
        self.assertEqual(p1 * 2, Point3D(2, 4, 6))
        self.assertEqual(2 * p1, Point3D(2, 4, 6))
        self.assertEqual(p1 - p2, Point3D(-3, -3, -3))

    def test_starlike_curve(self):
        n = 100
        collocation = CollocationData2D(n=n, theta=np.linspace(0, 2*np.pi, n))
        curve = StarlikeCurve.from_radial(collocation, lambda s: 1.0)
        # base_curve = starlike_circle_base(collocation)
        # r_values = np.ones(n)
        # curve = starlike_curve(r_values, base_curve)
        x, y = curve.raw_points()
        np.testing.assert_almost_equal(x, np.cos(collocation.theta))
        np.testing.assert_almost_equal(y, np.sin(collocation.theta))

    def test_starlike_surface(self):
        n_theta = 50
        n_phi = 50
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        collocation = CollocationData3D(
            n_theta=n_theta, n_phi=n_phi,
            theta=theta, phi=phi,
            theta_grid=theta_grid, phi_grid=phi_grid)
        base_surface = starlike_sphere_base(collocation)
        r_grid = np.ones_like(theta_grid)
        surface = starlike_surface(r_grid, collocation)
        x, y, z = surface.raw_points()
        np.testing.assert_almost_equal(x, (np.sin(theta_grid) * np.cos(phi_grid)).ravel())
        np.testing.assert_almost_equal(y, (np.sin(theta_grid) * np.sin(phi_grid)).ravel())
        np.testing.assert_almost_equal(z, np.cos(theta_grid).ravel())

if __name__ == '__main__':
    unittest.main()