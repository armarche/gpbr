import unittest
import numpy as np
from gpbr.direct.common.collocation import collocation_points_2d, collocation_points_3d, linspace_points_3d, CollocationData2D, CollocationData3D

class TestCollocationPoints(unittest.TestCase):

    def test_collocation_points_2d(self):
        n_theta = 10
        result = collocation_points_2d(n_theta, startpoint=False)
        self.assertIsInstance(result, CollocationData2D)
        self.assertEqual(result.n, n_theta)
        self.assertEqual(len(result.theta), n_theta)
        np.testing.assert_almost_equal(result.theta[-1], 2*np.pi)
        np.testing.assert_almost_equal(result.theta[0], 2*np.pi/n_theta)

        result_with_startpoint = collocation_points_2d(n_theta, startpoint=True)
        self.assertEqual(result_with_startpoint.n, n_theta + 1)
        self.assertEqual(len(result_with_startpoint.theta), n_theta + 1)
        self.assertEqual(result_with_startpoint.theta[0], 0)

    def test_collocation_points_3d(self):
        n_theta = 10
        n_phi = 20
        result = collocation_points_3d(n_theta, n_phi)
        self.assertIsInstance(result, CollocationData3D)
        self.assertEqual(result.n_theta, n_theta)
        self.assertEqual(result.n_phi, n_phi)
        self.assertEqual(len(result.theta), n_theta)
        self.assertEqual(len(result.phi), n_phi)
        self.assertEqual(result.theta_grid.shape, (n_phi, n_theta))
        self.assertEqual(result.phi_grid.shape, (n_phi, n_theta))

    def test_linspace_points_3d(self):
        n_theta = 10
        n_phi = 20
        result = linspace_points_3d(n_theta, n_phi)
        self.assertIsInstance(result, CollocationData3D)
        self.assertEqual(result.n_theta, n_theta)
        self.assertEqual(result.n_phi, n_phi)
        self.assertEqual(len(result.theta), n_theta)
        self.assertEqual(len(result.phi), n_phi)
        self.assertEqual(result.theta_grid.shape, (n_phi, n_theta))
        self.assertEqual(result.phi_grid.shape, (n_phi, n_theta))

if __name__ == '__main__':
    unittest.main()