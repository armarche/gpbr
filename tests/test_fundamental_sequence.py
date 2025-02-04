import unittest
import numpy as np
from scipy.special import k0
from gpbr.direct.common.boundary import Point2D, StarlikeCurve
from gpbr.direct.common.collocation import CollocationData2D
from gpbr.direct.common.source import SourcePoints2D
from gpbr.direct.common.distance import point_distance
from gpbr.direct.heat_equation.common import MFSData
from gpbr.direct.heat_equation.polynomial import MFSPolinomials2D, MFSPolinomials3D, calculate_2d_polinomials

from gpbr.direct.heat_equation.fundamental_sequence import (
    FundamentalSequence, FundamentalSequenceCoefs, fundamental_sequence_2d, fs_2d, fs_3d
)

class TestFundamentalSequence(unittest.TestCase):

    def setUp(self):
        # self.M = 5
        # self.N = 10
        # self.phis = np.random.rand(self.N+1, self.M, self.M)
        # self.alpha = np.random.rand(self.N+1, self.M, self.M)
        # self.fund_seq = FundamentalSequence(self.M, self.phis)
        # self.fund_seq_coefs = FundamentalSequenceCoefs(self.alpha)
        T = 2 # final time
        N = 9 # N+1=10 time points
        M = 8 # number of collocation points
        tn = np.array([0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6, 1.8, 2. ])
        h = T/(N+1)
        nu = np.sqrt(2/h)
        beta_array = np.array([np.nan, -20.0, 20.0, -20.0, 20.0, -20.0, 20.0, -20.0, 20.0, -20.0])
    
        self.mfs_data = MFSData(
            N=N,
            T=T,
            tn=tn,
            M=M,
            Beta=beta_array,
            nu=nu
        )
        self.N = N
        polynomials = calculate_2d_polinomials(self.mfs_data, self.N)

    # def test_fundamental_sequence_getitem(self):
    #     for n in range(self.N+1):
    #         np.testing.assert_array_equal(self.fund_seq[n], self.phis[n])

    # def test_fundamental_sequence_get(self):
    #     for n in range(self.N+1):
    #         np.testing.assert_array_equal(self.fund_seq.get(n), self.phis[n])

    # def test_fundamental_sequence_coefs_getitem(self):
    #     for n in range(self.N+1):
    #         np.testing.assert_array_equal(self.fund_seq_coefs[n], self.alpha[n])

    # def test_fundamental_sequence_coefs_get(self):
    #     for n in range(self.N+1):
    #         np.testing.assert_array_equal(self.fund_seq_coefs.get(n), self.alpha[n])

    def test_fundamental_sequence_2d_zero_is_k0(self):
        x = Point2D(0.25,1.75)
        y = Point2D(2.33,3.81)
        polynomials = calculate_2d_polinomials(self.mfs_data, self.N)
        actual = fs_2d(0,point_distance(x,y), self.mfs_data.nu, polynomials)
        self.assertAlmostEqual(actual, k0(self.mfs_data.nu*point_distance(x,y)))

    def test_fundamental_sequence_2d_zero_is_k0(self):
        x = Point2D(0.25,1.75)
        y = Point2D(2.33,3.81)
        polynomials = calculate_2d_polinomials(self.mfs_data, self.N)
        actual = fs_2d(0,point_distance(x,y), self.mfs_data.nu, polynomials)
        self.assertAlmostEqual(actual, k0(self.mfs_data.nu*point_distance(x,y)))








    # def test_fundamental_sequence_2d(self):
    #     collocation = CollocationData2D(self.M, np.linspace(0, 2*np.pi, self.M, endpoint=False))
    #     points = [Point2D(np.cos(theta), np.sin(theta)) for theta in collocation.theta]
    #     curve = StarlikeCurve(collocation, points)
    #     eta1 = 1.0  # or some appropriate value
    #     eta2 = 1.0  # or some appropriate value
    #     source_points = source_points_2d(eta1, eta2, curve, curve)
    #     eta2 = 1.0  # or some appropriate value
    #     source_points = source_points_2d(eta1, eta2, curve, curve)
    #     mfs_data = MFSData(self.N, 1.0)
    #     mfs_poly = MFSPolinomials2D(self.N)
    #     result = fundamental_sequence_2d(curve, source_points, mfs_data, mfs_poly)
    #     self.assertIsInstance(result, FundamentalSequence)
    #     self.assertEqual(result.M, self.M)
    #     self.assertEqual(result.phis.shape, (self.N+1, self.M, self.M))

    # def test_fs_2d(self):
    #     n = 5
    #     arg = 1.0
    #     nu = 1.0
    #     polynomials = MFSPolinomials2D(self.N)
    #     result = fs_2d(n, arg, nu, polynomials)
    #     self.assertIsInstance(result, np.float64)

    # def test_fs_3d(self):
    #     n = 5
    #     arg = 1.0
    #     nu = 1.0
    #     polynomials = MFSPolinomials3D(self.N)
    #     result = fs_3d(n, arg, nu, polynomials)
    #     self.assertIsInstance(result, np.float64)

if __name__ == '__main__':
    unittest.main()