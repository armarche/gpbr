import unittest
import numpy as np
from numpy.polynomial import Polynomial

from gpbr.direct.heat_equation.common import MFSConfig
from gpbr.direct.heat_equation.helpers import precalculate_mfs_data
from gpbr.direct.heat_equation.polynomial import (
    MFSPolinomials3D,
    MFSPolinomials2D,
    calculate_3d_polinomials,
    calculate_2d_polinomials,
)

class TestPolynomials(unittest.TestCase):

    def setUp(self):
        T = 1 # final time
        N = 10 # N+1=10 time points
        M = 32 # number of collocation points
        self.ETA1= 0.5
        self.ETA2= 2.0
        config = MFSConfig(
            N=N,
            n_coll=M,
            n_source=M,
            T=T,
            eta1=self.ETA1,
            eta2=self.ETA2,
            f1=lambda x: 0.0,
            f2=lambda x: 0.0
        )
        self.mfs_data = precalculate_mfs_data(config)

    @unittest.skip("3D not yet implemented")
    def test_3d_polinomials_satisfy_differentiation_equation(self):
        polynomials = calculate_3d_polinomials(self.mfs_data)
        points = np.arange(0, 20)
        points

        for n, polinomial in enumerate(polynomials.polynomials):
            dv = polinomial.deriv()(points)
            d2v = polinomial.deriv(2)(points)
            right = 0
            for m in range(n):
                right += self.mfs_data.Beta[n-m]*polynomials.polynomials[m](points)
            difference = d2v - 2*self.mfs_data.nu*dv - right
            ## Note: for the polynomial of degree 10, there is a difference of 1e-4
            ## Not sure, if it is a numerical error or a bug
            self.assertTrue(np.allclose(difference, 0, atol=1e-4))
    @unittest.skip("3D not yet implemented")
    def test_3d_polinomials_coefficient_first_column_ones(self):
        polynomials = calculate_3d_polinomials(self.mfs_data)
        print(polynomials.A[:, 0])
        self.assertTrue(np.allclose(polynomials.A[:, 0], 1.0))
    @unittest.skip("3D not yet implemented")
    def test_3d_polinomials_diagonal_coefficients(self):
        def calc_diagonal(n, nu, beta1):
            if n == 0:
                return 1.0
            return -1.0/(2*nu*n)*beta1*calc_diagonal(n-1, nu, beta1)

        vals = np.empty(self.N+1)
        vals[:] = np.nan
        for n in range(self.N+1):
            vals[n] = calc_diagonal(n, self.mfs_data.nu, self.mfs_data.Beta[1])

        polynomials = calculate_3d_polinomials(self.mfs_data)
        self.assertTrue(np.allclose(polynomials.A.diagonal(), vals))
    @unittest.skip("3D not yet implemented")
    def test_3d_polinomials_coefficient_matrix_shape(self):
        polynomials = calculate_3d_polinomials(self.mfs_data)
        self.assertEqual(polynomials.A.shape, (self.N+1, self.N+1))
    @unittest.skip("3D not yet implemented")
    def test_3d_polinomial_of_zero_degree(self):
        polynomials = calculate_3d_polinomials(self.mfs_data)
        self.assertAlmostEqual(polynomials.polynomials[0].coef,[1.0])


    def test_2d_polinomials_dimensions(self):
        result = self.mfs_data.polynomials
        self.assertIsInstance(result, MFSPolinomials2D)
        self.assertEqual(result.A.shape, (self.mfs_data.N+1, self.mfs_data.N+1))
        self.assertEqual(len(result.v_polynomials), self.mfs_data.N+1)
        self.assertEqual(len(result.w_polynomials), self.mfs_data.N+1)
        for poly in result.v_polynomials:
            self.assertIsInstance(poly, Polynomial)
        for poly in result.w_polynomials:
            self.assertIsInstance(poly, Polynomial)

    def test_2d_polinomials_of_zero_degree(self):
        polynomials = self.mfs_data.polynomials
        v_poly = polynomials.v_polynomials
        w_poly = polynomials.w_polynomials

        self.assertAlmostEqual(v_poly[0].coef,[1.0])
        self.assertAlmostEqual(w_poly[0].coef,[0.0])

    def test_2d_polinomials_diagonal_coefficients(self):
        def calc_diagonal(n, nu, beta1):
            if n == 0:
                return 1.0
            return -1.0/(2*nu*n)*beta1*calc_diagonal(n-1, nu, beta1)

        vals = np.empty(self.mfs_data.N+1)
        vals[:] = np.nan
        for n in range(self.mfs_data.N+1):
            vals[n] = calc_diagonal(n, self.mfs_data.nu, self.mfs_data.Beta[1])

        polynomials = self.mfs_data.polynomials
        self.assertTrue(np.allclose(polynomials.A.diagonal(), vals))

if __name__ == '__main__':
    unittest.main()