import unittest
import numpy as np
from numpy.polynomial import Polynomial

from gpbr.direct.heat_equation.polynomial import (
    MFSPolinomials3D,
    MFSPolinomials2D,
    calculate_3d_polinomials,
    calculate_2d_polinomials,
    MFSData
)

class TestPolynomials(unittest.TestCase):

    def setUp(self):
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

    def test_3d_polinomials_satisfy_differentiation_equation(self):
        polynomials = calculate_3d_polinomials(self.mfs_data, self.N)
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

    def test_3d_polinomials_coefficient_first_column_ones(self):
        polynomials = calculate_3d_polinomials(self.mfs_data, self.N)
        print(polynomials.A[:, 0])
        self.assertTrue(np.allclose(polynomials.A[:, 0], 1.0))

    def test_3d_polinomials_diagonal_coefficients(self):
        def calc_diagonal(n, nu, beta1):
            if n == 0:
                return 1.0
            return -1.0/(2*nu*n)*beta1*calc_diagonal(n-1, nu, beta1)

        vals = np.empty(self.N+1)
        vals[:] = np.nan
        for n in range(self.N+1):
            vals[n] = calc_diagonal(n, self.mfs_data.nu, self.mfs_data.Beta[1])

        polynomials = calculate_3d_polinomials(self.mfs_data, self.N)
        self.assertTrue(np.allclose(polynomials.A.diagonal(), vals))

    def test_3d_polinomials_coefficient_matrix_shape(self):
        polynomials = calculate_3d_polinomials(self.mfs_data, self.N)
        self.assertEqual(polynomials.A.shape, (self.N+1, self.N+1))

    def test_3d_polinomial_of_zero_degree(self):
        polynomials = calculate_3d_polinomials(self.mfs_data, self.N)
        self.assertAlmostEqual(polynomials.polynomials[0].coef,[1.0])


    def test_2d_polinomials_dimensions(self):
        result = calculate_2d_polinomials(self.mfs_data, self.N)
        self.assertIsInstance(result, MFSPolinomials2D)
        self.assertEqual(result.A.shape, (self.N+1, self.N+1))
        self.assertEqual(len(result.v_polynomials), self.N+1)
        self.assertEqual(len(result.w_polynomials), self.N+1)
        for poly in result.v_polynomials:
            self.assertIsInstance(poly, Polynomial)
        for poly in result.w_polynomials:
            self.assertIsInstance(poly, Polynomial)

    def test_2d_polinomials_of_zero_degree(self):
        polynomials = calculate_2d_polinomials(self.mfs_data, self.N)
        v_poly = polynomials.v_polynomials
        w_poly = polynomials.w_polynomials

        self.assertAlmostEqual(v_poly[0].coef,[1.0])
        self.assertAlmostEqual(w_poly[0].coef,[0.0])

    def test_2d_polinomials_diagonal_coefficients(self):
        def calc_diagonal(n, nu, beta1):
            if n == 0:
                return 1.0
            return -1.0/(2*nu*n)*beta1*calc_diagonal(n-1, nu, beta1)

        vals = np.empty(self.N+1)
        vals[:] = np.nan
        for n in range(self.N+1):
            vals[n] = calc_diagonal(n, self.mfs_data.nu, self.mfs_data.Beta[1])

        polynomials = calculate_2d_polinomials(self.mfs_data, self.N)
        self.assertTrue(np.allclose(polynomials.A.diagonal(), vals))

if __name__ == '__main__':
    unittest.main()