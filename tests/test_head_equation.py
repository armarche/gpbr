import unittest
import numpy as np
from scipy.special import k0
from gpbr.direct.common.boundary import Point2D, StarlikeCurve
from gpbr.direct.common.collocation import CollocationData2D, collocation_points_2d
from gpbr.direct.common.source import SourcePoints2D
from gpbr.direct.common.distance import point_distance
from gpbr.direct.heat_equation.common import MFSData
from gpbr.direct.heat_equation.polynomial import MFSPolinomials2D, MFSPolinomials3D, calculate_2d_polinomials
from gpbr.direct.common.boundary import StarlikeCurve, starlike_circle_base, starlike_curve
from gpbr.direct.common.source import SourcePoints2D, source_points_2d
from gpbr.direct.heat_equation.polynomial import MFSPolinomials2D, calculate_2d_polinomials
from gpbr.direct.heat_equation.fundamental_sequence import fs_2d, fundamental_sequence_2d, FundamentalSequence
from gpbr.direct.heat_equation.helpers import form_fs_matrix, form_fs_vector_2d
from gpbr.direct.heat_equation.fundamental_sequence import FundamentalSequenceCoefs
from gpbr.direct.heat_equation.helpers import u_2d
from numpy.linalg import lstsq


def circle_boundary(M, radius: float):
    coll_2d = collocation_points_2d(M, startpoint=False)
    point_circle = starlike_circle_base(coll_2d)
    return point_circle*radius

def generate_test_points(num_points, inner_radius, outer_radius):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radii = np.linspace(inner_radius, outer_radius, num_points)
    points = [Point2D(r * np.cos(angle), r * np.sin(angle)) for r, angle in zip(radii, angles)]
    return points

class TestHeatEquationMFS(unittest.TestCase):
    def setUp(self):
        T = 2 # final time
        N = 9 # N+1=10 time points
        # M = 16 # number of collocation points
        M = 64 # number of collocation points
        tn = np.array([(n+1)*(T/(N+1)) for n in range(0, N+1)])
        h = T/(N+1)
        nu = np.sqrt(2/h)
        betta_array = []
        for n in range(0, N+1):
            sign = (-1)**n
            betta_array.append(sign*(4/h))
        betta_array[0] = np.nan
        betta_array
        self.mfs_data =MFSData(N, T, tn, M, betta_array, nu)
        

    # def test_u_constant(self):
    #     def u_exact_const(x,t):
    #         if t < 1e-5:
    #             return 0
    #         return 5
    #     G1_radius = 0.5
    #     Gamma1 = circle_boundary(self.mfs_data.M, G1_radius)
    #     Gamma1_source = circle_boundary(self.mfs_data.M//2, G1_radius)

    #     G2_radius = 1.5
    #     Gamma2 = circle_boundary(self.mfs_data.M, G2_radius)
    #     Gamma2_source = circle_boundary(self.mfs_data.M//2, G2_radius)

    #     test_points = generate_test_points(30, G1_radius, G2_radius)

    #     source_points = source_points_2d(0.5, 2.0, Gamma1_source, Gamma2_source)

    #     mfs_polynomyals = calculate_2d_polinomials(self.mfs_data, self.mfs_data.N)
    #     fundamental_sequence_gamma1 = fundamental_sequence_2d(Gamma1, source_points, self.mfs_data, mfs_polynomyals)
    #     fundamental_sequence_gamma2 = fundamental_sequence_2d(Gamma2, source_points, self.mfs_data, mfs_polynomyals)

    #     PHI_MAT = form_fs_matrix(fundamental_sequence_gamma1, fundamental_sequence_gamma2)

    #     F1 = np.empty((self.mfs_data.M,1), dtype=np.float64)
    #     F2 = np.empty((self.mfs_data.M,1), dtype=np.float64)
    #     F1[:] = np.nan
    #     F2[:] = np.nan

    #     time_point = self.mfs_data.tn[0]
    #     for i in range(0, self.mfs_data.M): # i = 1,...,M
    #         F1[i]= u_exact_const(Gamma1[i], time_point)

    #     for i in range(0, self.mfs_data.M): # i = 1,...,M
    #         F2[i]= u_exact_const(Gamma2[i], time_point)
    #     F =np.concatenate((F1,F2))

    #     alpha_zero = lstsq(PHI_MAT, F)[0]
    #     print(alpha_zero)

    #     for tp in test_points:
    #         u_approx = 0.0
    #         for j in range(0, self.mfs_data.M): # j =1,...,M
    #             delta = point_distance(tp, source_points[j])
    #             u_approx+= alpha_zero[j]*fs_2d(0, delta, self.mfs_data.nu, mfs_polynomyals)
    #         u_exact = u_exact_const(tp,0)
    #         print(f"Diff: {abs(u_exact-u_approx)}")
    #         # u_approx = u_2d(tp, 0, source_points, fs_coefs, mfs_polynomyals, self.mfs_data)
    #         # u_exact = u_exact_const(tp,0)
    #         # print(f"Diff: {abs(u_exact-u_approx)}")

    #     # alpha_coeeff = np.empty((self.mfs_data.N+1, self.mfs_data.M), dtype=np.float64)
    #     # fs_coefs = FundamentalSequenceCoefs(alpha_coeeff)
    #     # ## n=0, t=0.2
    #     # F = form_fs_vector_2d(0,
    #     #                       fundamental_sequence_gamma1,
    #     #                       fundamental_sequence_gamma2,
    #     #                       Gamma1,
    #     #                       Gamma2,
    #     #                       fs_coefs,
    #     #                       u_exact_const,
    #     #                       u_exact_const,
    #     #                       self.mfs_data)
    #     # alphas_zero = lstsq(PHI_MAT, F)[0]
    #     # fs_coefs.alpha[0] = alphas_zero.T
    #     # for tp in test_points:
    #     #     u_approx = u_2d(tp, 0, source_points, fs_coefs, mfs_polynomyals, self.mfs_data)
    #     #     u_exact = u_exact_const(tp,0)
    #     #     print(f"Diff: {abs(u_exact-u_approx)}")

    def test_u_nonlinear(self):
        def u_exact_const(x,t):
            if t < 1e-5:
                return 0
            return 1.5
        # def u_exact_const(x,t):
        #     if t < 1e-5:
        #         return 0
        #     return x.x**2 + x.y**2 +4*t
        # def u_exact_const(x,t):
        #     if t < 1e-5:
        #         return 0
        #     return x.x**2 + x.y**2 +4*t
        G1_radius = 0.5
        Gamma1 = circle_boundary(self.mfs_data.M, G1_radius)
        Gamma1_source = circle_boundary(self.mfs_data.M//2, G1_radius)

        G2_radius = 1.5
        Gamma2 = circle_boundary(self.mfs_data.M, G2_radius)
        Gamma2_source = circle_boundary(self.mfs_data.M//2, G2_radius)

        test_points = generate_test_points(30, G1_radius, G2_radius)

        source_points = source_points_2d(0.5, 2.0, Gamma1_source, Gamma2_source)

        PHI_GAMMA1 = np.empty((self.mfs_data.M, self.mfs_data.M), dtype=np.float64)
        for i in range(0, self.mfs_data.M): # i = 1, ..., M
            for j in range(0, self.mfs_data.M): # j = 1, ..., M
                delta = point_distance(Gamma1[i], source_points[j])
                PHI_GAMMA1[i, j] = k0(self.mfs_data.nu*delta)

        PHI_GAMMA2 = np.empty((self.mfs_data.M, self.mfs_data.M), dtype=np.float64)
        for i in range(0, self.mfs_data.M): # i = 1, ..., M
            for j in range(0, self.mfs_data.M): # j = 1, ..., M
                delta = point_distance(Gamma2[i], source_points[j])
                PHI_GAMMA2[i, j] = k0(self.mfs_data.nu*delta)


        PHI_MAT = np.concatenate((PHI_GAMMA1,PHI_GAMMA2), axis=0)

        F1 = np.empty((self.mfs_data.M,1), dtype=np.float64)
        F2 = np.empty((self.mfs_data.M,1), dtype=np.float64)
        F1[:] = np.nan
        F2[:] = np.nan

        time_point = self.mfs_data.tn[0]
        for i in range(0, self.mfs_data.M): # i = 1,...,M
            F1[i]= u_exact_const(Gamma1[i], time_point)

        for i in range(0, self.mfs_data.M): # i = 1,...,M
            F2[i]= u_exact_const(Gamma2[i], time_point)
        F =np.concatenate((F1,F2))
        print(F)

        alpha_zero = lstsq(PHI_MAT, F)[0]
        print(alpha_zero)

        for tp in test_points:
            u_approx = 0.0
            for j in range(0, self.mfs_data.M): # j =1,...,M
                delta = point_distance(tp, source_points[j])
                # u_approx+= alpha_zero[j]*fs_2d(0, delta, self.mfs_data.nu, mfs_polynomyals)
                u_approx+= alpha_zero[j]*k0(self.mfs_data.nu*delta)
            u_exact = u_exact_const(tp,0)
            print(f"Diff: {abs(u_exact-u_approx)}")
            # u_approx = u_2d(tp, 0, source_points, fs_coefs, mfs_polynomyals, self.mfs_data)
            # u_exact = u_exact_const(tp,0)
            # print(f"Diff: {abs(u_exact-u_approx)}")

        # alpha_coeeff = np.empty((self.mfs_data.N+1, self.mfs_data.M), dtype=np.float64)
        # fs_coefs = FundamentalSequenceCoefs(alpha_coeeff)
        # ## n=0, t=0.2
        # F = form_fs_vector_2d(0,
        #                       fundamental_sequence_gamma1,
        #                       fundamental_sequence_gamma2,
        #                       Gamma1,
        #                       Gamma2,
        #                       fs_coefs,
        #                       u_exact_const,
        #                       u_exact_const,
        #                       self.mfs_data)
        # alphas_zero = lstsq(PHI_MAT, F)[0]
        # fs_coefs.alpha[0] = alphas_zero.T
        # for tp in test_points:
        #     u_approx = u_2d(tp, 0, source_points, fs_coefs, mfs_polynomyals, self.mfs_data)
        #     u_exact = u_exact_const(tp,0)
        #     print(f"Diff: {abs(u_exact-u_approx)}")


if __name__ == '__main__':
    unittest.main()


        # alpha_coeeff = np.empty((self.mfs_data.N+1, self.mfs_data.M), dtype=np.float64)
        # alpha_coeeff[:] = np.nan
        # fs_coefs = FundamentalSequenceCoefs(alpha_coeeff)
        # for n in range(0, self.mfs_data.N+1):
        #     F = form_fs_vector_2d(n,
        #                         fundamental_sequence_gamma1,
        #                         fundamental_sequence_gamma2,
        #                         Gamma1,
        #                         Gamma2,
        #                         fs_coefs,
        #                         u_exact_const,
        #                         u_exact_const,
        #                         self.mfs_data)
        #     from numpy.linalg import lstsq
        #     alpha_n = lstsq(PHI_MAT, F)[0]
        #     fs_coefs.alpha[n] = alpha_n.T
        #     print(fs_coefs.alpha)