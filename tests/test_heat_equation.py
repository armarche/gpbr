import unittest
from matplotlib.pylab import lstsq
import numpy as np

from gpbr.direct.common.boundary import Point2D, Point3D, StarlikeCurve, StarlikeSurface
from gpbr.direct.common.distance import point_distance
from gpbr.direct.common.collocation import collocation_points_2d
from gpbr.direct.common.source import SourcePoints2D, SourcePoints3D
from gpbr.direct.heat_equation.common import Dimension, MFSConfig, MFSConfig2D, MFSConfig3D
from gpbr.direct.heat_equation.fundamental_sequence import FundamentalSequenceCoefs, fundamental_sequence_2d, fundamental_sequence_3d
from gpbr.direct.heat_equation.helpers import dbu_2d, form_fs_matrix, form_fs_vector_2d, form_fs_vector_3d, precalculate_mfs_data, u_2d, u_3d
from gpbr.direct.heat_equation.polynomial import calculate_2d_polinomials


def generate_test_points(num_points, inner_radius, outer_radius):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radii = np.linspace(inner_radius, outer_radius, num_points)
    points = [Point2D(r * np.cos(angle), r * np.sin(angle)) for r, angle in zip(radii, angles)]
    return points

def generate_test_points3d(num_points, inner_radius, outer_radius):
    phi = np.linspace(0, np.pi, num_points, endpoint=False)
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radii = np.linspace(inner_radius, outer_radius, num_points)
    points = [
        Point3D(
            r * np.sin(p) * np.cos(t),
            r * np.sin(p) * np.sin(t),
            r * np.cos(p)
        )
        for r, p, t in zip(radii, phi, theta)
    ]
    return points

def u2de(xp, t):
    point = Point2D(0,4)
    dist =point_distance(point, xp)
    return 1/4/np.pi/t*np.exp(-dist**2/4/t)

def du2de(xp, nx, t):
    x1 = (-0.125*xp.x*np.exp((-xp.x**2/4 - (xp.y - 4)**2/4)/t)/(np.pi*t**2))*nx.x
    x2 = (0.25*(2 - xp.y/2)*np.exp((-xp.x**2/4 - (xp.y  - 4)**2/4)/t)/(np.pi*t**2))*nx.y
    return x1 + x2

def f1(x: Point2D, t):
    return u2de(x, t)

def f2(x: Point2D, t: float) -> np.array:
    return u2de(x, t)


def u3de(xp, t):
    point = Point3D(0,4, 0)
    dist =point_distance(point, xp)
    return 1/4/np.pi/t*np.exp(-dist**2/4/t)



class TestHeatEquation2D(unittest.TestCase):
    def setUp(self):
        T = 1 # final time
        N = 10 # N+1=10 time points
        M = 32 # number of collocation points
        self.ETA1= 0.5
        self.ETA2= 2.0
        config = MFSConfig2D(
            N=N,
            n_coll=M,
            n_source=M,
            T=T,
            eta1=self.ETA1,
            eta2=self.ETA2,
            f1=f1,
            f2=f2
        )
        self.mfs_data = precalculate_mfs_data(config)

    
    def test_equation_case1(self):
        def r1_func(s):
            return 0.8

        def r2_func(s):
            return 1.6

        coll_2d = collocation_points_2d(self.mfs_data.M, startpoint=False)
        Gamma1 = StarlikeCurve.from_radial(coll_2d, r1_func)
        Gamma2 = StarlikeCurve.from_radial(coll_2d, r2_func)

        source_coll_2d = collocation_points_2d(self.mfs_data.M//2, startpoint=False)
        Gamma1_source = StarlikeCurve.from_radial(source_coll_2d, r1_func)
        Gamma2_source = StarlikeCurve.from_radial(source_coll_2d, r2_func)
        source_points = SourcePoints2D(self.mfs_data.M, self.ETA1, self.ETA2, Gamma1_source, Gamma2_source)

        fundamental_sequence_gamma1 = fundamental_sequence_2d(Gamma1, source_points, self.mfs_data)
        fundamental_sequence_gamma2 = fundamental_sequence_2d(Gamma2, source_points, self.mfs_data)
        PHI_MAT = form_fs_matrix(fundamental_sequence_gamma1, fundamental_sequence_gamma2)
        alpha_coeeff = np.empty((self.mfs_data.N+1, self.mfs_data.M), dtype=np.float64)
        alpha_coeeff[:] = np.nan
        fs_coefs = FundamentalSequenceCoefs(alpha_coeeff)

        test_points = generate_test_points(50, 0.8, 1.6)

        for n in range(0, self.mfs_data.N+1):
            F = form_fs_vector_2d(n,
                                fundamental_sequence_gamma1,
                                fundamental_sequence_gamma2,
                                Gamma1,
                                Gamma2,
                                fs_coefs,
                                f1, f2, self.mfs_data)
            alpha_n = lstsq(PHI_MAT, F)[0]
            fs_coefs.alpha[n] = alpha_n.T


        for n in range(0, self.mfs_data.N+1):
            for p in test_points:
                u_approx = u_2d(p, n, source_points, fs_coefs, self.mfs_data)
                u_exact = u2de(p, self.mfs_data.tn[n])
                np.testing.assert_almost_equal(u_exact, u_approx, decimal=4)

    def test_equation_case2(self):
        def r1_func(s):
            return 0.11

        def r2_func(s):
            return 1.9

        coll_2d = collocation_points_2d(self.mfs_data.M, startpoint=False)
        Gamma1 = StarlikeCurve.from_radial(coll_2d, r1_func)
        Gamma2 = StarlikeCurve.from_radial(coll_2d, r2_func)

        source_coll_2d = collocation_points_2d(self.mfs_data.M//2, startpoint=False)
        Gamma1_source = StarlikeCurve.from_radial(source_coll_2d, r1_func)
        Gamma2_source = StarlikeCurve.from_radial(source_coll_2d, r2_func)
        source_points = SourcePoints2D(self.mfs_data.M, self.ETA1, self.ETA2, Gamma1_source, Gamma2_source)

        fundamental_sequence_gamma1 = fundamental_sequence_2d(Gamma1, source_points, self.mfs_data)
        fundamental_sequence_gamma2 = fundamental_sequence_2d(Gamma2, source_points, self.mfs_data)
        PHI_MAT = form_fs_matrix(fundamental_sequence_gamma1, fundamental_sequence_gamma2)
        alpha_coeeff = np.empty((self.mfs_data.N+1, self.mfs_data.M), dtype=np.float64)
        alpha_coeeff[:] = np.nan
        fs_coefs = FundamentalSequenceCoefs(alpha_coeeff)

        test_points = generate_test_points(50, 0.11, 1.9)

        for n in range(0, self.mfs_data.N+1):
            F = form_fs_vector_2d(n,
                                fundamental_sequence_gamma1,
                                fundamental_sequence_gamma2,
                                Gamma1,
                                Gamma2,
                                fs_coefs,
                                f1, f2, self.mfs_data)
            alpha_n = lstsq(PHI_MAT, F)[0]
            fs_coefs.alpha[n] = alpha_n.T


        for n in range(0, self.mfs_data.N+1):
            for p in test_points:
                u_approx = u_2d(p, n, source_points, fs_coefs, self.mfs_data)
                u_exact = u2de(p, self.mfs_data.tn[n])
                np.testing.assert_almost_equal(u_exact, u_approx, decimal=4)

    def test_equation_derivative(self):
        def r1_func(s):
            return 0.11

        def r2_func(s):
            return 1.9

        def dr1_func(s):
            return 0.0

        def dr2_func(s):
            return 0.0

        coll_2d = collocation_points_2d(self.mfs_data.M, startpoint=False)
        Gamma1 = StarlikeCurve.from_radial_with_derivative(coll_2d, r1_func, dr1_func)
        Gamma2 = StarlikeCurve.from_radial_with_derivative(coll_2d, r2_func, dr2_func)

        # source_coll_2d = collocation_points_2d(self.mfs_data.M//2, startpoint=False)
        # Gamma1_source = StarlikeCurve.from_radial(source_coll_2d, lambda s: self.ETA1*r1_func(s))
        # Gamma2_source = StarlikeCurve.from_radial(source_coll_2d, lambda s: self.ETA2*r2_func(s))
        # source_points = SourcePoints2D(self.mfs_data.M, self.ETA1, self.ETA2, Gamma1_source, Gamma2_source)
        source_coll_2d = collocation_points_2d(self.mfs_data.M//2, startpoint=False)
        Gamma1_source = StarlikeCurve.from_radial(source_coll_2d, r1_func)
        Gamma2_source = StarlikeCurve.from_radial(source_coll_2d, r2_func)
        source_points = SourcePoints2D(self.mfs_data.M, self.ETA1, self.ETA2, Gamma1_source, Gamma2_source)

        fundamental_sequence_gamma1 = fundamental_sequence_2d(Gamma1, source_points, self.mfs_data)
        fundamental_sequence_gamma2 = fundamental_sequence_2d(Gamma2, source_points, self.mfs_data)
        PHI_MAT = form_fs_matrix(fundamental_sequence_gamma1, fundamental_sequence_gamma2)
        alpha_coeeff = np.empty((self.mfs_data.N+1, self.mfs_data.M), dtype=np.float64)
        alpha_coeeff[:] = np.nan
        fs_coefs = FundamentalSequenceCoefs(alpha_coeeff)

        for n in range(0, self.mfs_data.N+1):
            F = form_fs_vector_2d(n,
                                fundamental_sequence_gamma1,
                                fundamental_sequence_gamma2,
                                Gamma1,
                                Gamma2,
                                fs_coefs,
                                f1, f2, self.mfs_data)
            alpha_n = lstsq(PHI_MAT, F)[0]
            fs_coefs.alpha[n] = alpha_n.T


        for n in range(0, self.mfs_data.N+1):
            norms = []
            for x, nx in zip(Gamma2.point_list, Gamma2.normal_list):
                du_approx = dbu_2d(x, nx, n, source_points, fs_coefs, self.mfs_data)
                du_exact = du2de(x, nx, self.mfs_data.tn[n])
                norms.append(abs(du_approx - du_exact))
                np.testing.assert_almost_equal(du_exact, du_approx, decimal=3)
class TestHeatEquation3D(unittest.TestCase):
    def setUp(self):
        T = 1 # final time
        N = 7
        m1 = 16
        m2 = 8
        M = m1 * m2 # number of collocation points
        ETA1 = 0.5
        ETA2 = 2.0
        config = MFSConfig3D(
            N=N,
            n_coll=M,
            n_source=M,
            T=T,
            eta1=ETA1,
            eta2=ETA2,
            f1=f1,
            f2=f2,
            n_coll_theta=m1,
            n_coll_phi=m2,
            n_source_theta=m1,
            n_source_phi=m2
        )
        self.mfs_data = precalculate_mfs_data(config)
    
    def test_equation_case1(self):
        def r1_func(theta, phi):
            return np.ones_like(theta) * 0.4

        def r2_func(theta, phi):
            return np.ones_like(theta) * 0.8
        
        # def r1_func(theta, phi):
        #     return np.ones_like(theta) * 0.8

        # def r2_func(theta, phi):
        #     return np.ones_like(theta) * 1.6
        
        Gamma1 = StarlikeSurface.from_radial(self.mfs_data.collocation, r1_func)
        Gamma2 = StarlikeSurface.from_radial(self.mfs_data.collocation, r2_func)

        Gamma1_source = StarlikeSurface.from_radial(self.mfs_data.source_collocation, r1_func)
        Gamma2_source = StarlikeSurface.from_radial(self.mfs_data.source_collocation, r2_func)

        src_cnt = self.mfs_data.source_collocation.n_theta * self.mfs_data.source_collocation.n_phi
        source_points = SourcePoints3D(src_cnt, self.mfs_data.eta1, self.mfs_data.eta2, Gamma1_source, Gamma2_source)

        fundamental_sequence_gamma1 = fundamental_sequence_3d(Gamma1, source_points, self.mfs_data)
        fundamental_sequence_gamma2 = fundamental_sequence_3d(Gamma2, source_points, self.mfs_data)
        PHI_MAT = form_fs_matrix(fundamental_sequence_gamma1, fundamental_sequence_gamma2)


        alpha_coeeff = np.empty((self.mfs_data.N+1, self.mfs_data.M), dtype=np.float64)
        alpha_coeeff[:] = np.nan
        fs_coefs = FundamentalSequenceCoefs(alpha_coeeff)
        for n in range(0, self.mfs_data.N+1):
            F = form_fs_vector_3d(n,
                                fundamental_sequence_gamma1,
                                fundamental_sequence_gamma2,
                                Gamma1,
                                Gamma2,
                                fs_coefs,
                                u3de, u3de, self.mfs_data)
            alpha_n = lstsq(PHI_MAT, F)[0]
            fs_coefs.alpha[n] = alpha_n.T

        # test_points = generate_test_points3d(500, 0.8, 1.6)
        test_points = generate_test_points3d(500, 0.4, 0.8)
        errs = []
        for n in range(0, self.mfs_data.N+1):
            for p in test_points:
                u_approx = u_3d(p, n, source_points, fs_coefs, self.mfs_data)
                u_exact = u3de(p, self.mfs_data.tn[n])
                err = abs(u_exact - u_approx)
                errs.append(err)
                np.testing.assert_almost_equal(u_exact, u_approx, decimal=3)

        print("\nMax error:", max(errs)) # Max error: 0.00026898224078584793
        # Max error: 0.00026898224078584793
