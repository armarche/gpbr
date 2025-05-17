"""
MFS helpers
"""

from collections.abc import Callable

from ..common.collocation import collocation_points_2d, collocation_points_3d

from ..common.distance import point_distance

from ..common.source import SourcePoints2D, SourcePoints3D, source_points_2d
from .polynomial import MFSPolinomials2D, MFSPolinomials3D, calculate_2d_polinomials, calculate_3d_polinomials

from ..common.boundary import Point2D, Point3D, StarlikeCurve, StarlikeSurface, starlike_circle_base, starlike_curve
from .common import MFSConfig, MFSData, Dimension
from .fundamental_sequence import FundamentalSequence, FundamentalSequenceCoefs, dfs_2d, dfs_3d, fs_2d, fs_3d, matdfs_2d, matdfs_3d, matfs_2d, matfs_3d
import numpy as np


def form_fs_matrix(g1_sequnce: FundamentalSequence, g2_sequnce: FundamentalSequence) -> np.ndarray:
    """
    Form the Phi_0 matrix for the method of fundamental solutions
    """
    # return np.concatenate((g1_sequnce.get(0), g2_sequnce.get(0)), axis=0)
    return np.concatenate((g1_sequnce[0], g2_sequnce[0]), axis=0)


def form_fs_vector_2d(
        n: int,
        g1_sequnce: FundamentalSequence,
        g2_sequnce: FundamentalSequence,
        g1: StarlikeCurve,
        g2: StarlikeCurve,
        coeffs: FundamentalSequenceCoefs,
        f1_func: Callable[[Point2D | Point3D], np.float64],
        f2_func: Callable[[Point2D | Point3D], np.float64],
        mfs_data: MFSData) -> np.ndarray:
    """
    Form the vector for the method of fundamental solutions
    """
    # TODO: optimize this function
    M = g1_sequnce.M
    tn = mfs_data.tn[n]
    F = np.empty((2*M, 1), dtype=np.float64)
    F[:] = np.nan
    for i in range(1, M+1):
        right_sum = 0
        for m in range(0, n): # m in [0,...,n-1]
            for j in range(1, M+1):
                phi_index = n-m
                phi1_g1 = g1_sequnce[phi_index]
                right_sum += coeffs.alpha[m, j-1]*phi1_g1[i-1, j-1]
        F[i-1] = f1_func(g1[i-1], tn) - right_sum
        
        right_sum = 0
        for m in range(0, n): # m in [0,...,n-1]
            for j in range(1, M+1):
                phi_index = n-m
                phi2_g2 = g2_sequnce[phi_index]
                right_sum += coeffs.alpha[m, j-1]*phi2_g2[i-1, j-1]
        F[M+i-1] = f2_func(g2[i-1], tn) - right_sum
    return F

def matform_fs_vector_2d(
        n: int,
        g1_sequnce: FundamentalSequence,
        g2_sequnce: FundamentalSequence,
        g1: StarlikeCurve,
        g2: StarlikeCurve,
        coeffs: FundamentalSequenceCoefs,
        f1_func: Callable[[Point2D | Point3D], np.float64],
        f2_func: Callable[[Point2D | Point3D], np.float64],
        mfs_data: MFSData) -> np.ndarray:
    """
    Form the vector for the method of fundamental solutions
    """
    # TODO: optimize this function
    M = g1_sequnce.M
    tn = mfs_data.tn[n]
    F = np.empty((2*M, 1), dtype=np.float64)
    F[:M] = f1_func(g1.points_array, tn).reshape(-1,1)
    F[M:] = f2_func(g2.points_array, tn).reshape(-1,1)
    for m in range(0, n): # m in [0,...,n-1]
        F[:M] -= np.dot(g1_sequnce[n-m], coeffs.alpha[m].reshape(-1,1))
        F[M:] -= np.dot(g2_sequnce[n-m], coeffs.alpha[m].reshape(-1,1))

    return F

def form_fs_vector_3d(
        n: int,
        g1_sequnce: FundamentalSequence,
        g2_sequnce: FundamentalSequence,
        g1: StarlikeCurve,
        g2: StarlikeCurve,
        coeffs: FundamentalSequenceCoefs,
        f1_func: Callable[[Point2D | Point3D], np.float64],
        f2_func: Callable[[Point2D | Point3D], np.float64],
        mfs_data: MFSData) -> np.ndarray:
    """
    Form the vector for the method of fundamental solutions
    """
    # TODO: optimize this function
    M = g1_sequnce.M
    tn = mfs_data.tn[n]
    F = np.empty((2*M, 1), dtype=np.float64)
    F[:] = np.nan
    f1_right_sum = np.zeros((M, 1), dtype=np.float64)
    f2_right_sum = np.zeros((M, 1), dtype=np.float64)
    for i in range(1, M+1):
        right_sum = 0
        for m in range(0, n): # m in [0,...,n-1]
            for j in range(1, M+1):
                phi_index = n-m
                phi1_g1 = g1_sequnce[phi_index]
                right_sum += coeffs.alpha[m, j-1]*phi1_g1[i-1, j-1]
        F[i-1] = f1_func(g1[i-1], tn) - right_sum
        f1_right_sum[i-1] = right_sum
        
        right_sum = 0
        for m in range(0, n): # m in [0,...,n-1]
            for j in range(1, M+1):
                phi_index = n-m
                phi2_g2 = g2_sequnce[phi_index]
                right_sum += coeffs.alpha[m, j-1]*phi2_g2[i-1, j-1]
        f2_right_sum[i-1] = right_sum
        F[M+i-1] = f2_func(g2[i-1], tn) - right_sum
    return F


def matform_fs_vector_3d(
        n: int,
        g1_sequnce: FundamentalSequence,
        g2_sequnce: FundamentalSequence,
        g1: StarlikeSurface,
        g2: StarlikeSurface,
        coeffs: FundamentalSequenceCoefs,
        f1_func: Callable[[Point2D | Point3D], np.float64],
        f2_func: Callable[[Point2D | Point3D], np.float64],
        mfs_data: MFSData) -> np.ndarray:
    """
    Form the vector for the method of fundamental solutions
    """
    # TODO: optimize this function
    M = g1_sequnce.M
    tn = mfs_data.tn[n]
    F = np.empty((2*M, 1), dtype=np.float64)
    F[:M] = f1_func(g1.mesh, tn).reshape(-1,1)
    F[M:] = f2_func(g2.mesh, tn).reshape(-1,1)
    for m in range(0, n): # m in [0,...,n-1]
        F[:M] -= np.dot(g1_sequnce[n-m], coeffs.alpha[m].reshape(-1,1))
        F[M:] -= np.dot(g2_sequnce[n-m], coeffs.alpha[m].reshape(-1,1))
    # f1_right_sum = np.zeros((M, 1), dtype=np.float64)
    # f2_right_sum = np.zeros((M, 1), dtype=np.float64)
    # for m in range(0, n): # m in [0,...,n-1]
    #     f1_right_sum += np.dot(g1_sequnce[n-m], coeffs.alpha[m].reshape(-1,1))
    #     f2_right_sum += np.dot(g2_sequnce[n-m], coeffs.alpha[m].reshape(-1,1))
    # F[:M] -= f1_right_sum
    # F[M:] -= f2_right_sum
    return F



def u_2d(x: Point2D | Point3D, n: int, source_points: SourcePoints2D, fs_coefs: FundamentalSequenceCoefs, mfs_data: MFSData ):
    """
     u(x,t_n) = u_n,M(x), x \in D
     n = 0,...,N - time point
    """
    u = 0.0
    for m in range(0, n+1): # m = 0,...,n
        alpha_m = fs_coefs[m]
        for j in range(0, mfs_data.M): # j =1,...,M
            u+= alpha_m[j]*fs_2d(n-m, x, source_points[j], mfs_data.nu, mfs_data.polynomials)
    return u

def dbu_2d(x: Point2D, nx: Point2D, n: int, source_points: SourcePoints2D, fs_coefs: FundamentalSequenceCoefs, mfs_data: MFSData ):
    """
     Normal derivative by the boundary
     du(x,t_n) = du_n,M(x), x \in D
     n = 0,...,N - time point
    """
    u = 0.0
    for m in range(0, n+1): # m = 0,...,n
        alpha_m = fs_coefs[m]
        for j in range(0, mfs_data.M): # j =1,...,M
            u+= alpha_m[j]*dfs_2d(n-m, x, nx, source_points[j], mfs_data.nu, mfs_data.polynomials)
    return u

def matdbu_2d(X: Point2D, NX: Point2D, n: int, source_points: SourcePoints2D, fs_coefs: FundamentalSequenceCoefs, mfs_data: MFSData):
    """
     Normal derivative by the boundary
     du(x,t_n) = du_n,M(x), x \in D
     n = 0,...,N - time point
    """
    Xnew = X.reshape(2, -1)
    Ynew = source_points.points().reshape(2, -1)
    NXnew = NX.reshape(2, -1)

    XX, YY = zip(*[np.meshgrid(Xnew[i], Ynew[i], indexing='ij') for i in range(2)])
    XX = np.array(XX)
    YY = np.array(YY)

    NXY = np.broadcast_to(NXnew[:, :, np.newaxis], (2, mfs_data.M, mfs_data.M)).transpose(0, 1, 2)

    U = np.zeros(((mfs_data.M, 1)), dtype=np.float64)
    for m in range(0, n+1): # m = 0,...,n
        U += np.dot(matdfs_2d(n-m, XX, NXY, YY, mfs_data.nu, mfs_data.polynomials), fs_coefs[m].reshape(-1,1))

    return U


def u_3d(x: Point3D, n: int, source_points: SourcePoints3D, fs_coefs: FundamentalSequenceCoefs, mfs_data: MFSData):
    """
     u(x,t_n) = u_n,M(x), x \in D
     n = 0,...,N - time point
    """
    u = 0.0
    for m in range(0, n+1): # m = 0,...,n
        alpha_m = fs_coefs[m]
        for j in range(0, mfs_data.M): # j =1,...,M
            u+= alpha_m[j]*fs_3d(n-m, x, source_points[j], mfs_data.nu, mfs_data.polynomials)
    return u

def dbu_3d(x: Point2D, nx: Point2D, n: int, source_points: SourcePoints2D, fs_coefs: FundamentalSequenceCoefs, mfs_data: MFSData ):
    """
     Normal derivative by the boundary
     du(x,t_n) = du_n,M(x), x \in D
     n = 0,...,N - time point
    """
    u = 0.0
    for m in range(0, n+1): # m = 0,...,n
        alpha_m = fs_coefs[m]
        for j in range(0, mfs_data.M): # j =1,...,M
            # print(f'm= {m}, j={j}, alpha_m[j]= {alpha_m[j]}, dfs_3d= {dfs_3d(n-m, x, nx, source_points[j], mfs_data.nu, mfs_data.polynomials)}')
            # print(alpha_m[j]*dfs_3d(n-m, x, nx, source_points[j], mfs_data.nu, mfs_data.polynomials))
            u+= alpha_m[j]*dfs_3d(n-m, x, nx, source_points[j], mfs_data.nu, mfs_data.polynomials)
    return u
# def dbu_3d(x: Point2D, nx: Point2D, n: int, source_points: SourcePoints2D, fs_coefs: FundamentalSequenceCoefs, mfs_data: MFSData ):
#     """
#      Normal derivative by the boundary
#      du(x,t_n) = du_n,M(x), x \in D
#      n = 0,...,N - time point
#     """
#     u = 0.0
#     for m in range(0, n+1): # m = 0,...,n
#         alpha_m = fs_coefs[m]
#         for j in range(0, mfs_data.M): # j =1,...,M
#             u+= alpha_m[j]*dfs_3d(n-m, x, nx, source_points[j], mfs_data.nu, mfs_data.polynomials)
#     return u

def matu_3d(X, n: int, source_points: SourcePoints3D, fs_coefs: FundamentalSequenceCoefs, mfs_data: MFSData):
    """
     u(x,t_n) = u_n,M(x), x \in D
     n = 0,...,N - time point
    """
    Xnew = X.reshape(3, -1)
    Ynew = source_points.mesh().reshape(3, -1)

    XX, YY = zip(*[np.meshgrid(Xnew[i], Ynew[i], indexing='ij') for i in range(3)])
    XX = np.array(XX)
    YY = np.array(YY)
    U = np.zeros(((mfs_data.M, 1)), dtype=np.float64)
    for m in range(0, n+1): # m = 0,...,n
        U += np.dot(matfs_3d(n-m, XX, YY, mfs_data.nu, mfs_data.polynomials), fs_coefs[m].reshape(-1,1))
    return U


def matdbu_3d(X, NX, n: int, source_points: SourcePoints2D, fs_coefs: FundamentalSequenceCoefs, mfs_data: MFSData ):
    """
     Normal derivative by the boundary
     du(x,t_n) = du_n,M(x), x \in D
     n = 0,...,N - time point
    """
    Xnew = X.reshape(3, -1)
    Ynew = source_points.mesh().reshape(3, -1)
    NXnew = NX.reshape(3, -1)

    XX, YY = zip(*[np.meshgrid(Xnew[i], Ynew[i], indexing='ij') for i in range(3)])
    XX = np.array(XX)
    YY = np.array(YY)

    NXY = np.broadcast_to(NXnew[:, :, np.newaxis], (3, mfs_data.M, mfs_data.M)).transpose(0, 1, 2)

    U = np.zeros(((mfs_data.M, 1)), dtype=np.float64)
    for m in range(0, n+1): # m = 0,...,n
        U += np.dot(matdfs_3d(n-m, XX, NXY, YY, mfs_data.nu, mfs_data.polynomials), fs_coefs[m].reshape(-1,1))

    return U

def matdbu_3d_all(X, NX, n: int, source_points: SourcePoints2D, fs_coefs: FundamentalSequenceCoefs, mfs_data: MFSData ):
    """
     Normal derivative by the boundary
     du(x,t_n) = du_n,M(x), x \in D
     n = 0,...,N - time point
    """
    Xnew = X.reshape(3, -1)
    Ynew = source_points.mesh().reshape(3, -1)
    NXnew = NX.reshape(3, -1)

    XX, YY = zip(*[np.meshgrid(Xnew[i], Ynew[i], indexing='ij') for i in range(3)])
    XX = np.array(XX)
    YY = np.array(YY)

    NXY = np.broadcast_to(NXnew[:, :, np.newaxis], (3, mfs_data.M, mfs_data.M)).transpose(0, 1, 2)

    fundamental_solutions = np.zeros((n+1, mfs_data.M, mfs_data.M), dtype=np.float64)
    ## Calculate fundamental solutions up to n
    for m in range(0, n+1):
        fundamental_solutions[n-m] =matdfs_3d(n-m, XX, NXY, YY, mfs_data.nu, mfs_data.polynomials)
    
    U = np.zeros(((n+1, mfs_data.M, 1)), dtype=np.float64)
    for nn in range(0, n+1):
        for m in range(0, nn+1):
            U[nn] += np.dot(fundamental_solutions[nn-m], fs_coefs[m].reshape(-1,1))

    return U


    # U = np.zeros(((n+1, mfs_data.M, 1)), dtype=np.float64)
    # U[0] = np.dot(fundamental_solutions[0], fs_coefs[0].reshape(-1,1))
    # for m in range(1, n+1): # m = 0,...,n
    #     U[m] = U[m-1] + np.dot(fundamental_solutions[n-m], fs_coefs[m].reshape(-1,1))
    


    # for ni in range(0, n+1):
    #     for m in range(0, n+1):
    #         fundamental_solutions[m] = np.dot(fundamental_solutions[m], fs_coefs[ni].reshape(-1,1))

    # U = np.zeros(((n+1, mfs_data.M, 1)), dtype=np.float64) 

    # for m in range(0, n+1): # m = 0,...,n
    #     U += np.dot(matdfs_3d(n-m, XX, NXY, YY, mfs_data.nu, mfs_data.polynomials), fs_coefs[m].reshape(-1,1))

    # return U






# def matdbu_3d(X, NX, n: int, source_points: SourcePoints2D, fs_coefs: FundamentalSequenceCoefs, mfs_data: MFSData ):
#     """
#      Normal derivative by the boundary
#      du(x,t_n) = du_n,M(x), x \in D
#      n = 0,...,N - time point
#     """
#     Xnew = X.reshape(3, -1)
#     Ynew = source_points.mesh().reshape(3, -1)
#     NXnew = NX.reshape(3, -1)

#     XX, YY = zip(*[np.meshgrid(Xnew[i], Ynew[i], indexing='ij') for i in range(3)])
#     XX = np.array(XX)
#     YY = np.array(YY)
#     U = np.zeros(((mfs_data.M, 1)), dtype=np.float64)
#     for m in range(0, n+1): # m = 0,...,n
#         U += np.dot(matdfs_3d(n-m, XX, NXnew, YY, mfs_data.nu, mfs_data.polynomials), fs_coefs[m].reshape(-1,1))
#     return U

#     u = 0.0
#     for m in range(0, n+1): # m = 0,...,n
#         alpha_m = fs_coefs[m]
#         for j in range(0, mfs_data.M): # j =1,...,M
#             u+= alpha_m[j]*dfs_3d(n-m, x, nx, source_points[j], mfs_data.nu, mfs_data.polynomials)
#     return u




# def u_3d(x: Point2D | Point3D, n: int, source_points: SourcePoints3D, fs_coefs: FundamentalSequenceCoefs, poly_3d: MFSPolinomials3D, mfs_data: MFSData):
#     """
#      u(x,t_n) = u_n,M(x), x \in D
#      n = 0,...,N - time point
#     """
#     ### ALARM changeme
    
#     return u

# ### Solver
# def form_boundary_data(M, r1_func, r2_func, eta1, eta2) -> BoundaryData:
#     coll_2d = collocation_points_2d(M, startpoint=False)
#     point_circle = starlike_circle_base(coll_2d)

#     Gamma1 = starlike_curve(r1_func(coll_2d.theta), point_circle)
#     Gamma2 = starlike_curve(r2_func(coll_2d.theta), point_circle)

#     source_coll_2d = collocation_points_2d(M//2, startpoint=False)
#     source_point_circle = starlike_circle_base(source_coll_2d)
#     Gamma1_source = starlike_curve(r1_func(source_coll_2d.theta), source_point_circle)
#     Gamma2_source = starlike_curve(r1_func(source_coll_2d.theta), source_point_circle)
#     source_points = source_points_2d(eta1, eta2, Gamma1_source, Gamma2_source)

#     return coll_2d, source_coll_2d, Gamma1, Gamma2, eta1, eta2, source_points


def precalculate_mfs_data(config: MFSConfig) -> MFSData:
    tn = np.array([(n+1)*(config.T/(config.N+1)) for n in range(0, config.N+1)])
    h = config.T/(config.N+1)
    nu = np.sqrt(2/h)
    # betas = np.empty(config.N+1)
    # betas[0] = np.nan # Ensure that beta_0 is not used
    # betas[1::2] = 4/h
    # betas[2::2] = -4/h
    betta_array = []
    for n in range(0, config.N+1):
        sign = (-1)**n
        betta_array.append(sign*(4/h))
    betta_array[0] = np.nan
    if config.dim == Dimension.TWO_D:
        polynomials = calculate_2d_polinomials(config.N, nu, betta_array)
        coll = collocation_points_2d(config.n_coll, startpoint=False)
        source_coll = collocation_points_2d(config.n_source//2, startpoint=False)
    elif config.dim == Dimension.THREE_D:
        polynomials = calculate_3d_polinomials(config.N, nu, betta_array)
        coll = collocation_points_3d(config.n_coll_theta, config.n_coll_phi)
        source_coll = collocation_points_3d(config.n_source_theta, config.n_source_phi)

    return MFSData(
        config=config,
        h=h,
        tn=tn,
        Beta=betta_array,
        nu=nu,
        collocation=coll,
        source_collocation=source_coll,
        polynomials=polynomials
        # f1_values=f1_vals,
        # f2_values=f2_vals
    )
# def precalculate_mfs_data(config: MFSConfig) -> MFSData:
#     tn = np.array([(n+1)*(config.T/(config.N+1)) for n in range(0, config.N+1)])
#     h = config.T/(config.N+1)
#     nu = np.sqrt(2/h)
#     # betas = np.empty(config.N+1)
#     # betas[0] = np.nan # Ensure that beta_0 is not used
#     # betas[1::2] = 4/h
#     # betas[2::2] = -4/h
#     betta_array = []
#     for n in range(0, config.N+1):
#         sign = (-1)**n
#         betta_array.append(sign*(4/h))
#     betta_array[0] = np.nan
#     if config.dim == Dimension.TWO_D:
#         polynomials_2d = calculate_2d_polinomials(config.N, nu, betta_array)
#         coll_2d = collocation_points_2d(config.n_coll, startpoint=False)
#         source_coll_2d = collocation_points_2d(config.n_source//2, startpoint=False)
#     elif config.dim == Dimension.THREE_D:
#         pass
#     # f1_vals = np.empty((config.n_coll, config.N+1))
#     # f2_vals = np.empty((config.n_coll, config.N+1))
#     # for n, t in enumerate(tn):
#     #     for i, theta in enumerate(coll_2d.theta):
#     #         f1_vals[i, n] = config.f1(theta, t)
#     #         f2_vals[i, n] = config.f2(theta, t)

#     return MFSData(
#         config=config,
#         h=h,
#         tn=tn,
#         Beta=betta_array,
#         nu=nu,
#         collocation=coll_2d,
#         source_collocation=source_coll_2d,
#         polynomials=polynomials_2d
#         # f1_values=f1_vals,
#         # f2_values=f2_vals
#     )
