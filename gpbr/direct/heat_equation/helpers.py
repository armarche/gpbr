"""
MFS helpers
"""

from collections.abc import Callable

from ..common.distance import point_distance

from ..common.source import SourcePoints2D, SourcePoints3D
from .polynomial import MFSPolinomials2D, MFSPolinomials3D

from ..common.boundary import Point2D, Point3D, StarlikeCurve
from .common import MFSData
from .fundamental_sequence import FundamentalSequence, FundamentalSequenceCoefs, fs_2d, fs_3d
import numpy as np

from .polynomial import MFSPolinomials3D

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

# def form_fs_vector_3d(
#         n: int,
#         g1_sequnce: FundamentalSequence,
#         g2_sequnce: FundamentalSequence,
#         g1: StarlikeCurve,
#         g2: StarlikeCurve,
#         coeffs: FundamentalSequenceCoefs,
#         f1_func: Callable[[Point2D | Point3D], np.float64],
#         f2_func: Callable[[Point2D | Point3D], np.float64],
#         mfs_data: MFSData) -> np.ndarray:
#     """
#     Form the vector for the method of fundamental solutions
#     """
#     # TODO: optimize this function
#     M = g1_sequnce.M
#     tn = mfs_data.tn[n]
#     F = np.empty((2*M, 1), dtype=np.float64)
#     F[:] = np.nan
#     for i in range(1, M+1):
#         right_sum = 0
#         for m in range(0, n): # m in [0,...,n-1]
#             for j in range(1, M+1):
#                 phi_index = n-m
#                 phi1_g1 = g1_sequnce[phi_index]
#                 right_sum += coeffs.alpha[m, j-1]*phi1_g1[i-1, j-1]
#         F[i-1] = f1_func(g1[i-1], tn) - right_sum
        
#         right_sum = 0
#         for m in range(0, n): # m in [0,...,n-1]
#             for j in range(1, M+1):
#                 phi_index = n-m
#                 phi2_g2 = g2_sequnce[phi_index]
#                 right_sum += coeffs.alpha[m, j-1]*phi2_g2[i-1, j-1]
#         F[M+i-1] = f2_func(g2[i-1], tn) - right_sum
#     return F

def form_fs_vector_3d(
        n: int,
        g1_sequnce: FundamentalSequence,
        g2_sequnce: FundamentalSequence,
        g1: StarlikeCurve,
        g2: StarlikeCurve,
        source_points: SourcePoints3D,
        coeffs: FundamentalSequenceCoefs,
        f1_func: Callable[[Point2D | Point3D], np.float64],
        f2_func: Callable[[Point2D | Point3D], np.float64],
        mfs_data: MFSData,
        polinomials: MFSPolinomials3D) -> np.ndarray:
    """
    Form the vector for the method of fundamental solutions
    """
    # TODO: optimize this function
    F1 = np.empty((mfs_data.M,1), dtype=np.float64)
    F2 = np.empty((mfs_data.M,1), dtype=np.float64)
    F1[:] = np.nan
    F2[:] = np.nan

    time_point = mfs_data.tn[n]
    for i in range(0, mfs_data.M): # i = 1,...,M
        x_i = g1[i]
        res = 0
        for m in range(0, n): # m = 0,...,n-1
            alphas = coeffs.alpha[m]
            for j in range(0, mfs_data.M): # j = 1,...,M
                ### WARNING
                res += alphas[j] * fs_3d(n-m, point_distance(x_i,source_points[j]),mfs_data.nu, polinomials)
        F1[i]= f1_func(x_i, time_point) - res

    for i in range(0, mfs_data.M): # i = 1,...,M
        x_i = g2[i]
        res = 0
        for m in range(0, n): # m = 0,...,n-1
            alphas = coeffs.alpha[m]
            for j in range(0, mfs_data.M): # j = 1,...,M
                ### WARNING
                res += alphas[j] * fs_3d(n-m, point_distance(x_i,source_points[j]),mfs_data.nu, polinomials)
        F2[i]= f2_func(x_i, time_point) - res
    F =np.concatenate((F1,F2))
    return F
    # return np.concatenate((F1,F2))


def u_2d(x: Point2D | Point3D, n: int, source_points: SourcePoints2D, fs_coefs: FundamentalSequenceCoefs, poly_2d: MFSPolinomials2D, mfs_data: MFSData ):
    """
     u(x,t_n) = u_n,M(x), x \in D
     n = 0,...,N - time point
    """
    u = 0.0
    alpha_n = fs_coefs[n]
    for m in range(0, n+1): # m = 0,...,n
        print(f"m = {m}")
        for j in range(0, mfs_data.M): # j =1,...,M
            delta = point_distance(x, source_points[j])
            u+= alpha_n[j]*fs_2d(n-m, delta, mfs_data.nu, poly_2d)
    
    return u

def u_3d(x: Point2D | Point3D, n: int, source_points: SourcePoints3D, fs_coefs: FundamentalSequenceCoefs, poly_3d: MFSPolinomials3D, mfs_data: MFSData):
    """
     u(x,t_n) = u_n,M(x), x \in D
     n = 0,...,N - time point
    """
    u = 0.0
    alpha_n = fs_coefs[n]
    for m in range(0, n+1): # m = 0,...,n
        print(f"m = {m}")
        for j in range(0, mfs_data.M): # j =1,...,M
            delta = point_distance(x, source_points[j])
            u+= alpha_n[j]*fs_3d(n-m, delta, mfs_data.nu, poly_3d)
    
    return u

#     for m in range(0, mfs_data.N+1): # m = 0,...,N
#         alpha_n = fs_coefs[m]
#         for j in range(0, mfs_data.M): # j =1,...,M
#             delta = point_distance(x, source_points[j])
#             u+= alpha_n[j]*fs_2d(m, delta, mfs_data.nu, poly_2d)

#     return u


# def du_2d(x: Point2D | Point3D, source_points: SourcePoints2D, fs_coefs: FundamentalSequenceCoefs, poly_2d: MFSPolinomials2D, mfs_data: MFSData ):
#     u = 0.0
#     for m in range(0, mfs_data.N+1): # m = 0,...,N
#         alpha_n = fs_coefs[m]
#         for j in range(0, mfs_data.M): # j =1,...,M
#             delta = point_distance(x, source_points[j])
#             u+= alpha_n[j]*fs_2d(m, delta, mfs_data.nu, poly_2d)

#     return u







# def form_fs_vector(
#         g1_sequnce: FundamentalSequence,
#         g2_sequnce: FundamentalSequence,
#         g1: StarlikeCurve,
#         g2: StarlikeCurve,
#         coeffs: FundamentalSequenceCoefs,
#         f1_func,
#         f2_func,
#         mfs_data: MfSData) -> np.ndarray:
#     """
#     Form the vector for the method of fundamental solutions
#     """
    
#     F = np.empty((2*g1_sequnce.M, 1), dtype=np.float64)
#     F[:] = np.nan
#     M = mfs_data.M
#     tn = mfs_data.tn
#     for n in range(0, mfs_data.N+1):
#         for i in range(1, M+1):
#             right_sum = 0
#             for m in range(0, n): # m in [0,...,n-1]
#                 for j in range(1, M+1):
#                     phi_index = n-m
#                     phi_g1 = g1_sequnce.get(phi_index)
#                     right_sum += coeffs.alpha[m, j-1]*phi_g1[i-1, j-1]
#  ]           F[i-1] = f1_func([g1.x[i-1], g1.y[i-1]], tn[n]) - right_sum

#             right_sum = 0
#             for m in range(0, n): # m in [0,...,n-1]
#                 for j in range(1, M+1):
#                     phi_index = n-m
#                     phi_g2 = g2_sequnce.get(phi_index)
#                     right_sum += coeffs.alpha[m, j-1]*phi_g2[i-1, j-1]
#             F[M+i-1] = f2_func([g2.x[i-1], g2.y[i-1]], tn[n]) - right_sum

#     return F

