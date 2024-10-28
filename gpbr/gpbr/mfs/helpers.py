"""
MFS helpers
"""

from dataclasses import dataclass

from gpbr.gpbr.boundary import StarlikeCurve
from gpbr.gpbr.mfs.data import MfSData
from .fundamental_sequence import FundamentalSequence, FundamentalSequenceCoefs
import numpy as np

def form_fs_matrix(g1_sequnce: FundamentalSequence, g2_sequnce: FundamentalSequence) -> np.ndarray:
    """
    Form the Phi_0 matrix for the method of fundamental solutions
    """
    return np.concatenate((g1_sequnce.get(0), g2_sequnce.get(0)), axis=0)


def form_fs_vector_2d(
        n: int,
        g1_sequnce: FundamentalSequence,
        g2_sequnce: FundamentalSequence,
        g1: StarlikeCurve,
        g2: StarlikeCurve,
        coeffs: FundamentalSequenceCoefs,
        f1_func,
        f2_func,
        mfs_data: MfSData) -> np.ndarray:
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
                phi1_g1  = g1_sequnce.get(phi_index)
                right_sum += coeffs.alpha[m, j-1]*phi1_g1[i-1, j-1]
        F[i-1] = f1_func([g1.x[i-1], g1.y[i-1]], tn) - right_sum
        
        right_sum = 0
        for m in range(0, n): # m in [0,...,n-1]
            for j in range(1, M+1):
                phi_index = n-m
                phi2_g2  = g2_sequnce.get(phi_index)
                right_sum += coeffs.alpha[m, j-1]*phi2_g2[i-1, j-1]
        F[M+i-1] = f2_func([g2.x[i-1], g2.y[i-1]], tn) - right_sum
    return F

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
#             F[i-1] = f1_func([g1.x[i-1], g1.y[i-1]], tn[n]) - right_sum

#             right_sum = 0
#             for m in range(0, n): # m in [0,...,n-1]
#                 for j in range(1, M+1):
#                     phi_index = n-m
#                     phi_g2 = g2_sequnce.get(phi_index)
#                     right_sum += coeffs.alpha[m, j-1]*phi_g2[i-1, j-1]
#             F[M+i-1] = f2_func([g2.x[i-1], g2.y[i-1]], tn[n]) - right_sum

#     return F

