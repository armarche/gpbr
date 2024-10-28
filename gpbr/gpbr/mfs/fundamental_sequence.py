"""
Fundamental sequence for the elliptic equations
"""
from dataclasses import dataclass
import numpy as np
from numpy import linalg
from scipy.special import kn

from gpbr.gpbr.boundary import StarlikeCurve, StarlikeSurface
from gpbr.gpbr.mfs.data import MfSData
from gpbr.gpbr.source import SourcePoints2D
from .polynomial import MFSPolinomials3D, MFSPolinomials2D

@dataclass
class FundamentalSequence:
    """
    Fundamental sequence for the elliptic equations
    Note: assume that number of collocation points is the same as the number of source points
    """
    M: int
    phis: np.ndarray
    def get(self, n: int) -> np.ndarray:# Phis mesh in time point n
        """
        Return the phis mesh in time point n
        """
        return self.phis[n]
    

@dataclass
class FundamentalSequenceCoefs:
    alpha: np.ndarray
    def get(self, n: int) -> np.array:
        """
        Return alpha coefficients in time point n
        """
        return self.alpha[n]

def fundamental_sequence_2d(curve: StarlikeCurve, source_points: SourcePoints2D, mfs_data: MfSData, mfs_poly: MFSPolinomials2D) -> FundamentalSequence: #TODO: optimize this function
    '''
        Calculate the fundamental sequence for the 2D problem
        Note: assume that number of collocation points is the same as the number of source points
    '''
    M = curve.collocation.n # number of collocation points
    phis = np.empty((mfs_data.N+1, M, M), dtype=np.float64)
    for n in range(0, mfs_data.N+1): # N+1 time points    
        phi_vals = np.empty((M, M), dtype=np.float64)
        phi_vals[:] = np.nan
        for i in range(0, M): # i = 1, ..., M
            for j in range(0, M): # j = 1, ..., M
                x, y = source_points.get_point(j)
                delta = linalg.norm([curve.x[i]-x, curve.y[i]-y])
                phi_vals[i, j] = fs_2d(n, delta, mfs_data.nu, mfs_poly)
        phis[n] = phi_vals
    
    return FundamentalSequence(M, phis)

# def fundamental_sequence_3d(curve: StarlikeSurface, source_points: SourcePoints2D, mfs_data: MfSData, mfs_poly: MFSPolinomials2D) -> FundamentalSequence: #TODO: optimize this function
#     '''
#         Calculate the fundamental sequence for the 2D problem
#         Note: assume that number of collocation points is the same as the number of source points
#     '''
#     M = curve.collocation.n # number of collocation points
#     phis = np.empty((mfs_data.N+1, M, M), dtype=np.float64)
#     for n in range(0, mfs_data.N+1): # N+1 time points    
#         phi_vals = np.empty((M, M), dtype=np.float64)
#         phi_vals[:] = np.nan
#         for i in range(0, M): # i = 1, ..., M
#             for j in range(0, M): # j = 1, ..., M
#                 x, y = source_points.get_point(j)
#                 delta = linalg.norm([curve.x[i]-x, curve.y[i]-y])
#                 phi_vals[i, j] = fs_2d(n, delta, mfs_data.nu, mfs_poly)
#         phis[n] = phi_vals
    
#     return FundamentalSequence(M, phis)


def fs_2d(n: int, arg: np.float64, nu: float, polynomials: MFSPolinomials2D) -> np.float64:
    v_poly = polynomials.v_polynomials[n]
    w_poly = polynomials.w_polynomials[n]
    return kn(0, nu*arg)*v_poly(arg) + kn(1, nu*arg)*w_poly(arg)


def fs_3d(n: int, arg: np.float64, nu: float, mfs_polynomials: MFSPolinomials3D) -> np.float64:
    poly = mfs_polynomials.polynomials[n]
    return (np.exp(-nu*arg)*poly(arg))/arg

# def fs_2d(n: int, x: np.ndarray, y: np.ndarray, nu: float, polynomials: MFSPolinomials2D) -> np.ndarray:
#     # delta = np.abs(x - y)
#     delta = np.norm(x - y)
#     v_poly = polynomials.v_polynomials[n]
#     w_poly = polynomials.w_polynomials[n]
#     return kn(0, nu*delta)*v_poly(delta) + delta*kn(1, nu*delta)*w_poly(delta)


# def fs_3d(n: int, x: np.ndarray, y: np.ndarray, nu: float, mfs_polynomials: MFSPolinomials3D) -> np.ndarray:
#     # delta = np.abs(x - y)
#     delta = np.norm(x - y)
#     poly = mfs_polynomials.polynomials[n]
#     return (np.exp(-nu*delta)*poly(delta))/delta
