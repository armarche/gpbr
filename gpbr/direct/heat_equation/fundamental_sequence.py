"""
Fundamental sequence for the elliptic equations
"""
from dataclasses import dataclass
import numpy as np
from numpy import linalg
# from scipy.special import kn
from scipy.special import k0, k1, kn


from ..common.boundary import Point2D, Point3D, StarlikeCurve, StarlikeSurface
from ..common.source import SourcePoints2D, SourcePoints3D
from ..common.distance import point_distance

from .common import MFSData
from .polynomial import MFSPolinomials3D, MFSPolinomials2D

@dataclass
class FundamentalSequence:
    """
    Fundamental sequence for the elliptic equations
    Note: assume that number of collocation points is the same as the number of source points
    """
    M: int
    phis: np.ndarray
    def __getitem__(self, n:int) -> np.ndarray:
        """
        Return the phis mesh in time point n
        """
        return self.phis[n]

@dataclass
class FundamentalSequenceCoefs:
    alpha: np.ndarray
    def __getitem__(self, n:int) -> np.ndarray:
        """
        Return alpha coefficients in time point n
        """
        return self.alpha[n]

def fundamental_sequence_2d(curve: StarlikeCurve, source_points: SourcePoints2D, mfs_data: MFSData) -> FundamentalSequence: #TODO: optimize this function
    '''
        Calculate the fundamental sequence for the 2D problem
        Note: assume that number of collocation points is the same as the number of source points
    '''
    M = curve.collocation.n # number of collocation points
    phis = np.empty((mfs_data.N+1, M, M), dtype=np.float64)
    for i in range(0, M): # i = 1, ..., M
        for j in range(0, M): # j = 1, ..., M
            for n in range(0, mfs_data.N+1): # N+1 time points
                phis[n][i, j] = fs_2d(n, curve[i], source_points[j], mfs_data.nu, mfs_data.polynomials)
    return FundamentalSequence(M, phis)
    
# def fundamental_sequence_3d(surface: StarlikeSurface, source_points: SourcePoints3D, mfs_data: MFSData) -> FundamentalSequence: #TODO: optimize this function
#     '''
#         Calculate the fundamental sequence for the 3D problem
#         Note: assume that number of collocation points is the same as the number of source points
#     '''
#     M = mfs_data.M
#     phis = np.empty((mfs_data.N+1, M, M), dtype=np.float64)
#     for i in range(0, M): # i = 1, ..., M
#         for j in range(0, M): # j = 1, ..., M
#             delta = point_distance(surface[i], source_points[j])
#             for n in range(0, mfs_data.N+1): # N+1 time points
#                 phis[n][i, j] = fs_3d(n, delta, mfs_data.nu, mfs_data.polynomials)
def fundamental_sequence_3d(surface: StarlikeSurface, source_points: SourcePoints3D, mfs_data: MFSData) -> FundamentalSequence: #TODO: optimize this function
    '''
        Calculate the fundamental sequence for the 3D problem
        Note: assume that number of collocation points is the same as the number of source points
    '''
    M = mfs_data.M
    phis = np.empty((mfs_data.N+1, M, M), dtype=np.float64)
    for i in range(0, M): # i = 1, ..., M
        for j in range(0, M): # j = 1, ..., M
            for n in range(0, mfs_data.N+1): # N+1 time points
                phis[n][i, j] = fs_3d(n, surface[i], source_points[j], mfs_data.nu, mfs_data.polynomials)
    return FundamentalSequence(M, phis)
    # phis = np.empty((mfs_data.N+1, M, M), dtype=np.float64)
    # for i in range(0, M): # i = 1, ..., M
    #     for j in range(0, M): # j = 1, ..., M
    #         delta = point_distance(surface[i], source_points[j])
    #         for n in range(0, mfs_data.N+1): # N+1 time points
    #             phis[n][i, j] = fs_3d(n, delta, mfs_data.nu, mfs_data.polynomials)
    
    # return FundamentalSequence(M, phis)

def fs_2d(n: int, x: Point2D, y: Point2D, nu: float, polynomials: MFSPolinomials2D) -> np.float64:
    """
    Fundamental solution for the 2D problem
    """
    xy = point_distance(x, y)
    v_poly = polynomials.v_polynomials[n]
    w_poly = polynomials.w_polynomials[n]
    return k0(nu*xy)*v_poly(xy) + k1(nu*xy)*w_poly(xy)

# def dfs_2d(n: int, x: Point2D, nx: Point2D, y: Point2D, nu: float, polynomials: MFSPolinomials2D) -> np.float64:
#     """
#     Derivative of the fundamental solution for the 2D problem
#     """
#     xy = point_distance(x, y)
#     z = nu * xy

#     k0z = k0(z)
#     k1z = k1(z)
#     k2z = kn(2, z)

#     v = polynomials.v_polynomials[n]
#     dv = v.deriv()

#     w = polynomials.w_polynomials[n]
#     dw = w.deriv()

#     common_term = k1z * dw(xy) + k0z * dv(xy) - v(xy) * k1z * nu - w(xy) * nu * (k0z / 2 + k2z / 2)
#     dphix1 = (common_term * (x.x - y.x)) / xy
#     dphix2 = (common_term * (x.y - y.y)) / xy

#     return dphix1 * nx.x + dphix2 * nx.y

def dfs_2d(n: int, x: Point2D, nx: Point2D, y: Point2D, nu: float, polynomials: MFSPolinomials2D) -> np.float64:
    """
    Derivative of the fundamental solution for the 2D problem
    """
    xy = point_distance(x, y)
    z = nu * xy

    k0z = k0(z)
    k1z = k1(z)
    k2z = kn(2, z)

    v = polynomials.v_polynomials[n]
    dv = v.deriv()

    w = polynomials.w_polynomials[n]
    dw = w.deriv()

    common_term = k1z * dw(xy) + k0z * dv(xy) - v(xy) * k1z * nu - w(xy) * nu * (k0z / 2 + k2z / 2)
    dphix1 = (common_term * (x.x - y.x)) / xy
    dphix2 = (common_term * (x.y - y.y)) / xy

    return dphix1 * nx.x + dphix2 * nx.y


def fs_3d(n: int, arg: np.float64, nu: float, mfs_polynomials: MFSPolinomials3D) -> np.float64:
    """
    Fundamental solution for the 3D problem
    """
    poly = mfs_polynomials.polynomials[n]
    return (np.exp(-nu*arg)*poly(arg))/arg


def fs_3d(n: int, x: Point3D, y: Point3D, nu: float, polynomials: MFSPolinomials3D) -> np.float64:
    """
    Fundamental solution for the 2D problem
    """
    xy = point_distance(x, y)
    v_poly = polynomials.polynomials[n]
    return np.exp(-nu*xy)*v_poly(xy)/xy


def dfs_3d(n: int, x: Point3D, nx: Point3D, y: Point3D, nu: float, polynomials: MFSPolinomials3D) -> np.float64:
    """
    Derivative of the fundamental solution for the 3D problem
    """
    xy = point_distance(x, y)
    z = nu * xy

    v = polynomials.polynomials[n]
    dv = v.deriv()

    common_term = (-nu*v(xy) - v(xy)/xy + dv(xy))

    dphix1 = common_term*(x.x-y.x)
    dphix2 = common_term*(x.y-y.y)
    dphix3 = common_term*(x.z-y.z)

    dphix1 *= np.exp(-z)/(xy**2)
    dphix2 *= np.exp(-z)/(xy**2)
    dphix3 *= np.exp(-z)/(xy**2)

    return dphix1 * nx.x + dphix2 * nx.y + dphix3 * nx.z


