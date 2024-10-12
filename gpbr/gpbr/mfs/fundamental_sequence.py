"""
Fundamental sequence for the elliptic equations
"""
import numpy as np
from numpy.linalg import norm
from scipy.special import kn
from .polynomial import MFSPolinomials3D, MFSPolinomials2D


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
