"""
    Polinomial for the method of fundamental solutions
"""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.polynomial import polyval
from numpy.polynomial import Polynomial
from collections.abc import Callable

@dataclass
class MFSPolinomials3D:
    """
    A is matrix of polinomials for MFS 3D
    /  1  a01 NaN  .  .  .  NaN  \\ 
    |  1  a11 a12  .  .  .  NaN  | 
    |  .  .   .    .  .  .  .    | 
    |  .  .   .    .  .  .  .    | 
    |  .  .   .    .  .  .  .    | 
    \\  1  an1 an2  .  .  .  ann  / 
    """
    A: np.ndarray
    polynomials: np.ndarray[Polynomial]


@dataclass
class MFSPolinomials2D:
    """
    A is matrix of polinomials for MFS 2D
    /  1  a01 NaN  .  .  .  NaN  \\ 
    |  1  a11 a12  .  .  .  NaN  | 
    |  .  .   .    .  .  .  .    | 
    |  .  .   .    .  .  .  .    | 
    |  .  .   .    .  .  .  .    | 
    \\  1  an1 an2  .  .  .  ann  / 
    """
    A: np.ndarray
    v_polynomials: np.ndarray[Polynomial]
    w_polynomials: np.ndarray[Polynomial]

def calculate_polinomials_coefs(coefs_func: Callable[[int, int, np.ndarray], np.float64], N: int, nu: float, betas: np.ndarray) -> MFSPolinomials3D:
    # a_nm, n=0,1,...N, m =0,1,...n
    A = np.empty((N+1,N+1))

    # Fill nan to be sure we don't use indexes we don't need
    A[:] = np.nan

    ## Fill first column
    A[:,0]=1

    ## Fill diagonal elements
    for n in range(1,N+1):
        A[n,n] = -(betas[1]*A[n-1,n-1])/(2*nu*n)

    for n in range(N+1):
        for m in range(n-1, 0, -1):
            A[n,m] = coefs_func(n,m,A)
    return A

def calculate_3d_polinomials(N: int, nu:float, betas: np.array) -> MFSPolinomials3D:
    def polinomial_coeff_3d(n: int,k:int, A: np.ndarray) -> np.float64:
        res = k*(k+1)*A[n,k+1]
        for m in range(k-1, n): # m = k-1;n-1
            res -=betas[n-m]*A[m, k-1] # TODO: test beta
        res /=(2*nu*k)
        return res
    A = calculate_polinomials_coefs(polinomial_coeff_3d, N, nu, betas)
    polinomials = np.array([Polynomial(A[n,:n+1]) for n in range(N+1)], dtype=Polynomial)
    return MFSPolinomials3D(A, polinomials)

def calculate_2d_polinomials(N: int, nu: float, betas: np.ndarray) -> MFSPolinomials2D:
    def polinomial_coeff_2d(n: int,k:int, A: np.ndarray) -> np.float64:
        res = (4*((k+1)//2)**2)*A[n,k+1]
        for m in range(k-1, n): # m = k-1;n-1
            res -=betas[n-m]*A[m, k-1] # TODO: test beta
        res /=(2*nu*k)
        return res

    A = calculate_polinomials_coefs(polinomial_coeff_2d, N, nu, betas)
    v_polynomials = [Polynomial([1])] # v_0(r)=1
    w_polynomials = [Polynomial([0])] # w_0(r)=0

    for n in range(1,N+1):
        v_n_coeff = np.zeros(n+1)
        for m in range(0, n//2+1):
            v_n_coeff[2*m] = A[n, 2*m]
        v_polynomials.append(Polynomial(v_n_coeff))

        w_n_coeff = np.zeros(n+1)
        for m in range(0, (n-1)//2+1):
            w_n_coeff[2*m+1] = A[n, 2*m+1]
        w_polynomials.append(Polynomial(w_n_coeff))

    return MFSPolinomials2D(A, v_polynomials, w_polynomials)

