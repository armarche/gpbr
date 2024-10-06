"""
    Polinomial for the method of fundamental solutions
"""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.polynomial import polyval
from numpy.polynomial import Polynomial

## TODO: this should be moved
@dataclass
class MfSData: # Will be filled during implementing the problem
    Beta: np.array # Array of beta coeficient.
    nu: np.float64 # constant. nu = beta_0



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
    polinomials: np.ndarray[Polynomial]


def calculate_3d_polinomials(mfs_data: MfSData, N: int) -> MFSPolinomials3D:
    def polinomial_coeff(n: int,k:int, A: np.ndarray) -> np.float64:
        res = k*(k+1)*A[n,k+1]
        for m in range(k-1, n): # m = k-1;n-1
            res -=mfs_data.Beta[n-m]*A[m, k-1] # TODO: test beta
        res /=(2*mfs_data.nu*k)
        return res

    # a_nm, n=0,1,...N, m =0,1,...n
    A = np.empty((N+1,N+1))

    # Fill nan to be sure we don't use indexes we don't need
    A[:] = np.nan

    ## Fill first column
    A[:,0]=1

    ## Fill diagonal elements
    for n in range(1,N+1):
        A[n,n] = -(mfs_data.Beta[1]*A[n-1,n-1])/(2*mfs_data.nu*n)

    polinomials = []
    for n in range(N+1):
        for m in range(n-1, 0, -1):
            A[n,m] = polinomial_coeff(n,m,A)
        polinomials.append(Polynomial(A[n,:n+1]))
    return MFSPolinomials3D(A, np.array(polinomials, dtype=Polynomial))
