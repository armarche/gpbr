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


# def calculate_3d_polinomials(mfs_data: MfSData, N: int) -> MFSPolinomials3D: #TODO: Optimize
#     def polinomial_coeffs(n, diagonal_el, mfs_data):
#         coefs = np.empty(n+1)
#         coefs[:] = np.nan
#         coefs[0]=1
#         coefs[n] =diagonal_el
#         for k in range(n-1, 0, -1): # 0 and n elements already filled
#             res = k*(k+1)*coefs[k+1]
#             for m in range(k-1, n): # m = k-1;n-1
#                 res -=mfs_data.Beta[n-m]*coefs[k-1] # TODO: test beta
#             res /=(2*mfs_data.nu*k)
#             coefs[k] = res
#         return coefs
#     diagonals = np.empty(N+1)
#     diagonals[:] = np.nan
#     diagonals[0] = 1

#     for n in range(1,N+1):
#         diagonals[n] = -(mfs_data.Beta[1]*diagonals[n-1])/(2*mfs_data.nu*n)

#     polynomials = []
#     for n in range(N+1):
#         polynomials.append(
#             Polynomial(polinomial_coeffs(n, diagonals[n], mfs_data)))
    
#     return MFSPolinomials3D(np.array(polynomials, dtype=Polynomial))



# @dataclass
# class MFSPolinomial3D:
#     """
#     Store matrix of polinomials for MFS 3D
#     /  1  a01 NaN  .  .  .  NaN  \\ 
#     |  1  a11 a12  .  .  .  NaN  | 
#     |  .  .   .    .  .  .  .    | 
#     |  .  .   .    .  .  .  .    | 
#     |  .  .   .    .  .  .  .    | 
#     \\  1  an1 an2  .  .  .  ann  / 
#     """
#     A: np.ndarray

# def calculate_3d_coeeficients(mfs_data: MfSData, N: int) -> MFSPolinomial3D: #TODO: Optimize
#     def polinomial(n: int,k:int, A: np.ndarray) -> np.float64:
#         res = k*(k+1)*A[n,k+1]
#         for m in range(k-1, n): # m = k-1;n-1
#             res -=mfs_data.Beta[n-m]*A[m, k-1] # TODO: test beta
#         res /=(2*mfs_data.nu*k)
#         return res

#     # a_nm, n=0,1,...N, m =0,1,...n
#     A = np.empty((N+1,N+1))

#     # Fill nan to be sure we don't use indexes we don't need
#     A[:] = np.nan

#     ## Fill first column
#     A[:,0]=1

#     ## Fill diagonal elements
#     for n in range(1,N+1):
#         A[n,n] = -(mfs_data.Beta[1]*A[n-1,n-1])/(2*mfs_data.nu*n)

#     for n in range(0,N+1): 
#         for m in range(n,-1, -1):
#             if m == 0: # Already filled
#                 continue
#             if n == m: # Already filled
#                 continue
#             A[n,m] = polinomial(n,m,A)
#     return MFSPolinomial3D(A)

# def polinomial_3d(ro: np.array, n: int, coeff: MFSPolinomial3D) -> np.float64: # TODO: calculate values of matrixes
#     return sum([coeff.A[n,m]*ro**m for m in range(0,n+1)])

# def dpolinomial_3d(ro: np.float64, n: int, coeff: MFSPolinomial3D) -> np.float64: # TODO: calculate values of matrixes
#     return sum([coeff.A[n,m]*m*ro**(m-1) for m in range(1,n+1)])

# def d2polinomial_3d(ro: np.float64, n: int, coeff: MFSPolinomial3D) -> np.float64: # TODO: calculate values of matrixes
#     return sum([coeff.A[n,m]*m*(m-1)*ro**(m-2) for m in range(2,n+1)])



# def polinomial_3d_arr(ro_arr: np.array, n: int, coeff: MFSPolinomial3D) -> np.array: # TODO: calculate values of matrixes
#     return polyval(ro_arr, coeff.A[n,:n+1], tensor=False)
#     return sum([coeff.A[n,m]*ro**m for m in range(0,n+1)])

# def dpolinomial_3d_arr(ro_arr: np.float64, n: int, coeff: MFSPolinomial3D) -> np.array: # TODO: calculate values of matrixes
#     poly = Polynomial(coeff.A[n,:n+1])
#     poly = np.polynomial.polynomial
#     powers = np.arange(0,n+1)
#     return polyval(ro_arr, powers*coeff.A[n,:n+1], tensor=False)
#     return sum([coeff.A[n,m]*m*ro**(m-1) for m in range(1,n+1)])

# def d2polinomial_3d_arr(ro_arr: np.float64, n: int, coeff: MFSPolinomial3D) -> np.array: # TODO: calculate values of matrixes
#     return sum([coeff.A[n,m]*m*(m-1)*ro**(m-2) for m in range(2,n+1)])

