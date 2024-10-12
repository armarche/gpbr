"""
Module for the data of the MFS problem.
"""
import numpy as np
from dataclasses import dataclass

## TODO: this should be moved
@dataclass
class MfSData: # Will be filled during implementing the problem
    N: np.int32 # Number of time points-1
    T: np.float64 # Time of the smulation
    tn: np.array # Time steps
    M: np.int64 # Number of collocation points
    Beta: np.array # Array of beta coeficient.
    nu: np.float64 # constant. nu = beta_0


