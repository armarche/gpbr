"""
Common data for the method of fundamental solutions for the heat equation problem
"""
from enum import Enum
from typing import Callable
import numpy as np
from dataclasses import dataclass

from ..common.boundary import Point2D, Point3D

from ..common.collocation import CollocationData2D, CollocationData3D
from .polynomial import MFSPolinomials2D, MFSPolinomials3D ## Potentialy circular dependency


class Dimension(Enum):
    TWO_D = 2
    THREE_D = 3

@dataclass(frozen=True)
class MFSConfig:
    N: np.int32 # Number of time points
    T: np.float64 # Time of the simulation
    n_coll: np.int64 # Number of collocation points
    n_source: np.int64 # Number of source points
    f1: Callable[[Point2D | Point3D], np.float64] # Data on Inner boundary
    f2: Callable[[Point2D | Point3D], np.float64] # Data on Outer boundary
    eta1: np.float64 # MFS Coeefficient for the inner boundary
    eta2: np.float64 # MFS Coeefficient for the outer boundary
    dim: Dimension = Dimension.TWO_D # Dimension of the problem
    ## 3D specific parameters
    n_coll_theta: np.int64 = None
    n_coll_phi: np.int64 = None
    n_source_theta: np.int64 = None
    n_source_phi: np.int64 = None

@dataclass(frozen=True)
class MFSData: # Will be filled during implementing the problem
    """
    Common data for the method of fundamental solutions
    """
    config: MFSConfig
    h: np.float64 # Time step
    tn: np.array # Time steps
    Beta: np.array # Array of beta coeficient.
    nu: np.float64 # constant. nu = beta_0
    collocation: CollocationData2D | CollocationData3D
    source_collocation: CollocationData2D | CollocationData3D
    polynomials: MFSPolinomials2D | MFSPolinomials3D
    # f1_values: np.ndarray # Pre-calculated values of f1
    # f2_values: np.ndarray # Pre-calculated values of f2

    @property
    def N(self):
        return self.config.N
    
    @property
    def T(self):
        return self.config.T
    
    @property
    def M(self):
        return self.config.n_coll
    
    @property
    def eta1(self):
        return self.config.eta1
    
    @property
    def eta2(self):
        return self.config.eta2
    
    @property
    def f1(self):
        return self.config.f1
    
    @property
    def f2(self):
        return self.config.f2
    

