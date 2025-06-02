"""
Common data for the method of fundamental solutions for the heat equation problem
"""
from enum import Enum
from typing import Callable
from dataclasses import dataclass

import numpy as np

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

@dataclass(frozen=True)
class MFSConfig2D(MFSConfig):
    @property
    def dim(self):
        return Dimension.TWO_D

@dataclass(frozen=True)
class MFSConfig3D(MFSConfig):
    n_coll_theta: np.int64
    n_coll_phi: np.int64
    n_source_theta: np.int64
    n_source_phi: np.int64

    @property
    def dim(self):
        return Dimension.THREE_D

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
    

