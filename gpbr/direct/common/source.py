"""
Source points for the MFS
"""
from dataclasses import dataclass

import numpy as np

from .boundary import Point2D, Point3D, StarlikeCurve, StarlikeSurface

@dataclass
class SourcePoints2D:
    M: int # Even number of source points
    eta1: np.float64
    eta2: np.float64
    gart1: StarlikeCurve # Gamma1 artificial boundary
    gart2: StarlikeCurve # Gamma2 artificial boundary

    def __getitem__(self, j:int) -> Point2D:
        '''
            Return the j-th source point
        '''
        if j < self.M//2:
            return self.eta2*self.gart2[j]
        return self.eta1*self.gart1[j - self.M//2]

    def points(self) -> np.ndarray:
        '''
            Return the mesh of source points
        '''
        return np.concatenate(
            (
                self.eta2*self.gart2.points_np.reshape(2,-1),
                self.eta1*self.gart1.points_np.reshape(2,-1)
            ), axis=1)

@dataclass
class SourcePoints3D:
    M: int # Even number of source points
    eta1: np.float64
    eta2: np.float64
    gart1: StarlikeSurface # Gamma1 artificial boundary
    gart2: StarlikeSurface # Gamma2 artificial boundary
    def __getitem__(self, j:int) -> Point3D:
        '''
            Return the j-th source point
        '''
        if j < self.M//2:
            return self.eta2*self.gart2[j]
        return self.eta1*self.gart1[j - self.M//2]

    def mesh(self) -> np.ndarray:
        '''
            Return the mesh of source points
        '''
        return np.concatenate(
            (
                self.eta2*self.gart2.mesh_np.reshape(3,-1),
                self.eta1*self.gart1.mesh_np.reshape(3,-1)
            ), axis=1)

