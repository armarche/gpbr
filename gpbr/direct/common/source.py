"""
Source points for the MFS
"""
from dataclasses import dataclass
import numpy as np
from .boundary import StarlikeCurve, StarlikeSurface

@dataclass
class SourcePoints2D:
    M: int # Even number of source points
    eta1: np.float64
    eta2: np.float64
    gart1: StarlikeCurve # Gamma1 artificial boundary
    gart2: StarlikeCurve # Gamma2 artificial boundary
    def __getitem__(self, j:int):
        '''
            Return the j-th source point
        '''
        return self.gart2[j]  if j < self.M//2 else self.gart1[j - self.M//2]
    def get_point(self, j: int) -> np.ndarray:
        '''
            DEPRECATED
            Return the j-th source point
        '''
        if j < self.M//2:
            return self.gart2.points[j]
        else:
            return self.gart1.points[j - self.M//2]

    def as_boundary(self) -> StarlikeCurve:
        '''
            DEPRECATED
            Return the combined boundary
        '''
        raise NotImplementedError("This method is deprecated")
        return StarlikeCurve(self.gart1.collocation, [*self.gart1.points, *self.gart2.points])
        # return StarlikeCurve(self.gart1.collocation, np.concatenate((self.gart1.x, self.gart2.x)), np.concatenate((self.gart1.y, self.gart2.y)))
        

def source_points_2d(
        eta1: np.float64,
        eta2: np.float64,
        curve1: StarlikeCurve,
        curve2: StarlikeCurve) -> SourcePoints2D:
    '''
        Generate the source points for the MFS in 2D
        Note that we calculate source points in the same collocation points
    '''
    M = curve1.collocation.n + curve2.collocation.n
    if curve1.drf is None:
        cpcurve1 = StarlikeCurve.from_radial(curve1.collocation, lambda s: eta1*curve1.rf(s))
    else:
        cpcurve1 = StarlikeCurve.from_radial_with_derivative(curve1.collocation, lambda s: eta1*curve1.rf(s), lambda s: eta1*curve1.drf(s))

    if curve2.drf is None:
        cpcurve2 = StarlikeCurve.from_radial(curve2.collocation, lambda s: eta2*curve2.rf(s))
    else:
        cpcurve2 = StarlikeCurve.from_radial_with_derivative(curve2.collocation, lambda s: eta2*curve2.rf(s), lambda s: eta2*curve2.drf(s))
    
    return SourcePoints2D(M, eta1, eta2, cpcurve1, cpcurve2)
##################

@dataclass
class SourcePoints3D:
    M: int # Even number of source points
    eta1: np.float64
    eta2: np.float64
    gart1: StarlikeSurface # Gamma1 artificial boundary
    gart2: StarlikeSurface # Gamma2 artificial boundary
    def __getitem__(self, j:int):
        '''
            Return the j-th source point
        '''
        # M = 2*m1*m2
        return self.gart2[j]  if j < self.M//2 else self.gart1[j - self.M//2]

    def as_boundary(self) -> StarlikeSurface:
        '''
            Return the combined boundary
        '''
        return StarlikeSurface(self.gart1.collocation, [*self.gart1.points, *self.gart2.points])
        # return StarlikeCurve(self.gart1.collocation, np.concatenate((self.gart1.x, self.gart2.x)), np.concatenate((self.gart1.y, self.gart2.y)))
        

def source_points_3d(
        eta1: np.float64,
        eta2: np.float64,
        surface1: StarlikeSurface,
        surface2: StarlikeSurface) -> SourcePoints3D:
    '''
        Generate the source points for the MFS in 3D
        Note that we calculate source points in the same collocation points
    '''
    # Assume that the number of points in each surface is the same
    M = 2*(surface1.collocation.n_phi*surface1.collocation.n_theta)
    # g1 = StarlikeCurve(curve1.collocation, eta1*curve1.x, eta1*curve1.y)
    # g2 = StarlikeCurve(curve2.collocation, eta2*curve2.x, eta2*curve2.y)
    return SourcePoints3D(M, eta1, eta2, surface1*eta1, surface2*eta2)
