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
        # if j < self.M//2:
        #     return np.array([self.gart1.x[j], self.gart1.y[j]])
        # else:
        #     return np.array([self.gart2.x[j - self.M//2], self.gart2.y[j - self.M//2]])
        if j < self.M//2:
            return np.array([self.gart2.x[j], self.gart2.y[j]])
        else:
            return np.array([self.gart1.x[j - self.M//2], self.gart1.y[j - self.M//2]])
    def as_boundary(self) -> StarlikeCurve:
        '''
            Return the combined boundary
        '''
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
    # g1 = StarlikeCurve(curve1.collocation, eta1*curve1.x, eta1*curve1.y)
    # g2 = StarlikeCurve(curve2.collocation, eta2*curve2.x, eta2*curve2.y)
    return SourcePoints2D(M, eta1, eta2, curve1*eta1, curve2*eta2)
