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
    def get_point(self, j: int) -> np.ndarray:
        '''
            Return the j-th source point
        '''
        # if j < self.M//2:
        #     return np.array([self.gart1.x[j], self.gart1.y[j]])
        # else:
        #     return np.array([self.gart2.x[j - self.M//2], self.gart2.y[j - self.M//2]])
        if j < self.M//2:
            return np.array([self.gart2.x[j], self.gart2.y[j]])
        else:
            return np.array([self.gart1.x[j - self.M//2], self.gart1.y[j - self.M//2]])
        

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
    g1 = StarlikeCurve(curve1.collocation, eta1*curve1.x, eta1*curve1.y)
    g2 = StarlikeCurve(curve2.collocation, eta2*curve2.x, eta2*curve2.y)
    return SourcePoints2D(M, eta1, eta2, g1, g2)
