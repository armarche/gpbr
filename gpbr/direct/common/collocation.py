'''
    Collocation points for the MFS
'''
from dataclasses import dataclass

import numpy as np

@dataclass
class CollocationData2D:
    n: int
    theta: np.ndarray

@dataclass
class CollocationData3D:
    n_theta: int
    n_phi: int
    theta: np.array
    phi: np.array
    theta_grid: np.ndarray
    phi_grid: np.ndarray


def collocation_points_2d(n_theta: int, startpoint: False) -> CollocationData2D:
    '''
        Generate the collocation points for the MFS in 2D
    '''
    theta = np.linspace(2*np.pi, 0, n_theta, endpoint=False)[::-1]
    if startpoint:
        theta = np.insert(theta, 0, 0)
        n_theta += 1
    return CollocationData2D(n_theta, theta)

def collocation_points_3d(n_theta: int, n_phi: int) -> CollocationData3D:
    '''
        Generate the collocation points for the MFS in 3D
    '''
    ## TODO: Rewrite to pure numpy?
    theta = np.array([(np.pi*i)/(n_theta+1) for i in range(1, n_theta+1)], dtype=np.float64) ## TODO: Check why we need to add 1 in (np.pi*i)/(n_theta+1)
    phi = np.array([(2*np.pi*i)/n_phi for i in range(1, n_phi+1)], dtype=np.float64)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    return CollocationData3D(n_theta, n_phi, theta, phi, theta_grid, phi_grid)

def linspace_points_3d(n_theta: int, n_phi: int) -> CollocationData3D:
    '''
        Generate the collocation points for the MFS in 3D
    '''
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    return CollocationData3D(n_theta, n_phi, theta, phi, theta_grid, phi_grid)
