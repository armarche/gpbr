import numpy as np
from dataclasses import dataclass
from .collocation import CollocationData2D, CollocationData3D


@dataclass
class StarlikeCurve:
    collocation: CollocationData2D
    x: np.array
    y: np.array
    # mesh: np.ndarray # Looks like we don't need this

@dataclass
class StarlikeSurface:
    collocation: CollocationData3D
    x: np.array
    y: np.array
    z: np.array

## 2D
def starlike_circle_base(collocation: CollocationData2D) -> StarlikeCurve:
    '''
        Generate a circle of radius one
    '''
    x = np.cos(collocation.theta)
    y = np.sin(collocation.theta)
    # mesh = np.array([x, y]).T ## Do we need to transpose this?
    return StarlikeCurve(collocation, x, y)


def starlike_curve(r_values: np.array, base: StarlikeCurve) -> StarlikeCurve:
    '''
        Generate a starlike curve from a circle base
    '''
    x = r_values * base.x
    y = r_values * base.y
    return StarlikeCurve(base.collocation, x, y)

## 3D Mesh
def starlike_sphere_base(collocation: CollocationData3D) -> StarlikeSurface:
    '''
        Generate a sphere mesh of radius one
    '''
    x = np.outer(np.sin(collocation.theta), np.cos(collocation.phi))
    y = np.outer(np.sin(collocation.theta), np.sin(collocation.phi))
    z = np.outer(np.cos(collocation.theta), np.ones(np.size(collocation.phi)))

    return StarlikeSurface(collocation, x, y, z)

def starlike_surface(r_mesh: np.ndarray, base: StarlikeSurface) -> StarlikeSurface:
    '''
        Generate a starlike curve from a circle base
    '''
    x = r_mesh * base.x
    y = r_mesh * base.y
    z = r_mesh * base.z
    return StarlikeSurface(base.collocation, x, y, z)




# ## 3D
# def starlike_sphere_base(collocation: CollocationData3D) -> StarlikeSurface:
#     '''
#         Generate a sphere of radius one
#     '''
#     x = np.sin(collocation.theta)*np.cos(collocation.phi)
#     y = np.sin(collocation.theta)*np.sin(collocation.phi)
#     z = np.cos(collocation.theta)
#     return StarlikeSurface(collocation, x, y, z)

# def starlike_surface(r_values: np.array, base: StarlikeSurface) -> StarlikeSurface:
#     '''
#         Generate a starlike surface from a sphere base
#     '''
#     x = r_values * base.x
#     y = r_values * base.y
#     z = r_values * base.z
#     return StarlikeSurface(base.collocation, x, y, z)


# ## 3D Mesh
# def starlike_sphere_mesh_base(collocation: CollocationData3D) -> StarlikeSurface:
#     '''
#         Generate a sphere mesh of radius one
#     '''
#     x = np.outer(np.sin(collocation.theta), np.cos(collocation.phi))
#     y = np.outer(np.sin(collocation.theta), np.sin(collocation.phi))
#     z = np.outer(np.cos(collocation.theta), np.ones(np.size(collocation.phi)))

#     return StarlikeSurface(collocation, x, y, z)

# def starlike_surface_mesh(r_mesh: np.ndarray, base: StarlikeSurface) -> StarlikeSurface:
#     '''
#         Generate a starlike curve from a circle base
#     '''
#     x = r_mesh * base.x
#     y = r_mesh * base.y
#     z = r_mesh * base.z
#     return StarlikeSurface(base.collocation, x, y, z)




