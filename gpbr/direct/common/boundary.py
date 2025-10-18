from typing import Callable
from dataclasses import dataclass

import numpy as np

from .collocation import CollocationData2D, CollocationData3D

@dataclass(frozen=True)
class Point2D:
    x: float
    y: float
    def __mul__(self, num: float):
        return Point2D(self.x*num, self.y*num)
    def __add__(self, point):
        return Point2D(self.x+point.x, self.y+point.y)
    def __rmul__(self, num: float):
        return Point2D(self.x*num, self.y*num)
    def __sub__(self, point):
        return Point2D(self.x-point.x, self.y-point.y)

@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float
    def __mul__(self, num: float):
        return Point3D(self.x*num, self.y*num, self.z*num)
    def __rmul__(self, num: float):
        return Point3D(self.x*num, self.y*num, self.z*num)
    def __add__(self, point):
        return Point3D(self.x+point.x, self.y+point.y, self.z+point.z)
    def __sub__(self, point):
        return Point3D(self.x-point.x, self.y-point.y, self.z-point.z)



@dataclass(frozen=True)
class Curve:
    collocation: CollocationData2D
    point_list: list[Point2D]
    normal_list: list[Point2D] | None
    points_np: np.ndarray
    normals_np: np.ndarray | None


@dataclass(frozen=True)
class ParametricCurve:
    xf: Callable[[float], float]
    yf: Callable[[float], float]
    dxf: Callable[[float], float]
    dyf: Callable[[float], float]
    collocation: CollocationData2D
    point_list: list[Point2D]
    normal_list: list[Point2D] | None
    points_np: np.ndarray
    normals_np: np.ndarray | None
    @staticmethod
    def from_parametric(collocation: CollocationData2D, f1: Callable[[float], float], f2: Callable[[float], float]):
        xx = f1(collocation.theta)
        yy = f2(collocation.theta)
        return ParametricCurve(
            xf=f1,
            yf=f2,
            dxf=None,
            dyf=None,
            collocation=collocation,
            point_list=[Point2D(x, y) for x, y in zip(xx.ravel(), yy.ravel())],
            normal_list=None,
            points_np=np.array([xx, yy]),
            normals_np=None
        )

    @staticmethod
    def from_parametric_with_derivatives(
        collocation: CollocationData2D,
        f1: Callable[[float], float],
        df1: Callable[[float], float],
        f2: Callable[[float], float],
        df2: Callable[[float], float]
        ):
        xx, yy = f1(collocation.theta), f2(collocation.theta)

        tx, ty = df1(collocation.theta), df2(collocation.theta)

        nx = ty  # Outward normal
        ny = -tx

        norms = np.sqrt(nx**2 + ny**2)
        nx_normalized = nx / norms
        ny_normalized = ny / norms

        return ParametricCurve(
            xf=f1,
            yf=f2,
            dxf=df1,
            dyf=df2,
            collocation=collocation,
            point_list=np.array([Point2D(x, y) for x, y in zip(xx, yy)]),
            normal_list=np.array([Point2D(nx, ny) for nx, ny in zip(nx_normalized, ny_normalized)]),
            points_np=np.array([xx, yy]),
            normals_np=np.array([nx_normalized, ny_normalized])
        )
    def __call__(self,s: np.ndarray):
        return self.xf(s), self.yf(s)
@dataclass(frozen=True)
class StarlikeCurve:
    rf: Callable[[float], float]
    drf: Callable[[float], float]
    collocation: CollocationData2D
    point_list: list[Point2D]
    normal_list: list[Point2D] | None
    points_np: np.ndarray
    normals_np: np.ndarray | None
    @staticmethod
    def from_radial(collocation: CollocationData2D, rf: Callable[[float], float]):
        rvals  = rf(collocation.theta) # We use numpy funtions, so we can do that way
        xx, yy = rvals * np.cos(collocation.theta), rvals * np.sin(collocation.theta)
        return StarlikeCurve(
            rf=rf,
            drf=None,
            collocation=collocation,
            point_list=[Point2D(x,y) for x,y in zip(xx.ravel(), yy.ravel())],
            normal_list=None,
            points_np=np.array([xx, yy]),
            normals_np=None
        )

    @staticmethod
    def from_radial_with_derivative(collocation: CollocationData2D, rf: Callable[[float], float], drf: Callable[[float], float]):
        # Calculate all points at once using vectorized operations
        r_vals = rf(collocation.theta)
        xx, yy = r_vals * np.cos(collocation.theta), r_vals * np.sin(collocation.theta)

        # Calculate tangents (vectorized)
        dr_vals = drf(collocation.theta)
        tx = -np.sin(collocation.theta) * r_vals + np.cos(collocation.theta) * dr_vals
        ty = np.cos(collocation.theta) * r_vals + np.sin(collocation.theta) * dr_vals

        # Calculate normals (perpendicular to tangent)
        nx = ty  # Outward normal
        ny = -tx

        # Normalize
        norms = np.sqrt(nx**2 + ny**2)
        nx_normalized = nx / norms
        ny_normalized = ny / norms

        return StarlikeCurve(
            rf=rf,
            drf=drf,
            collocation=collocation,
            point_list=np.array([Point2D(x, y) for x, y in zip(xx, yy)]),
            normal_list=np.array([Point2D(nx, ny) for nx, ny in zip(nx_normalized, ny_normalized)]),
            points_np=np.array([xx, yy]),
            normals_np=np.array([nx_normalized, ny_normalized])
        )
    
    @staticmethod
    def from_radial_values(collocation: CollocationData2D, rf: Callable[[float], float], radial_values: np.array):
        return StarlikeCurve(rf, None, collocation,[Point2D(x,y) for x,y in zip(np.cos(collocation.theta),np.sin(collocation.theta))] , None)

    def __getitem__(self, index:int):
        return self.point_list[index]
    
    # def __call__(self, s):
    #     return Point2D(np.cos(s), np.sin(s))*self.rf(s)
    def __call__(self, s: np.ndarray):
        rvals = self.rf(s)
        return np.cos(s) * rvals, np.sin(s) * rvals

    def normal(self, s):
        raise DeprecationWarning("This function is no longer used")
        """Calculate the normal vector at a given parameter s.

        Args:
            s (float): The parameter at which to calculate the normal vector.

        Returns:
            Point2D: The normal vector at the given parameter.
        """
        tangent = Point2D(-np.sin(s), np.cos(s)) * self.rf(s) + Point2D(np.cos(s), np.sin(s)) * self.drf(s)
        normal = Point2D(tangent.y, -tangent.x) ## Outward normal
        # norm = np.sqrt(normal.x**2 + normal.y**2)
        norm = np.sqrt(self.rf(s)**2 + self.drf(s)**2)
        return Point2D(normal.x / norm, normal.y / norm)    

    def raw_points(self):
        raise DeprecationWarning("This function is no longer used")
        """Unpacks the list of 2D points into separate lists of x and y coordinates.

        Returns:
            tuple: A tuple of two lists, (x, y).
        """
        return np.array([point.x for point in self.point_list]), \
               np.array([point.y for point in self.point_list])

@dataclass(frozen=True)
class StarlikeSurface:
    rf: Callable[[float, float], float]
    drf_phi: Callable[[float, float], float]
    drf_theta: Callable[[float, float], float]
    collocation: CollocationData3D
    point_list: list[Point3D]
    normal_list: list[Point3D] | None
    mesh_np: np.ndarray | None 
    normals_mesh_np: np.ndarray | None

    @staticmethod
    def from_radial(collocation: CollocationData3D, rf: Callable[[float, float], float]):
        rvals  = rf(collocation.theta_grid, collocation.phi_grid) # We use numpy funtions, so we can do that way
        xx = rvals * np.sin(collocation.theta_grid) * np.cos(collocation.phi_grid)
        yy = rvals * np.sin(collocation.theta_grid) * np.sin(collocation.phi_grid)
        zz = rvals * np.cos(collocation.theta_grid)
        return StarlikeSurface(
            rf=rf,
            drf_phi=None,
            drf_theta=None,
            collocation=collocation,
            point_list=[Point3D(x,y,z) for x,y,z in zip(xx.ravel(), yy.ravel(), zz.ravel())],
            normal_list=None,
            mesh_np=np.array([xx, yy, zz]),
            normals_mesh_np=None
        )
    
    @staticmethod
    def from_radial_with_derivative(collocation: CollocationData3D, rf: Callable[[float, float], float], drf_phi: Callable[[float, float], float], drf_theta: Callable[[float, float], float]):
        # Compute the surface points
        theta_grid = collocation.theta_grid
        phi_grid = collocation.phi_grid
        r_vals = rf(theta_grid, phi_grid)
        xx = r_vals * np.sin(theta_grid) * np.cos(phi_grid)
        yy = r_vals * np.sin(theta_grid) * np.sin(phi_grid)
        zz = r_vals * np.cos(theta_grid)

        # Compute the partial derivatives of the surface
        # Partial derivative with respect to theta
        dr_theta = drf_theta(theta_grid, phi_grid)
        x_theta = dr_theta * np.sin(theta_grid) * np.cos(phi_grid) + \
                r_vals * np.cos(theta_grid) * np.cos(phi_grid)
        y_theta = dr_theta * np.sin(theta_grid) * np.sin(phi_grid) + \
                r_vals * np.cos(theta_grid) * np.sin(phi_grid)
        z_theta = dr_theta * np.cos(theta_grid) - \
                r_vals * np.sin(theta_grid)

        dr_phi = drf_phi(theta_grid, phi_grid)
        # Partial derivative with respect to phi
        x_phi = dr_phi * np.sin(theta_grid) * np.cos(phi_grid) - \
                r_vals * np.sin(theta_grid) * np.sin(phi_grid)
        y_phi = dr_phi * np.sin(theta_grid) * np.sin(phi_grid) + \
                r_vals * np.sin(theta_grid) * np.cos(phi_grid)
        z_phi = dr_phi * np.cos(theta_grid)

        # Compute the cross product of the partial derivatives
        nx = y_theta * z_phi - z_theta * y_phi
        ny = z_theta * x_phi - x_theta * z_phi
        nz = x_theta * y_phi - y_theta * x_phi

        # Normalize the normal vectors
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        nx /= norm
        ny /= norm
        nz /= norm

        return StarlikeSurface(
            rf=rf,
            drf_phi=drf_phi,
            drf_theta=drf_theta,
            collocation=collocation,
            point_list=[Point3D(x,y,z) for x,y,z in zip(xx.ravel(), yy.ravel(), zz.ravel())],
            normal_list=[Point3D(x,y,z) for x,y,z in zip(nx.ravel(), ny.ravel(), nz.ravel())],
            mesh_np= np.array([xx, yy, zz]),
            normals_mesh_np=np.array([nx, ny, nz])
        )

    def __getitem__(self, index:int):
        return self.point_list[index]
    
    def __call__(self, theta, phi):
        return Point3D(
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        )*self.rf(theta, phi)

    def normal(self, theta, phi):
        raise DeprecationWarning("This function is no longer used")
        r = self.rf(theta, phi)
        dr_theta = self.drf_theta(theta, phi)
        dr_phi = self.drf_phi(theta, phi)

        # Compute the partial derivatives of the surface
        tangent_theta = Point3D(
            dr_theta * np.sin(theta) * np.cos(phi) + r * np.cos(theta) * np.cos(phi),
            dr_theta * np.sin(theta) * np.sin(phi) + r * np.cos(theta) * np.sin(phi),
            dr_theta * np.cos(theta) - r * np.sin(theta)
        )
        tangent_phi = Point3D(
            dr_phi * np.sin(theta) * np.cos(phi) - r * np.sin(theta) * np.sin(phi),
            dr_phi * np.sin(theta) * np.sin(phi) + r * np.sin(theta) * np.cos(phi),
            dr_phi * np.cos(theta)
        )

        # Compute the cross product of the partial derivatives
        normal = Point3D(
            tangent_theta.y * tangent_phi.z - tangent_theta.z * tangent_phi.y,
            tangent_theta.z * tangent_phi.x - tangent_theta.x * tangent_phi.z,
            tangent_theta.x * tangent_phi.y - tangent_theta.y * tangent_phi.x
        )

        # Normalize the normal vector
        norm = np.sqrt(normal.x**2 + normal.y**2 + normal.z**2)
        return Point3D(normal.x / norm, normal.y / norm, normal.z / norm)
    def raw_points(self):
        raise DeprecationWarning("This function is no longer used")
        """Unpacks the list of 3D points into separate lists of x and y coordinates.

        Returns:
            tuple: A tuple of two lists, (x, y, z).
        """
        return np.array([point.x for point in self.point_list]), \
               np.array([point.y for point in self.point_list]), \
               np.array([point.z for point in self.point_list])

## 2D
def starlike_circle_base(collocation: CollocationData2D) -> StarlikeCurve:
    '''
        Generate a circle of radius one
    '''
    raise DeprecationWarning("This function is no longer used")
    x = np.cos(collocation.theta)
    y = np.sin(collocation.theta)
    return StarlikeCurve.from_raw_points(collocation, x, y)


def starlike_curve(r_values: np.array, base: StarlikeCurve) -> StarlikeCurve:
    '''
        Generate a starlike curve from a circle base
    '''
    raise DeprecationWarning("This function is no longer used")
    return base*r_values ## Note the order. If np.array go first, it will use itself overrided method

## 3D Mesh
def starlike_sphere_base(collocation: CollocationData3D) -> StarlikeSurface:
    '''
        Generate a sphere mesh of radius one
    '''
    raise DeprecationWarning("This function is no longer used")
    x = (np.sin(collocation.theta_grid) * np.cos(collocation.phi_grid)).ravel()
    y = (np.sin(collocation.theta_grid) * np.sin(collocation.phi_grid)).ravel()
    z = np.cos(collocation.theta_grid).ravel()
    return StarlikeSurface.from_raw_points(collocation, x, y, z)

def starlike_surface(r_grid: np.ndarray, collocation: CollocationData3D) -> StarlikeSurface:
    '''
        Generate a stralike surface from a sphere base
    '''
    raise DeprecationWarning("This function is no longer used")
    x = (r_grid * np.sin(collocation.theta_grid) * np.cos(collocation.phi_grid)).ravel()
    y = (r_grid * np.sin(collocation.theta_grid) * np.sin(collocation.phi_grid)).ravel()
    z = (r_grid * np.cos(collocation.theta_grid)).ravel()
    return StarlikeSurface.from_raw_points(collocation, x, y, z)

