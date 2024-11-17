import numpy as np
from dataclasses import dataclass
from .collocation import CollocationData2D, CollocationData3D


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float
    def __mul__(self, num: float):
        return Point2D(self.x*num, self.y*num)
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
    def __sub__(self, point):
        return Point3D(self.x-point.x, self.y-point.y, self.z-point.z)

@dataclass
class StarlikeCurve:
    collocation: CollocationData2D
    points: list[Point2D]
    def __getitem__(self, index:int):
        return self.points[index]
    def raw_points(self):
        """Unpacks the list of 2D points into separate lists of x and y coordinates.

        Returns:
            tuple: A tuple of two lists, (x, y).
        """
        return np.array([point.x for point in self.points]), \
               np.array([point.y for point in self.points])
    @staticmethod
    def from_raw_points(collocation: CollocationData2D, x: np.array, y: np.array):
        if np.shape(x) != np.shape(y):
            raise ValueError('Shape of x and y arrays missmatch!')
        return StarlikeCurve(collocation, [Point2D(a,b) for a,b in zip(x,y)])
    def __mul__(self, nums):
        if isinstance(nums, np.ndarray):
            return StarlikeCurve(self.collocation, [r*p for r,p in zip(nums, self.points)])
        return StarlikeCurve(self.collocation, [p*nums for p in self.points])
        raise ValueError('Unsupported value type')
    def __rmul__(self, nums):
        # if isinstance(nums, float | np.float64):
        #     return StarlikeCurve(self.collocation, [p*nums for p in self.points])
        if isinstance(nums, np.ndarray):
            return StarlikeCurve(self.collocation, [r*p for r,p in zip(nums, self.points)])
        return StarlikeCurve(self.collocation, [p*nums for p in self.points])
        raise ValueError('Unsupported value type')


@dataclass
class StarlikeSurface:
    collocation: CollocationData3D
    points: list[Point3D]
    def __getitem__(self, index:int):
        return self.points[index]
    def raw_points(self):
        """Unpacks the list of 3D points into separate lists of x and y coordinates.

        Returns:
            tuple: A tuple of three lists, (x, y, z).
        """
        return np.array([point.x for point in self.points]), \
               np.array([point.y for point in self.points]), \
               np.array([point.z for point in self.points])
    @staticmethod
    def from_raw_points(collocation: CollocationData3D, x: np.array, y: np.array, z: np.array):
        if not np.shape(x) == np.shape(y) == np.shape(z):
            raise ValueError('Shape of x, y, and z arrays missmatch!')
        return StarlikeCurve(collocation, [Point3D(a,b,c) for a,b,c in zip(x,y,z)])

## 2D
def starlike_circle_base(collocation: CollocationData2D) -> StarlikeCurve:
    '''
        Generate a circle of radius one
    '''
    x = np.cos(collocation.theta)
    y = np.sin(collocation.theta)
    return StarlikeCurve.from_raw_points(collocation, x, y)


def starlike_curve(r_values: np.array, base: StarlikeCurve) -> StarlikeCurve:
    '''
        Generate a starlike curve from a circle base
    '''
    return base*r_values

## 3D Mesh
def starlike_sphere_base(collocation: CollocationData3D) -> StarlikeSurface:
    '''
        Generate a sphere mesh of radius one
    '''
    x = (np.sin(collocation.theta_grid) * np.cos(collocation.phi_grid)).ravel()
    y = (np.sin(collocation.theta_grid) * np.sin(collocation.phi_grid)).ravel()
    z = np.cos(collocation.theta_grid).ravel()
    return StarlikeSurface.from_raw_points(collocation, x, y, z)

def starlike_surface(r_grid: np.ndarray, collocation: CollocationData3D) -> StarlikeSurface:
    '''
        Generate a sphere mesh of radius one
    '''
    x = (r_grid * np.sin(collocation.theta_grid) * np.cos(collocation.phi_grid)).ravel()
    y = (r_grid * np.sin(collocation.theta_grid) * np.sin(collocation.phi_grid)).ravel()
    z = (r_grid * np.cos(collocation.theta_grid)).ravel()
    return StarlikeSurface.from_raw_points(collocation, x, y, z)





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




