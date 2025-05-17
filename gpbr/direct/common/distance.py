"""
The distance module contains functions to calculate the distance between two points.
"""
import numpy as np
from .boundary import Point2D, Point3D, StarlikeCurve, StarlikeSurface

def point_distance(p1: Point2D | Point3D, p2: Point2D | Point3D | None = None) -> float:
    if isinstance(p1, Point2D):
        p2 = p2 if p2 is not None else Point2D(0,0)
        return np.linalg.norm([p1.x-p2.x, p1.y-p2.y])
    if isinstance(p1, Point3D):
        p2 = p2 if p2 is not None else Point3D(0,0,0)
        return np.linalg.norm([p1.x-p2.x, p1.y-p2.y, p1.z-p2.z])

def matpoint_distance(X: np.ndarray, Y: np.ndarray) -> float:
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")
    return np.sqrt(np.sum((X - Y) ** 2, axis=0))


def boundary_pointwise_distance(starlike1: StarlikeCurve | StarlikeSurface, starlike2: StarlikeCurve | StarlikeSurface) -> np.array:
    """
    Calculate the distance between two points
    """
    if isinstance(starlike1, StarlikeCurve) and isinstance(starlike2, StarlikeCurve): ## TODO: check if this is correct
        diff = starlike1.points_np - starlike2.points_np
        return np.linalg.norm(diff, axis=0)

    if isinstance(starlike1, StarlikeSurface) and isinstance(starlike2, StarlikeSurface): ## TODO: check if this is correct
        diff = starlike1.mesh_np - starlike2.mesh_np
        return np.linalg.norm(diff, axis=0)

    # if isinstance(starlike1, StarlikeCurve):
    #     x1, y1 = starlike1.raw_points()
    #     x2, y2 = starlike2.raw_points()
    #     return np.linalg.norm([
    #         (x1 - x2),
    #         (y1 - y2)],
    #         axis=0)

    # if isinstance(starlike1, StarlikeSurface):
    #     x1, y1, z1 = starlike1.raw_points()
    #     x2, y2, z2 = starlike2.raw_points()
    #     return np.linalg.norm([
    #         *(x1 - x2),
    #         *(y1 - y2),
    #         *(z1 - z2)],
    #         axis=0)

    raise ValueError("Invalid argumets types. Should be StarlikeCurve or StarlikeSurface")

