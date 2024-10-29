"""
The distance module contains functions to calculate the distance between two points.
"""
import numpy as np
from .boundary import StarlikeCurve, StarlikeSurface


def pointwise_distance(starlike1: StarlikeCurve | StarlikeSurface, starlike2: StarlikeCurve | StarlikeSurface) -> np.array:
    """
    Calculate the distance between two points
    """
    raise NotImplementedError("This function is not implemented yet")
    if isinstance(starlike1, StarlikeCurve):
        return np.linalg.norm([
            (starlike1.x - starlike2.x),
            (starlike1.y - starlike2.y)],
            axis=0)

    if isinstance(starlike1, StarlikeSurface):
        return np.linalg.norm([
            *(starlike1.x - starlike2.x),
            *(starlike1.y - starlike2.y),
            *(starlike1.z - starlike2.z)],
            axis=0)

    raise ValueError("Invalid argumets types. Should be StarlikeCurve or StarlikeSurface")

