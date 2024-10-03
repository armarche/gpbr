import numpy as np


def starlike_circle(points: np.array) -> np.array:    
    '''
    Calculate Starlike circle values
    '''
    return np.array([np.cos(points), np.sin(points)],dtype=float)

def starlike_curve(r_values: np.array, circle_points: np.ndarray) -> np.array:
    '''
    Calculate Starlike curve values
    '''
    return r_values*circle_points
