import numpy as np

def is_2pi_periodic(values, tolerance=1e-5):
    if isinstance(values, (np.integer, int, float)):
        return True
    return np.allclose(values[0], values[-1], atol=tolerance)