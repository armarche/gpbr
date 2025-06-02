import numpy as np

def protected_div(numerator, denominator):
    """
    Protected division function to avoid division by zero errors.
    
    Parameters:
    -----------
    numerator : float or numpy.ndarray
        The numerator in the division
    denominator : float or numpy.ndarray
        The denominator in the division
        
    Returns:
    --------
    result : float or numpy.ndarray
        The division result, with 1.0 returned where denominator == 0
    """
    # Handle scalar values
    if isinstance(denominator, (int, float)):
        return numerator / denominator if abs(denominator) > 1e-10 else 1.0
        
    # Handle numpy arrays
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
        if isinstance(result, np.ndarray):
            result[np.isinf(result)] = 1.0
            result[np.isnan(result)] = 1.0
            return result
        elif np.isinf(result) or np.isnan(result):
            return 1.0
        return result
    
def pow2(x):
    return np.pow(x,2)

def pow3(x):
    return np.pow(x,3)

def pow4(x):
    return np.pow(x,4)

def sqrtabs(x):
    return np.sqrt(np.abs(x))

def expplusone(x):
    return np.exp(np.ones_like(x, dtype=np.float64) + x)