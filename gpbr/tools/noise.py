import numpy as np

def noise(vals, noise_level, seed=None):
    """
    Add static noise to the input values with a specified noise level.

    Parameters:
    -----------
    vals : numpy.ndarray
        Input array to which noise will be added.
    noise_level : float, optional
        The noise level, determining the magnitude of the random noise.
    seed : int, optional
        Seed for the random number generator to produce static noise.

    Returns:
    --------
    numpy.ndarray
        The input array with added noise.
    """
    if seed is not None:
        # Use a temporary random seed
        state = np.random.get_state()  # Save the current random state
        np.random.seed(seed)          # Set the seed
        random_noise = noise_level * np.random.uniform(low=-1.0,high=1.0,size=vals.shape)
        np.random.set_state(state)    # Restore the original random state
    else:
        # Generate noise without setting a seed
        random_noise = noise_level * np.random.uniform(low=-1.0,high=1.0,size=vals.shape)

    return vals * (1 + random_noise)