import numpy as np
from numba import njit

@njit
def fast_sliding_kde(x, bandwidth, cutoff=3.0):
    """
    Perform KDE at particle positions using Gaussian kernel with cutoff.
    
    Parameters
    ----------
    sorted_particles : ndarray, shape (N,)
        Sorted array of particle positions.
    bandwidth : float
        Kernel bandwidth (standard deviation of the Gaussian).
    cutoff : float
        Cutoff in terms of bandwidth (e.g. 3 -> ignore points further than 3 * bandwidth).
    
    Returns
    -------
    densities : ndarray, shape (N,)
        Estimated density at each particle position.
    """
    N = x.shape[0]
    densities = np.zeros(N)
    cutoff_dist = cutoff * bandwidth
    inv_bw = 1.0 / bandwidth
    norm_const = 1.0 / (np.sqrt(2 * np.pi) * bandwidth)

    j_start = 0

    for i in range(N):
        if i % 1000 == 0:
            print(i)
        x_i = x[i]

        # Advance window start
        while j_start < N and x[j_start] < x_i - cutoff_dist:
            j_start += 1

        # Find window end
        j_end = j_start
        while j_end < N and x[j_end] <= x_i + cutoff_dist:
            j_end += 1

        # Compute kernel sum
        for j in range(j_start, j_end):
            dx = (x_i - x[j]) * inv_bw
            densities[i] += norm_const * np.exp(-0.5 * dx * dx)

    densities /= N
    return densities
