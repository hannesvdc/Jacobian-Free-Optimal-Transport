import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy import optimize as opt

from typing import Tuple, List, Callable

# empirical ICDF on percentile grid
def icdf_on_percentile_grid(particles : np.ndarray,
                            percentiles : np.ndarray) -> np.ndarray:
    """
    Compute the empirical ICDF of particles in a given percentile grid. Assumes the 
    percentiles map to integer indices.

    Parameters
    ----------
    particles : (N,) array_like
        Data points (need NOT be sorted).
    percentiles : (n_grid,) array_like
        Percentiles at which to evaluate the ICDF. Does not include 0 or 1.

    Returns
    -------
    icdf : (M,) ndarray
        Empirical ICDF values
    """
    N = len(particles)

    # Sort the particles once in O(N log N)
    particles = np.sort(particles)
    indices = (percentiles * N).astype(int)

    return particles[indices]

def particles_from_icdf(percentile_grid : np.ndarray,
                        icdf : np.ndarray,
                        N : int,
                        boundary = ((0.0, -20), (1.0, 20))) -> np.ndarray:
    """
    Sample a given inverse cumulative density function evaluated in a precentile grid.
    We first build a spline interpolator of the ICDF, and then evaluate it in N 
    equidistant percentiles between 0 and 1.

    Parameters
    ----------
    percentile_grid: (M,) array_like
        Monotole grid of given percentiles
    icdf: (M,) array_like
        The ICDF evaluated in the percentile_grid
    N: int
        The number of samples to generate

    Returns
    -------
    particles: (N,) ndarray
        Locations of the new samples / percentiles
    """

    # build a monotone cubic spline interpolator
    spline = PchipInterpolator(np.concatenate(([0.0], percentile_grid, [1.0])), np.concatenate(([-10.0], icdf, [10.0])), extrapolate=False)

    # target percentiles (k+0.5) / N
    probs = (np.arange(N) + 0.5) / N

    # Evaluate the ICDF in the new percentiles
    particles = spline(probs)

    return particles

def icdf_newton_krylov(
        icdf0: np.ndarray,
        percentile_grid: np.ndarray,
        particle_timestepper: Callable[[np.ndarray], np.ndarray],
        maxiter: int,
        rdiff: float,
        N: int) -> Tuple[np.ndarray, List]:
    
    # Create the icdf to icdf timestepper
    def timestepper(icdf):
        particles = particles_from_icdf(percentile_grid, icdf, N)
        new_particles = particle_timestepper(particles)
        icdf_new = icdf_on_percentile_grid(new_particles, percentile_grid)
        return icdf_new
    def psi(icdf):
        psi_val = icdf - timestepper(icdf)
        print('psi_val', np.linalg.norm(psi_val))
        return psi_val

    # Create a callback to store intermediate losses and particles
    losses = [np.linalg.norm(psi(icdf0))]
    icdfs = [np.copy(icdf0)]
    def callback(xk, fk):
        psi_val = np.linalg.norm(fk)
        losses.append(psi_val)
        icdfs.append(np.copy(xk))
        print(f"(N = {N}, rdiff = {rdiff}) Epoch {len(losses)}: psi_val = {psi_val}")
    
    # Solve psi(x) = 0 using scipy.newton_krylov.
    line_search = 'wolfe'
    tol = 1e-14
    try:
        icdf_inf = opt.newton_krylov(psi, icdf0, f_tol=tol, maxiter=maxiter, rdiff=rdiff, line_search=line_search, callback=callback, verbose=True)
    except KeyboardInterrupt:
        print('Stopping Newton-Krylov due to user interrupt')
        icdf_inf = icdfs[-1]
    except:
        print('Stopping Newton-Krylov because maximum number of iterations was reached.')
        icdf_inf = icdfs[-1]

    return icdf_inf, losses