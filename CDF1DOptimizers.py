import numpy as np
from scipy.interpolate import CubicSpline
from scipy import optimize as opt

# Empirical CDF from particles on a fixed grid
def empirical_cdf_on_grid(particles: np.ndarray,
                          grid:     np.ndarray) -> np.ndarray:
    """
    Compute the empirical CDF on a set of predetermined grid points.

    Parameters
    ----------
    particles : (N,) array_like
        Data points (need NOT be sorted).
    grid     : (M,) array_like
        Monotone grid (length 101 in your use-case) where the CDF is desired.

    Returns
    -------
    cdf_knots : (M,) ndarray
        Empirical CDF values  F(grid[j]) = #{x_i ≤ grid[j]} / N.
    """

    # Sort once for O(N log N); cumulative counts are then O(M+N)
    particles = np.sort(particles)
    idx  = np.searchsorted(particles, grid, side="right")
    return idx / particles.size   # vectorised division → CDF

#  Draw new particles from a tabulated CDF
def particles_from_cdf(grid: np.ndarray,
                       cdf:  np.ndarray,
                       N:    int,
                       eps: float = 1e-12,
                       solver_tol: float = 1e-10) -> np.ndarray:
    """
    Given F(x_j) on 101 knots, build a clamped cubic spline and
    return N particle locations at the  k/N  quantiles.

    Parameters
    ----------
    x_knots   : (M,) array_like
        Monotone grid (length 101).
    cdf_knots : (M,) array_like
        CDF values on that grid; should start at 0, end at 1, and be monotone.
    N         : int
        Number of particles (quantiles) to generate.
    eps       : float, default 1e-12
        Safety margin to keep percentiles inside (0,1).
    solver_tol: float, default 1e-10
        Absolute tolerance for the Brent root finder.

    Returns
    -------
    new_particles : (N,) ndarray
        Locations x_k  such that  F(x_k) = (k+½)/N  (mid-probability rule).
    """

    # build clamped cubic spline of the CDF. Dirichlet BC are part of the CDF values.
    spline = CubicSpline(grid, cdf, bc_type='not-a-knot')

    # target percentiles  k / N
    probs = (np.arange(N) + 1.0) / N

    # for each percentile, bracket and invert
    # cdf is monotone, so searchsorted gives the interval in O(log M)
    particles = np.zeros(N)
    for k, p in enumerate(probs):
        j = np.searchsorted(cdf, p)
        xl, xr = grid[j-1], grid[j]

        if j == len(grid)-1 and spline(xr) < p:
            particles[k] = grid[-1]
            continue

        # solve F(x) - p = 0 on [xl,xr]
        root = opt.brentq(lambda x: spline(x) - p, xl, xr, xtol=solver_tol)
        particles[k] = root

    return particles

def cdf_newton_krylov(
    cdf0: np.ndarray,
    grid : np.ndarray,
    particle_timestepper, # np.ndarray to np.ndarray
    maxiter: int,
    rdiff : float,
    N : int, 
    store_directory: str | None = None
) -> np.ndarray:
    
    # Create the cdf to cdf timestepper
    def timestepper(cdf):
        particles = particles_from_cdf(grid, cdf, N)
        new_particles = particle_timestepper(particles)
        cdf_new = empirical_cdf_on_grid(new_particles, grid)
        return cdf_new
    def psi(cdf):
        return cdf - timestepper(cdf)
    
    # Create a callback to store intermediate losses and particles
    losses = []
    densities = []
    def callback(xk, fk):
        # Compute loss (we need to recompute it here, since fk is just the gradient)
        psi_val = np.linalg.norm(fk)
        losses.append(psi_val)
        densities.append(xk)
        print(f"(N = {N}, rdiff = {rdiff}) Epoch {len(losses)}: psi_val = {psi_val}")

    # Solve F(x) = 0 using scipy.newton_krylov. The parameter rdiff is key!
    line_search = None
    tol = 1.e-14
    try:
        x_inf = opt.newton_krylov(psi, cdf0, f_tol=tol, maxiter=maxiter, rdiff=rdiff, line_search=line_search, callback=callback, verbose=True)
    except opt.NoConvergence as e:
        x_inf = e.args[0]
    except KeyboardInterrupt as e:
        print('Stopping newtion krylov')
        x_inf = densities[-1]

    return x_inf, losses