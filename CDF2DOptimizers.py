import numpy as np
from scipy.interpolate import RectBivariateSpline, PchipInterpolator

def empirical_joint_cdf_on_grid(particles: np.ndarray,
                                x_grid:    np.ndarray,
                                y_grid:    np.ndarray) -> np.ndarray:
    """
    Empirical joint CDF  F(x_i, y_j) = #{ (X_n ≤ x_i) & (Y_n ≤ y_j) } / N.

    Parameters
    ----------
    particles : (N, 2) array_like
        Unsordered (x, y) points.
    x_grid    : (Kx,)   array_like (monotone increasing)
    y_grid    : (Ky,)   array_like (monotone increasing)

    Returns
    -------
    F_table : (Kx, Ky) ndarray
        Joint CDF evaluated on the tensor grid.
    """
    N = particles.shape[0]
    H, _, _ = np.histogram2d(particles[:, 0], particles[:, 1], bins=[x_grid, y_grid])
    F = H.cumsum(axis=0).cumsum(axis=1) / N
    F /= F[-1,-1] # Divide by the last element to reduce rounding errors in cumsum

    return F

def invert_table_monotone(values: np.ndarray,
                          grid:   np.ndarray,
                          probs:  np.ndarray) -> np.ndarray:
    """
    Fast vectorised inverse of a monotone 1-D table (values[j] = F(grid[j])).

    Returns x such that linear-interp(values)(x) = probs[k].
    `values` must be strictly increasing, `grid` the matching knots.
    """
    idx = np.searchsorted(values, probs, side='right')
    idx = np.clip(idx, 1, len(grid) - 1)
    lam = (probs - values[idx-1]) / (values[idx] - values[idx-1])
    return (1.0 - lam) * grid[idx-1] + lam * grid[idx]

# ChatGPT
def particles_from_joint_cdf_cubic(x_grid: np.ndarray,
                                   y_grid: np.ndarray,
                                   cdf:  np.ndarray,
                                   N:      int,
                                   eps:    float = 1e-12) -> np.ndarray:
    """
    Lift N ≈ NxxNy particles from a joint CDF table using:
      • one bicubic spline (RectBivariateSpline) for F and ∂F/∂x
      • PCHIP splines for each 1-D conditional CDF row (cubic + monotone)
      • linear bracketing + one Newton step per inversion  →  O(N) time.

    Returns
    -------
    cloud : (N, 2) ndarray   deterministic percentile lattice.
    """
    # lattice dimensions
    Nx = int(np.floor(np.sqrt(N)))
    Ny = int(np.ceil(N / Nx))
    N  = Nx * Ny # number of particles returned

    # bicubic surface and its ∂F/∂x spline (built once)
    F_spl    = RectBivariateSpline(x_grid, y_grid, cdf, kx=3, ky=3, s=0)
    dFdx_spl = F_spl.partial_derivative(nu=1, nv=0)

    # 1-D marginal CDF  F_X(x)  (table already in cdf)
    Fx_vals = cdf[:, -1]
    Fx_spl  = PchipInterpolator(x_grid, Fx_vals)  # cubic, monotone

    # Invert for Nx mid-mass percentiles  (linear bracket + 1 Newton step)
    probs_x = (np.arange(Nx) + 1.0) / Nx
    x_q = invert_table_monotone(Fx_vals, x_grid, probs_x)           # linear
    x_q -= (Fx_spl(x_q) - probs_x) / Fx_spl.derivative()(x_q)       # refine vectorized

    # ∂F/∂x rows for all x_q in one vectorised call
    dFdx_rows = dFdx_spl(x_q, y_grid)           # shape (Nx, Ky)
    fX_vec    = dFdx_rows[:, -1]                # marginal densities
    G_rows    = dFdx_rows / (fX_vec[:, None] + eps)  # conditional CDF rows with zero detection

    # build particle cloud row-by-row 
    probs_y = (np.arange(Ny) + 1.0) / Ny
    particles = np.empty((N, 2))
    idx = 0
    for r, (xp, G_vals) in enumerate(zip(x_q, G_rows)):
        # cubic, monotone spline for this conditional CDF slice
        G_spl = PchipInterpolator(y_grid, G_vals)

        # invert: linear bracket + 1 Newton step
        y_q  = invert_table_monotone(G_vals, y_grid, probs_y)
        y_q -= (G_spl(y_q) - probs_y) / G_spl.derivative()(y_q)

        particles[idx:idx + Ny, 0] = xp
        particles[idx:idx + Ny, 1] = y_q
        idx += Ny

    return particles


# My implementation
# def particles_from_joint_cdf(x_grid: np.ndarray,
#                              y_grid: np.ndarray,
#                              cdf:  np.ndarray,
#                              N: int,
#                              eps: float = 1e-12,
#                              ) -> np.ndarray:
#     """
#     Generate N ≈ Nx x Ny particles whose joint law matches the tabulated CDF.

#     Parameters
#     ----------
#     x_grid, y_grid : monotone grids (Kx, Ky)
#     F_tab          : (Kx, Ky)   joint CDF table  (must start at 0, end at 1)
#     N              : total #particles desired
#     Nx             : optional #quantiles along x; default ≈ sqrt(N)
#     eps, root_tol  : numerical tolerances (see 1-D version)

#     Returns
#     -------
#     cloud : (N, 2) ndarray of lifted (x, y) particles.
#     """
#     Nx = int(np.floor(np.sqrt(N)))
#     Ny = int(np.ceil(N / Nx))
#     N = Nx * Ny # number of particles we will return

#     # Smooth CDF spline & x-derivative
#     F  = RectBivariateSpline(x_grid, y_grid, cdf, kx=3, ky=3, s=0)
#     dFdx = F.partial_derivative(nu=1, nv=0)

#     # Sample the marginal CDF F_X(x)
#     F_X = lambda x: F(x, y_grid[-1]) # Need a callable function for brentq
#     F_X_vals = F(x_grid, y_grid[-1])
#     x_particles = invert_1d_spline(F_X, F_X_vals, x_grid, Nx)

#     # Build the full particle cloud
#     particles = np.zeros((N, 2))
#     idx = 0
#     for x_index in range(Nx):
#         xp = x_particles[x_index]

#         # Construct a callable conditional CDF and also one evaluated in the y_grid locations
#         f_X_in_xp = dFdx(xp, y_grid[-1])
#         F_Y_given_xp = lambda y: dFdx(xp, y) / (f_X_in_xp + eps)
#         F_Y_given_xp_vals = dFdx(xp, y_grid) / (f_X_in_xp + eps)

#         # Sample conditional CDF
#         y_particles = invert_1d_spline(F_Y_given_xp, F_Y_given_xp_vals, y_grid, Ny)

#         # 3c. store Ny particles with this xp
#         particles[idx:idx+Ny, 0] = xp
#         particles[idx:idx+Ny, 1] = y_particles
#         idx += Ny

#     return particles