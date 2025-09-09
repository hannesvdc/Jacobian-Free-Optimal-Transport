import numpy as np
from scipy.interpolate import RectBivariateSpline, PchipInterpolator
import scipy.optimize as opt

from typing import Tuple, List, Callable

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
    x_edges = np.concatenate(([-np.inf], x_grid))
    y_edges = np.concatenate(([-np.inf], y_grid))
    H, *_ = np.histogram2d(particles[:, 0], particles[:, 1],
                           bins=[x_edges, y_edges])
    F = H.cumsum(axis=0).cumsum(axis=1)
    F /= F[-1,-1]

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
    particles : (N, 2) ndarray   deterministic percentile lattice.
    """
    # lattice dimensions
    Nx = int(np.floor(np.sqrt(N)))
    Ny = int(np.ceil(N / Nx))
    N  = Nx * Ny # number of particles returned

    # bicubic surface and its ∂F/∂x spline (built once)
    F_spl    = RectBivariateSpline(x_grid, y_grid, cdf, kx=3, ky=3, s=0)
    dFdx_spl = F_spl.partial_derivative(dx=1, dy=0)

    # 1-D marginal CDF  F_X(x)  (table already in cdf)
    Fx_vals = cdf[:, -1]
    Fx_spl  = PchipInterpolator(x_grid, Fx_vals)  # cubic, monotone

    # Invert for Nx mid-mass percentiles  (linear bracket + 1 Newton step)
    probs_x = (np.arange(Nx) + 0.5) / Nx
    x_q = invert_table_monotone(Fx_vals, x_grid, probs_x)  # linear interpolation
    x_q -= (Fx_spl(x_q) - probs_x) / Fx_spl.derivative()(x_q)  # Newton refine vectorized

    # ∂F/∂x rows for all x_q in one vectorised call
    dFdx_rows = dFdx_spl(x_q, y_grid) # shape (Nx, Ky)
    fX_vec    = dFdx_rows[:, -1]  # marginal densities
    G_rows    = dFdx_rows / (fX_vec[:, None] + eps)  # conditional CDF rows with zero detection

    # build particle cloud row-by-row 
    probs_y = (np.arange(Ny) + 0.5) / Ny
    particles = np.empty((N, 2))
    idx = 0
    for _, (xp, G_vals) in enumerate(zip(x_q, G_rows)):
        # cubic, monotone spline for this conditional CDF slice
        G_spl = PchipInterpolator(y_grid, G_vals)

        # invert: linear bracket + 1 Newton step
        y_q  = invert_table_monotone(G_vals, y_grid, probs_y)
        y_q -= (G_spl(y_q) - probs_y) / G_spl.derivative()(y_q)

        particles[idx:idx + Ny, 0] = xp
        particles[idx:idx + Ny, 1] = y_q
        idx += Ny

    return particles

def cdf_newton_krylov(
    cdf0: np.ndarray,
    x_grid : np.ndarray,
    y_grid : np.ndarray,
    particle_timestepper: Callable[[np.ndarray], np.ndarray],
    maxiter: int,
    rdiff : float,
    N : int,
    line_search : str | None ='wolfe') -> Tuple[np.ndarray, List]:

    x_grid_points = len(x_grid)
    y_grid_points = len(y_grid)
    if len(cdf0.shape) > 1:
        cdf0 = cdf0.flatten()
    
    # Create the cdf to cdf timestepper
    def timestepper(cdf):
        cdf = cdf.reshape(x_grid_points, y_grid_points)
        particles = particles_from_joint_cdf_cubic(x_grid, y_grid, cdf, N) 
        new_particles = particle_timestepper(particles)
        cdf_new = empirical_joint_cdf_on_grid(new_particles, x_grid, y_grid)
        return cdf_new.flatten()
    def psi(cdf):
        psi_val = cdf - timestepper(cdf)
        print('psi_val', np.linalg.norm(psi_val))
        return psi_val
    
    # Create a callback to store intermediate losses and particles
    losses = [np.linalg.norm(psi(cdf0))]
    cdfs = [np.copy(cdf0)]
    def callback(xk, fk):
        # Compute loss (we need to recompute it here, since fk is just the gradient)
        psi_val = np.linalg.norm(fk)
        losses.append(psi_val)
        cdfs.append(np.copy(xk))
        print(f"(N = {N}, rdiff = {rdiff}) Epoch {len(losses)}: psi_val = {psi_val}")

    # Solve F(x) = 0 using scipy.newton_krylov. The parameter rdiff is key!
    tol = 1.e-14
    try:
        x_inf = opt.newton_krylov(psi, cdf0, f_tol=tol, maxiter=maxiter, rdiff=rdiff, line_search=line_search, callback=callback, verbose=True)
    except KeyboardInterrupt:
        print('Stopping Newton-Krylov due to user interrupt')
        x_inf = cdfs[-1]
    except:
        print('Stopping Newton-Krylov because maximum number of iterations was reached.')
        x_inf = cdfs[-1]
    x_inf = x_inf.reshape(x_grid_points, y_grid_points)

    return x_inf, losses