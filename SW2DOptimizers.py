import numpy as np
from scipy.interpolate import RectBivariateSpline, PchipInterpolator
import scipy.optimize as opt

from typing import Tuple, List, Callable

xy_max = 4
EPS_reg = 1e-5

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

def angular_cdf_from_2d_cdf(cdf_2d : RectBivariateSpline,
                                angular_grid : np.ndarray) -> np.ndarray:
    Fxy = cdf_2d.partial_derivative(dx=1, dy=1)

    # Build a Gauss-Legendre integration algorithm for F along the rays
    n_radial_points = 64
    gn, gw = np.polynomial.legendre.leggauss(n_radial_points)
    t = 0.5 * (gn + 1.0)      # nodes in [0,1]
    w01 = 0.5 * gw            # weights on [0,1]

    # Iterate over every angle theta in angular_grid and compute J(theta).
    # Assumes theta = -np.pi and theta = np.pi are in `angular_grid`.
    pdf_values = np.zeros_like(angular_grid)
    for index in range(len(angular_grid)):
        theta = angular_grid[index]
        c = np.cos(theta)
        s = np.sin(theta)

        # Build the conditional/radial PDF
        r_max = xy_max / max(np.abs(c), np.abs(s))
        r_nodes_int = r_max * t
        x_values = r_nodes_int * c
        y_values = r_nodes_int * s
        rho_values = np.clip(Fxy(x_values, y_values, grid=False), EPS_reg, None)

        # Calculate the angular PDF by integrating over rays
        pdf_value = np.dot(rho_values * r_nodes_int, r_max * w01)
        pdf_values[index] = pdf_value
    
    # Build the CDF from density values
    dtheta = np.diff(angular_grid)
    F = np.concatenate(([0.0], np.cumsum(0.5*(pdf_values[:-1] + pdf_values[1:]) * dtheta)))
    F /= F[-1]
    
    return F

def particles_from_angular_and_radial_cdf(cdf_2d : RectBivariateSpline,
                                          angular_grid : np.ndarray,
                                          angular_cdf_values: np.ndarray,
                                          N : int) -> np.ndarray:
    # Calculate the two-dimensional density (rho = F_xy)
    Fxy = cdf_2d.partial_derivative(dx=1, dy=1)
    Na = int(np.floor(np.sqrt(N)))
    Nr = int(np.ceil(N / Na))

    # Generate samples from the angular CDF by building an inverse interpolator
    A_inverse = PchipInterpolator(angular_cdf_values, angular_grid, extrapolate=False)
    probs_a = (np.arange(Na) + 0.5) / Na
    a_samples = A_inverse(probs_a)

    # Prepare the collocation points for Gauss-Legendre quadrature of rho
    n_radial_points = 64
    gn, gw = np.polynomial.legendre.leggauss(n_radial_points)
    t = 0.5 * (gn + 1.0)      # nodes in [0,1]
    w01 = 0.5 * gw            # weights on [0,1]

    # For each theta, calculate the radial CDF and invert to sample
    probs_r = (np.arange(Nr) + 0.5) / Nr
    xy_samples = np.zeros((Na * Nr, 2))
    for index in range(len(a_samples)):
        theta = a_samples[index]
        c = np.cos(theta)
        s = np.sin(theta)

        # Build the conditional/radial CDF
        r_max = xy_max / max(np.abs(c), np.abs(s))
        r_nodes_int = r_max * t
        x_values = r_nodes_int * c
        y_values = r_nodes_int * s
        rho_values = np.clip(Fxy(x_values, y_values, grid=False), EPS_reg, None)

        # Do partial integration up to r for CDF calculations
        contributions = rho_values * r_nodes_int * (r_max * w01)
        C = np.cumsum(contributions)
        r_nodes = np.concatenate(([0.0], r_nodes_int))
        C = np.concatenate(([0.0], C))

        # Some post-processing on C
        F_radial = C / C[-1]
        F_radial = np.maximum.accumulate(np.clip(F_radial, 0.0, 1.0))

        # Generate samples from the radial CDF (F_radial)
        inverse_F_radial = PchipInterpolator(F_radial, r_nodes, extrapolate=False)
        r_samples = inverse_F_radial(probs_r)
        xy_samples[index*Nr : (index+1)*Nr, 0] = c * r_samples
        xy_samples[index*Nr : (index+1)*Nr, 1] = s * r_samples

    # Return the xy samples
    return xy_samples

def sw_newton_krylov(
    cdf0 : np.ndarray,
    x_grid : np.ndarray,
    y_grid : np.ndarray,
    angular_grid : np.ndarray,
    particle_timestepper : Callable[[np.ndarray], np.ndarray],
    maxiter : int,
    rdiff : float,
    N : int,
    line_search : str | None = 'wolfe') -> Tuple[np.ndarray, List]:

    x_grid_points = len(x_grid)
    y_grid_points = len(y_grid)
    if len(cdf0.shape) > 1:
        cdf0 = cdf0.flatten() # flatten for newton_krylov optimizer

    # create the CDF to CDF timestepper
    def timestepper(cdf):
        cdf = cdf.reshape(x_grid_points, y_grid_points)
        cdf_spline = RectBivariateSpline(x_grid, y_grid, cdf, kx=3, ky=3, s=0)
        angular_cdf_values = angular_cdf_from_2d_cdf(cdf_spline, angular_grid)
        particles = particles_from_angular_and_radial_cdf(cdf_spline, angular_grid, angular_cdf_values, N)
        new_particles = particle_timestepper(particles)
        new_cdf = empirical_joint_cdf_on_grid(new_particles, x_grid, y_grid)
        return new_cdf.flatten()
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

    # Solve F(x) = 0 using newton_krylov.
    tol = 1.e-14
    try:
        x_inf = opt.newton_krylov(psi, cdf0, f_tol=tol, maxiter=maxiter, rdiff=rdiff, line_search=line_search, callback=callback, verbose=True)
    except KeyboardInterrupt:
        print('Stopping Newton-Krylov due to user interrupt')
        x_inf = cdfs[-1]
    except Exception as e:
        print('Stopping Newton-Krylov because maximum number of iterations was reached.')
        print(e)
        x_inf = cdfs[-1]
    x_inf = x_inf.reshape(x_grid_points, y_grid_points)

    return x_inf, losses