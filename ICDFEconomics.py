import math
import numpy as np
import scipy.optimize as opt
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import argparse

import EconomicAgentTimestepper as agents
import EconomicPDETimestepper as pde
import ICDF1DOptimizers as icdfopt

# Assumes cdf_grid and rho_grid are interleaving with len(cdf_grid) = len(rho_grid) + 1
def from_density_to_cdf(rho, rho_grid, cdf_grid):
    dx = rho_grid[1] - rho_grid[0]
    cdf = np.zeros_like(cdf_grid)
    for cdf_index in range(1, len(cdf)):
        cdf[cdf_index] = dx * np.sum(rho[0:cdf_index])
    return cdf / cdf[-1]

def invertCDF(cdf, cdf_grid, percentile_grid):
    spline = interpolate.PchipInterpolator(cdf_grid, cdf, extrapolate=True)

    particles = np.zeros_like(percentile_grid)
    for k, p in enumerate(percentile_grid):
        j = np.searchsorted(cdf, p)
        if j == len(cdf_grid):
            print('Index error', j, p, cdf[-1])
        xl, xr = cdf_grid[j-1], cdf_grid[j]

        if j == len(cdf_grid)-1 and spline(xr) < p:
            particles[k] = cdf_grid[-1]
            continue

        # solve F(x) - p = 0 on [xl,xr]
        root = opt.brentq(lambda x: spline(x) - p, xl, xr, xtol=1e-10)
        particles[k] = root

    return particles

def ICDFNewtonKrylov():
    # Model parameters
    N = 100_000
    eplus = 0.075
    eminus = -0.072
    vplus = 20
    vminus = 20
    vpc = vplus
    vmc = vminus
    gamma = 1
    g = 38.0

    # Time stepping parameters
    Tpsi = 1.0
    dt = 0.25
    n_steps = int(Tpsi / dt)
    def agent_timestepper(X: np.ndarray) -> np.ndarray: # Input shape (N,1) for consistency
        x = agents.evolveAgentsNumpy(X, n_steps, dt, gamma, vplus, vminus, vpc, vmc, eplus, eminus, g, len(X), verbose=False)
        return x

    # Sample particles and build the initial CDF
    n_cdf_points = 101
    n_icdf_points = 100
    cdf_grid = np.linspace(-1.0, 1.0, n_cdf_points)
    percentile_grid = (np.arange(n_icdf_points) + 0.5) / n_icdf_points
    sigma0 = 0.1
    X0 = np.random.normal(0.0, sigma0, N)
    X0[X0 <= -1.0] = 0.0
    X0[X0 >=  1.0] = 0.0
    icdf0 = icdfopt.icdf_on_percentile_grid(X0, percentile_grid)

    # Find the steady-state CDF
    boundary = ((0.0, -1.0), (1.0, 1.0))
    maxiter = 100
    rdiff = 1e-1
    icdf_inf, losses = icdfopt.icdf_newton_krylov(icdf0, percentile_grid, agent_timestepper, maxiter, rdiff, N, boundary)
    particles_from_icdf_inf = icdfopt.particles_from_icdf(percentile_grid, icdf_inf, N, boundary)

    # Calculate the final CDF
    sigma0_pde = 1.0
    dt_pde = 1.e-4
    maxiter = 100
    N_faces = 100
    x_faces = np.linspace(-1.0, 1.0, N_faces)
    x_centers = 0.5 * (x_faces[1:] + x_faces[:-1])
    rho0 = np.exp(-x_centers**2 / (2.0 * sigma0_pde**2)) / np.sqrt(2.0 * np.pi * sigma0_pde**2)
    F = lambda rho: rho - pde.PDETimestepper(rho, x_faces, dt_pde, Tpsi, gamma, vplus, vminus, eplus, eminus, g)
    try:
        rho_nk = opt.newton_krylov(F, rho0, maxiter=maxiter)
    except opt.NoConvergence as e:
        rho_nk = e.args[0]
    cdf_nk = from_density_to_cdf(rho_nk, x_centers, cdf_grid)
    icdf_nk = invertCDF(cdf_nk, cdf_grid, percentile_grid)
    icdf_nk = np.concatenate(([boundary[0][1]], icdf_nk, [boundary[1][1]]))

    # Plot both optimized ICDFs
    icdf_inf = np.concatenate(([boundary[0][1]], icdf_inf, [boundary[1][1]]))
    percentile_grid = np.concatenate(([boundary[0][0]], percentile_grid, [boundary[1][0]]))
    plt.plot(percentile_grid, icdf_nk, label='Exact Invariant ICDF')
    plt.plot(percentile_grid, icdf_inf, label='Optimized ICDF')
    plt.xlabel(r'$p$')
    plt.legend()

    plt.figure()
    plt.hist(particles_from_icdf_inf, density=True, bins=int(math.sqrt(N)), label='Particles from Invariant ICDF')
    plt.plot(x_centers, rho_nk, label='Invariant Density')
    plt.xlabel(r'$x$')
    plt.legend()
 
    plt.figure()
    plt.semilogy(np.arange(len(losses)), losses, label='NK Losses')
    plt.xlabel('Newton-Krylov Iteration')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    ICDFNewtonKrylov()