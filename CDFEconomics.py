import math
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import argparse

from concurrent.futures import ThreadPoolExecutor

import EconomicAgentTimestepper as agents
import EconomicPDETimestepper as pde
import CDF1DOptimizers as cdfopt

# Assumes cdf_grid and rho_grid are interleaving with len(cdf_grid) = len(rho_grid) + 1
def from_density_to_cdf(rho, rho_grid, cdf_grid):
    dx = rho_grid[1] - rho_grid[0]
    cdf = np.zeros_like(cdf_grid)
    for cdf_index in range(1, len(cdf)):
        cdf[cdf_index] = dx * np.sum(rho[0:cdf_index])
    return cdf / cdf[-1]

def CDFNewtonKrylov():
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
    n_grid_points = 101
    grid = np.linspace(-1.0, 1.0, n_grid_points)
    sigma0 = 0.1
    X0 = np.random.normal(0.0, sigma0, N)
    X0[X0 <= -1.0] = 0.0
    X0[X0 >=  1.0] = 0.0
    cdf0 = np.array([np.mean(X0 <= grid[i]) for i in range(len(grid))])
    print('initial cdf', cdf0)

    # Find the steady-state CDF
    maxiter = 20
    rdiff = 1e-1
    cdf_inf, losses = cdfopt.cdf_newton_krylov(cdf0, grid, agent_timestepper, maxiter, rdiff, N)
    particles_from_cdf_inf = cdfopt.particles_from_cdf(grid, cdf_inf, N)

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
    cdf_nk = from_density_to_cdf(rho_nk, x_centers, grid)

    # Plot the initial, and optimized CDFs
    plt.plot(grid, cdf0, label='Initial CDF')
    plt.plot(grid, cdf_inf, label='Optimized CDF')
    plt.plot(grid, cdf_nk, label='Steady-State CDF')
    plt.plot(x_centers, rho_nk, label='Steady-State Density (from PDE)')
    plt.xlabel(r'$x$')
    plt.legend()

    # Plot the partilces sampled from the ICDF, and the analytic invariant density
    plt.figure()
    plt.hist(particles_from_cdf_inf, density=True, bins=100, label='Particles from CDF')
    plt.plot(x_centers, rho_nk, label='Invariant Density')
    plt.xlabel(r'$x$')
    plt.legend()
 
    plt.figure()
    plt.semilogy(np.arange(len(losses)), losses, label='NK Losses')
    plt.xlabel('Newton-Krylov Iteration')
    plt.legend()
    plt.show()

def optimalRDiff():
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
    n_grid_points = 101
    grid = np.linspace(-1.0, 1.0, n_grid_points)
    sigma0 = 0.1
    X0 = np.random.normal(0.0, sigma0, N)
    cdf0 = np.array([np.mean(X0 <= grid[i]) for i in range(len(grid))])
    print('initial cdf', cdf0)

    # Try many rdiffs
    maxiter = 100
    rdiffs = [1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.0, 10.0]
    def run_one(rdiff):
        print('rdiff', rdiff)
        cdf_inf, losses = cdfopt.cdf_newton_krylov(cdf0, grid, agent_timestepper, maxiter, rdiff, N)
        return losses
    with ThreadPoolExecutor() as ex:
        total_losses = list(ex.map(run_one, rdiffs))

    # Plot the losses for every rdiff
    for index in range(len(rdiffs)):
        iterations = np.arange(len(total_losses[index]))
        plt.semilogy(iterations, total_losses[index], label=f"rdiff = {rdiffs[index]}")
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, dest='experiment', default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parseArguments()
    if args.experiment == 'newton-krylov':
        CDFNewtonKrylov()
    elif args.experiment == 'optimal-rdiff':
        optimalRDiff()