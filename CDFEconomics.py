import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import argparse

from concurrent.futures import ThreadPoolExecutor

import EconomicAgentTimestepper as agents
import EconomicPDETimestepper as pde
import CDF1DOptimizers as cdfopt

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
    sigma0 = 1.0
    X0 = np.random.normal(0.0, sigma0, N)
    X0[X0 <= -1.0] = 0.0
    X0[X0 >=  1.0] = 0.0
    cdf0 = np.array([np.mean(X0 <= grid[i]) for i in range(len(grid))])
    print('initial cdf', cdf0)

    # Find the steady-state CDF
    maxiter = 100
    rdiff = 1e0
    cdf_inf, losses = cdfopt.cdf_newton_krylov(cdf0, grid, agent_timestepper, maxiter, rdiff, N)

    # Calculate the final CDF
    N_faces = 100
    x_faces = np.linspace(-1.0, 1.0, N_faces)
    x_centers = 0.5 * (x_faces[1:] + x_faces[:-1])
    rho0 = np.exp(-x_centers**2 / (2.0 * sigma0**2)) / np.sqrt(2.0 * np.pi * sigma0**2)
    F = lambda rho: rho - pde.PDETimestepper(rho, x_faces, dt, Tpsi, gamma, vplus, vminus, eplus, eminus, g)
    try:
        rho_nk = opt.newton_krylov(F, rho0, maxiter=maxiter)
    except opt.NoConvergence as e:
        rho_nk = e.args[0]
    cdf_nk = np.concatenate(([0.0], np.cumsum(rho_nk)))
    cdf_nk /= cdf_nk[-1]

    # Plot the initial, and optimized CDFs
    plt.plot(grid, cdf0, label='Initial CDF')
    plt.plot(grid, cdf_inf, label='Optimized CDF')
    plt.plot(x_faces, cdf_nk, label='Steady-State CDF')
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
    sigma0 = 1.0
    X0 = np.random.normal(0.0, sigma0, N)
    X0[X0 <= -1.0] = 0.0
    X0[X0 >=  1.0] = 0.0
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
    if args.experiment == 'cdf':
        CDFNewtonKrylov()
    elif args.experiment == 'optimal-rdiff':
        optimalRDiff()