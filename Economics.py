import math
import numpy as np
import torch as pt
import scipy.optimize as opt
import matplotlib.pyplot as plt
import argparse

from concurrent.futures import ThreadPoolExecutor

import EconomicAgentTimestepper as agents
import EconomicPDETimestepper as pde
import Wasserstein1DOptimizers as wopt
import CDF1DOptimizers as cdfopt

def compareAgentsAndPDE():
    N = 50000
    eplus = 0.075
    eminus = -0.072
    vplus = 20
    vminus = 20
    gamma = 1
    g = 38.0

    # Time stepping parameters
    T = 100.0
    dt = 0.25

    # Agent time evolution up to time T
    vpc = vplus
    vmc = vminus
    sigma0 = 0.1
    x0 = np.random.normal(0, sigma0, N)
    x0[x0 <= -1.0] = 0.0
    x0[x0 >=  1.0] = 0.0
    k = int(T / dt)
    x = agents.evolveAgentsNumpy(x0, k, dt, gamma, vplus, vminus, vpc, vmc, eplus, eminus, g, N)

    # PDE time evolution up to time T
    dt = 1.e-4
    N_faces = 100
    x_faces = np.linspace(-1.0, 1.0, N_faces)
    x_centers = 0.5 * (x_faces[1:] + x_faces[:-1])
    rho0 = np.exp(-x_centers**2 / (2.0 * sigma0**2)) / np.sqrt(2.0 * np.pi * sigma0**2)
    rho_T = pde.PDETimestepper(rho0, x_faces, dt, T, gamma, vplus, vminus, eplus, eminus, g)
    print('rho_T', rho_T)

    # Find the steady-state of the PDE through Newton-Krylov
    Tpsi = 1.e-1
    F = lambda rho: rho - pde.PDETimestepper(rho, x_faces, dt, Tpsi, gamma, vplus, vminus, eplus, eminus, g)
    rho_nk = opt.newton_krylov(F, rho0, maxiter=100)

    # Plot the histogram and density
    print('Average particle location', np.mean(x))
    plt.hist(x, bins=int(math.sqrt(N)), density=True, label=rf"$T =${T}")
    plt.plot(x_centers, rho_T, label='Density after Time Evolution')
    plt.plot(x_centers, rho_nk, linestyle='--', label='Density after Newton-Krylov')
    plt.plot(x_centers, rho0, label='Initial Distribution')
    plt.xlabel('Agents')
    plt.legend()
    plt.show()

def agentSteadyStateAdam():
    # Model parameters
    N = 50_000
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
    def agent_timestepper(X: pt.Tensor): # Input shape (B, 1)
        B = X.size()[0]
        x = pt.squeeze(X)
        x = agents.evolveAgentsTorch(x, n_steps, dt, gamma, vplus, vminus, vpc, vmc, eplus, eminus, g, B, verbose=False)
        x = pt.unsqueeze(x, dim=1)
        return x

    # Agent time evolution up to time T
    sigma0 = 0.1
    x0 = sigma0 * pt.randn(size=(N,1))
    x0[x0 <= -1.0] = 0.0
    x0[x0 >=  1.0] = 0.0

    # Do Wasserstein-Adam optimization
    batch_size = N
    lr = 1.e-2
    lr_decrease_factor = 0.1
    lr_decrease_step = 1000
    n_lrs = 4
    epochs = n_lrs * lr_decrease_step
    xf, losses, gradnorms = wopt.wasserstein_adam(x0, agent_timestepper, epochs, batch_size, lr, lr_decrease_factor, lr_decrease_step, pt.device('cpu'), store_directory=None)

    # Run the agents to a large time T to compare
    T = 100.0
    x0 = np.random.normal(0, sigma0, N)
    x0[x0 <= -1.0] = 0.0
    x0[x0 >=  1.0] = 0.0
    k = int(T / dt)
    x_evolution = agents.evolveAgentsNumpy(x0, k, dt, gamma, vplus, vminus, vpc, vmc, eplus, eminus, g, N)

    # Also get the PDE solution at time T
    dt = 1.e-4
    N_faces = 100
    x_faces = np.linspace(-1.0, 1.0, N_faces)
    x_centers = 0.5 * (x_faces[1:] + x_faces[:-1])
    rho0 = np.exp(-x_centers**2 / (2.0 * sigma0**2)) / np.sqrt(2.0 * np.pi * sigma0**2)
    F = lambda rho: rho - pde.PDETimestepper(rho, x_faces, dt, Tpsi, gamma, vplus, vminus, eplus, eminus, g)
    rho_nk = opt.newton_krylov(F, rho0, maxiter=100)

    # Plot a histogram of the final particles
    print('Average particle location', np.mean(xf.numpy()))
    plt.hist(xf.numpy(), bins=int(math.sqrt(N)), density=True, alpha=0.5, label='Optimized Agents')
    plt.hist(x_evolution, bins=int(math.sqrt(N)), density=True, alpha=0.5, label='Time-Evolved Agents')
    plt.plot(x_centers, rho_nk, label='PDE Solution')
    plt.plot(x_centers, rho0, label='Initial Distribution')
    plt.xlabel('Agents')
    plt.legend()

    epoch_counter = np.linspace(0, len(losses), len(losses))
    plt.figure()
    plt.semilogy(epoch_counter, losses, label='Losses')
    plt.semilogy(epoch_counter, gradnorms, label='Gradient Norms')
    plt.legend()
    plt.show()

def agentSteadyStateNewtonKrylov():
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
    def agent_timestepper(X: pt.Tensor) -> pt.Tensor: # Input shape (N,1) for consistency
        B = X.size()[0]
        x = pt.squeeze(X)
        x = agents.evolveAgentsTorch(x, n_steps, dt, gamma, vplus, vminus, vpc, vmc, eplus, eminus, g, B, verbose=False)
        x = pt.unsqueeze(x, dim=1)
        return x

    # Agent time evolution up to time T
    sigma0 = 1.0
    X0 = np.random.normal(0.0, sigma0, N)
    X0[X0 <= -1.0] = 0.0
    X0[X0 >=  1.0] = 0.0
    print('X0', X0)

    # Newton-Krylov
    burnin_T = None
    device = pt.device('cpu')
    dtype = pt.float64
    rdiff = 1.e-1 # the epsilon parameter
    line_search = 'wolfe'
    maxiter = 100
    x_inf, losses, grad_norms = wopt.wasserstein_newton_krylov(X0, agent_timestepper, maxiter, rdiff, line_search, burnin_T, device, dtype, store_directory=None)

    # Also get the PDE solution at time T
    dt = 1.e-4
    N_faces = 100
    x_faces = np.linspace(-1.0, 1.0, N_faces)
    x_centers = 0.5 * (x_faces[1:] + x_faces[:-1])
    rho0 = np.exp(-x_centers**2 / (2.0 * sigma0**2)) / np.sqrt(2.0 * np.pi * sigma0**2)
    F = lambda rho: rho - pde.PDETimestepper(rho, x_faces, dt, Tpsi, gamma, vplus, vminus, eplus, eminus, g)
    rho_nk = opt.newton_krylov(F, rho0, maxiter=100)

    # Plot a histogram of the final particles
    print('Average particle location', np.mean(x_inf))
    plt.hist(x_inf, bins=int(math.sqrt(N)), density=True, alpha=0.5, label='Optimized Agents')
    plt.plot(x_centers, rho_nk, label='PDE Solution')
    plt.plot(x_centers, rho0, label='Initial Distribution')
    plt.xlabel('Agents')
    plt.legend()

    epoch_counter = np.linspace(0, len(losses), len(losses))
    plt.figure()
    plt.title('Newton-Krylov Convergence')
    plt.semilogy(epoch_counter, losses, label='Losses')
    plt.semilogy(epoch_counter, grad_norms, label='Gradient Norms')
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
    def agent_timestepper(X: pt.Tensor) -> pt.Tensor: # Input shape (N,1) for consistency
        B = X.size()[0]
        x = pt.squeeze(X)
        x = agents.evolveAgentsTorch(x, n_steps, dt, gamma, vplus, vminus, vpc, vmc, eplus, eminus, g, B, verbose=False)
        x = pt.unsqueeze(x, dim=1)
        return x

    # Agent time evolution up to time T
    sigma0 = 1.0
    X0 = np.random.normal(0.0, sigma0, N)
    X0[X0 <= -1.0] = 0.0
    X0[X0 >=  1.0] = 0.0

    # Newton-Krylov
    burnin_T = None
    device = pt.device('cpu')
    dtype = pt.float64
    line_search = 'wolfe'
    maxiter = 100

    # Try many rdiffs
    rdiffs = [1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.0, 10.0]
    def run_one(rdiff):
        print('rdiff', rdiff)
        x_inf, losses, grad_norms = wopt.wasserstein_newton_krylov(X0, agent_timestepper, maxiter, rdiff, line_search, burnin_T, device, dtype, store_directory=None)
        return losses
    with ThreadPoolExecutor() as ex:
        total_losses = list(ex.map(run_one, rdiffs))

    # Plot the losses for every rdiff
    iterations = np.linspace(0.0, len(total_losses[0]), len(total_losses))
    for index in range(len(rdiffs)):
        plt.semilogy(iterations, total_losses[index], label=f"rdiff = {rdiffs[index]}")
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()

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
    rdiff = 1.e-1
    cdf_inf = cdfopt.cdf_newton_krylov(cdf0, grid, agent_timestepper, maxiter, rdiff, N)

    # Calculate the final CDF
    N_faces = 100
    x_faces = np.linspace(-1.0, 1.0, N_faces)
    x_centers = 0.5 * (x_faces[1:] + x_faces[:-1])
    rho0 = np.exp(-x_centers**2 / (2.0 * sigma0**2)) / np.sqrt(2.0 * np.pi * sigma0**2)
    F = lambda rho: rho - pde.PDETimestepper(rho, x_faces, dt, Tpsi, gamma, vplus, vminus, eplus, eminus, g)
    rho_nk = opt.newton_krylov(F, rho0, maxiter=100)
    cdf_nk = np.concatenate(([0.0], np.cumsum(rho_nk)))
    cdf_nk /= cdf_nk[-1]

    # Plot the initial, and optimized CDFs
    plt.plot(grid, cdf0, label='Initial CDF')
    plt.plot(grid, cdf_inf, label='Optimized CDF')
    plt.plot(grid, cdf_nk='Steady-State CDF')
    plt.xlabel(r'$x$')
    plt.legend()
    plt.show()

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, dest='experiment', default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parseArguments()
    if args.experiment == 'compareAgentsAndPDE':
        compareAgentsAndPDE()
    elif args.experiment == 'adam':
        agentSteadyStateAdam()
    elif args.experiment == 'newton-krylov':
        agentSteadyStateNewtonKrylov()
    elif args.experiment == 'optimal-rdiff':
        optimalRDiff()
    elif args.experiment == 'cdf':
        CDFNewtonKrylov()