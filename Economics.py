import math
import numpy as np
import torch as pt
import scipy.optimize as opt
import matplotlib.pyplot as plt
import argparse

import EconomicAgentTimestepper as agents
import EconomicPDETimestepper as pde
import Wasserstein1DOptimizers as wopt

def compareAgentsAndPDE():
    N = 50000
    eplus = 0.075
    eminus = -0.072
    vplus = 20
    vminus = 20
    gamma = 1
    g = 38.0

    # Calculate aggregate constants
    c = vplus * eplus + vminus * eminus
    sigma = math.sqrt(vplus * eplus**2 + vminus * eminus**2)

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
    rho_T = pde.PDETimestepper(rho0, x_faces, dt, T, gamma, c, sigma)
    print('rho_T', rho_T)

    # Find the steady-state of the PDE through Newton-Krylov
    Tpsi = 1.e-1
    F = lambda rho: rho - pde.PDETimestepper(rho, x_faces, dt, Tpsi, gamma, c, sigma)
    rho_nk = opt.newton_krylov(F, rho0, maxiter=100)

    # Plot the histogram and density
    plt.hist(x, bins=int(math.sqrt(N)), density=True, label=rf"$T =${T}")
    plt.plot(x_centers, rho_T, label='Density after Time Evolution')
    plt.plot(x_centers, rho_nk, linestyle='--', label='Density after Newton-Krylov')
    plt.xlabel('Agents')
    plt.legend()
    plt.show()

def agentSteadyStateWasserstein():
    # Model parameters
    N = 50000
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
    lr = 1.e-3
    lr_decrease_factor = 0.1
    lr_decrease_step = 1000
    n_lrs = 4
    epochs = n_lrs * lr_decrease_step
    xf, losses, gradnorms = wopt.wasserstein_adam(x0, agent_timestepper, epochs, batch_size, lr, lr_decrease_factor, lr_decrease_step, pt.device('cpu'), store_directory=None)

    # Plot a histogram of the final particles
    plt.hist(xf.numpy(), bins=int(math.sqrt(N)), density=True)
    plt.xlabel('Agents')
    plt.legend()

    epoch_counter = np.linspace(0, len(losses), len(losses))
    plt.figure()
    plt.semilogy(epoch_counter, losses, label='Losses')
    plt.semilogy(epoch_counter, gradnorms, label='Gradient Norms')
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
    elif args.experiment == 'wasserstein':
        agentSteadyStateWasserstein()