import sys
sys.path.append('../')

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from Density1DOptimizers import reflected_hmc_from_tabulated_density, kde_1d_fft_neumann

def testSamplingAndKDE():
    # Setup the distribution
    S = lambda x: np.tanh(x)
    D = 0.1
    L = 10
    mu = lambda x: np.exp( (S(x) + S(x)**3 / 6.0) / D)
    grid = np.linspace(-L, L, 101)
    mu_grid = mu(grid)
    Z = np.trapz(mu_grid, grid)
    mu_grid /= Z

    # Sampling parameters
    N = 10**5
    step = 2.0

    # Do sampling
    samples = reflected_hmc_from_tabulated_density(grid, mu_grid, N, step)

    # Do KDE again
    bw = 0.25
    kde = kde_1d_fft_neumann(samples, grid, bw)

    # Plot samples versus density
    x_array = np.linspace(-L, L, 1001)
    dist = np.exp( (S(x_array) + S(x_array)**3 / 6.0) / D)
    Z = np.trapz(dist, x_array)
    dist = dist / Z
    plt.hist(samples, density=True, bins=100, color='tab:orange', label='MCMC')
    plt.plot(x_array, dist, linestyle='--', linewidth=2, color='tab:red', label='Steady State')
    plt.plot(grid, kde, label='Gaussian KDE')
    plt.xlabel('x')
    plt.ylabel(r'$\mu(x)$')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    testSamplingAndKDE()