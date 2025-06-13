import sys
sys.path.append('../')

import numpy as np
import scipy.integrate as intg
import matplotlib.pyplot as plt

from CDF1DOptimizers import empirical_cdf_on_grid, particles_from_cdf

EPS = 1.e-10

# Setup the cumulative distribution
S = lambda x: np.tanh(x)
D = 0.1
L = 10
mu = lambda x: np.exp( (S(x) + S(x)**3 / 6.0) / D)
Z, _ = intg.quad(mu, -L, L)

n_points = 1001
grid = np.linspace(-L, L, n_points)
dx = 2.0 * L / (n_points - 1)
mu_grid = mu(grid) * dx / Z
cdf_grid = mu_grid.cumsum()
cdf_grid = cdf_grid / cdf_grid[-1] # Rescale to ensure final value is 1
cdf_grid = np.where(cdf_grid < EPS, 0.0, cdf_grid)

# Lifting: generate particles
N = 10**5
particles = particles_from_cdf(grid, cdf_grid, N)

# Restriction: Build the CDF from the particles
new_cdf = empirical_cdf_on_grid(particles, grid)

# Plot samples versus density
plt.hist(particles, density=True, bins=1000, color='tab:orange', label='Lifted Particles')
plt.plot(grid, cdf_grid, linestyle='--', linewidth=2, color='tab:red', label='Initial CDF')
plt.plot(grid, new_cdf, color='tab:green', label='Restricted CDF')
plt.plot(grid, mu(grid) / Z, color='k', label='Initial Density')
plt.xlabel('x')
plt.ylabel(r'$\mu(x)$')
plt.grid(True)
plt.legend()
plt.show()