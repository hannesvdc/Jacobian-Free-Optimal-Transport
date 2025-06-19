import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

from CDF2DOptimizers import particles_from_joint_cdf_cubic

EPS = 1.e-10

def testGaussianBimodal():
    # Setup the potential energy for this distribution
    gaussian_V = lambda x: (x - 1.0)**2 / (2.0 * 0.5**2) 
    bimodal_V = lambda y: 0.5 * (y**2 - 1)**2
    unnormalize_density = lambda x, y: np.exp(-gaussian_V(x) - bimodal_V(y))

    # Build the cumulative density
    n_points = 101
    x_min = -1
    x_max = 3
    y_min = -3
    y_max = 3
    grid_x = np.linspace(x_min, x_max, n_points)
    grid_y = np.linspace(y_min, y_max, n_points)
    dx = 2.0 * (x_max - x_min) / (n_points - 1)
    dy = 2.0 * (y_max - y_min) / (n_points - 1)
    X, Y = np.meshgrid(grid_x, grid_y)
    density_grid = unnormalize_density(X, Y).transpose() * dx * dy
    cdf_grid = density_grid.cumsum(axis=0).cumsum(axis=1)
    cdf_grid = cdf_grid / cdf_grid[-1, -1] # Rescale to ensure final value is 1
    cdf_grid = np.where(cdf_grid < EPS, 0.0, cdf_grid)

    # Lifting: generate particles
    N = 10**5
    particles = particles_from_joint_cdf_cubic(grid_x, grid_y, cdf_grid, N, eps=EPS)
    particles_x = particles[:,0]
    particles_y = particles[:,1]

    # Marginalize and plot in each dimension separately (there is no 'correlation' between X and Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, cdf_grid, cmap='viridis') # type: ignore
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    plt.figure()
    x_plot_array = np.linspace(x_min, x_max, 1001)
    x_density = np.exp(-gaussian_V(x_plot_array))
    x_density /= np.trapz(x_density, x_plot_array)
    plt.hist(particles_x, bins=40, density=True, label=r'CDF Sampling $X$')
    plt.plot(x_plot_array, x_density, label=r'Marginal Distribution in $X$')
    plt.xlabel(r'$x$')
    plt.legend()

    plt.figure()
    y_plot_array = np.linspace(y_min, y_max, 1001)
    y_density = np.exp(-bimodal_V(y_plot_array))
    y_density /= np.trapz(y_density, y_plot_array)
    plt.hist(particles_y, bins=40, density=True, label=r'CDF Sampling $Y$')
    plt.plot(y_plot_array, y_density, label=r'Marginal Distribution in $Y$')
    plt.xlabel(r'$y$')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    testGaussianBimodal()