import sys
sys.path.append('../')

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

from FastKDE import fast_sliding_kde

def testGaussianKDE():
    # Draw sorted samples from the standard normal gaussian
    N = 1_000_000
    rng = rd.RandomState()
    particles = rng.normal(0.0, 1.0, N)
    particles = np.sort(particles)

    # Bandwidth
    bandwidth = 0.05 # 1 stdev. Perhaps too large.

    # Compute KDE
    print('Calculating KDE')
    kde = fast_sliding_kde(particles, bandwidth)

    # Plot
    print('Plotting')
    gaussian_density = lambda x: np.exp(-x**2 / 2.0) / np.sqrt(2.0 * np.pi)
    x_array = np.linspace(-4, 4, 1001)
    plt.plot(x_array, gaussian_density(x_array), label='Standard Normal')
    plt.plot(particles, kde, label='KDE')
    plt.xlabel(r'$x$')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    testGaussianKDE()