import math
import numpy as np
import matplotlib.pyplot as plt

b = lambda x: 1.0 + x**2
sigma = lambda x: 1.0 + x**2

def particleTimestepper(X, h, rng):
    return X + b(X) * h + sigma(X) * math.sqrt(h) * rng.normal(0.0, 1.0, len(X))

def plotInvariantDistribution():
    rng = np.random.RandomState()

    N = 10**5
    X = rng.normal(0.0, 0.1, N)

    # Do timstepping
    h = 1.e-5
    T = 10.0
    n_steps = int(T / h)
    for n in range(n_steps):
        print('t =', round((n+1)*h, 4), np.count_nonzero(~np.isnan(X)))
        X = particleTimestepper(X, h, rng)
    X = X[np.abs(X) < 8] # Filter diverged particles
    print('Remaining Particles:', len(X))

    # Plot histogram next to the invariant distribution
    x_array = np.linspace(-6.0, 6.0, 1001)
    p = lambda x: np.exp(2.0 * np.arctan(x)) / (1.0 + x**2)**2
    Z = np.trapz(p(x_array), x_array)

    plt.hist(X, bins=int(math.sqrt(len(X))), density=True)
    plt.plot(x_array, p(x_array) / Z)
    plt.xlabel(r'$x$')
    plt.show()


if __name__ == '__main__':
    plotInvariantDistribution()