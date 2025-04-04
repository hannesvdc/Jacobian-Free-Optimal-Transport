import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import scipy.optimize as opt

from OTFramework import *

def eulerOTTimestepper(X, h, mu, sigma, rng):
    Xnew = X + mu(X) * h +  np.sqrt(2.0 * h) * sigma(X) * rng.normal(0.0, 1.0, size=X.shape)
    return np.sort(Xnew)

def eulerOTpsi(X0, h, Tpsi, mu, sigma, rng):
    X = np.copy(X0)
    for _ in range(int(Tpsi/h)):
        X = eulerOTTimestepper(X, h, mu, sigma, rng)

    return X0 - X

def OUTimeEvolution():
    # Simple Ornstein-Uhlenbeck example
    N = 1000000
    mean = 2.0
    stdev = 2.0
    rng = rd.RandomState()
    X = rng.normal(mean, stdev, N)

    # define drift and diffusion
    mu = lambda x: -x
    sigma = lambda x: 1.0
 
    # Do 10 Euler-OT time steps
    h = 0.01
    T = 10.0
    n_steps = int(T / h)

    # Lists to store evolution of moments
    means = np.zeros(n_steps + 1); means[0] = np.mean(X)
    stdevs = np.zeros(n_steps + 1); stdevs[0] = np.std(X)
    for n in range(n_steps):
        print('t =', n*h)
        X = eulerOTTimestepper(X, h, mu, sigma, rng)

        means[n+1] = np.mean(X)
        stdevs[n+1] = np.std(X)

    # Plot the simulation results
    plt.hist(X, bins=int(np.sqrt(N)), density=True)
    plt.xlabel(r'$x$')
    plt.title('Steady-state Time-Evolution')

    plt.figure()
    plt.plot(np.linspace(0.0, T, n_steps+1), means, label=r'$\mathbb{E}[X(t)]$')
    plt.plot(np.linspace(0.0, T, n_steps+1), stdevs, label=r'$\mathbb{V}[X(t)]$')
    plt.xlabel(r'$t$')
    plt.title('Steady-state Time-Evolution')
    plt.legend()
    plt.show()

def OUSteadyState():
    # Sample the (random?) initial condtion
    N = 1000000
    mean = 1.0
    stdev = 1.0
    rng = rd.RandomState()
    X0 = np.sort(rng.normal(mean, stdev, size=N))

    # define drift and diffusion
    mu = lambda x: -x
    sigma = lambda x: 1.0
    h = 0.01
    Tpsi = 0.1
    rdiff = 10000

    # Define Newton-Krylov parameters
    print('Starting Newton-Krylov...')
    f = lambda x: eulerOTpsi(x, h, Tpsi, mu, sigma, rng)
    try:
        X_ss = opt.newton_krylov(f, X0, verbose=True, rdiff=rdiff, maxiter=100, line_search=None, method='gmres')
    except opt.NoConvergence as e:
        X_ss = e.args[0]

    # Plot the steady-state histogran
    print('Steady-State Mean', np.mean(X_ss))
    print('Steady-State Variance', np.var(X_ss))
    plt.hist(X_ss, bins=int(np.sqrt(N)), density=True)
    plt.xlabel(r'$x$')
    plt.title('Steady-state Distribution Newton-Krylov')
    plt.show()

if __name__ == '__main__':
    OUSteadyState()