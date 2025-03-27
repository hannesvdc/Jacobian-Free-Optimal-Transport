import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import scipy.optimize as opt

from OTFramework import *

def eulerOTTimestepper(X, h, mu, sigma, rng):
    N = X.size
    Xnew = X + mu(X) * h +  np.sqrt(2.0 * h) * sigma(X) * rng.normal(0.0, 1.0, size=(N,1))
    return find_ot_assignment(X, Xnew)

def eulerOTpsi(X, h, mu, sigma, rng):
    return X - eulerOTTimestepper(X, h, mu, sigma, rng)

def OUTimeEvolution():
    # Simple Ornstein-Uhlenbeck example
    N = 1000
    mean = 2.0
    stdev = 2.0
    rng = rd.RandomState()
    X = rng.normal(mean, stdev, size=(N,1))

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
    plt.hist(X[:,0], bins=int(np.sqrt(N)), density=True)
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
    N = 1000
    mean = 2.0
    stdev = 2.0
    rng = rd.RandomState()
    X0 = np.sort(rng.normal(mean, stdev, N))

    # define drift and diffusion
    mu = lambda x: -x
    sigma = lambda x: 1.0
    h = 0.01

    # Define Newton-Krylov parameters
    f = lambda x: eulerOTpsi(np.reshape(x, (N,1)), h, mu, sigma, rng)[:,0]
    try:
        X_ss = opt.newton_krylov(f, X0, verbose=True, maxiter=100)
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
    OUTimeEvolution()