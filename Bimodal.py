import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import scipy.optimize as opt

V = lambda x: 0.5* (x**2 - 1)**2
mu = lambda x: - 2 * (x**2 - 1) * x
sigma = lambda x: 1.0

def EMOTTimestepper(X, h, mu, sigma, rng):
    Xnew = X + mu(X) * h +  np.sqrt(2.0 * h) * sigma(X) * rng.normal(0.0, 1.0, size=X.shape)
    return np.sort(Xnew)

def EMOTpsi(X0, h, Tpsi, mu, sigma, rng):
    X = np.copy(X0)
    for _ in range(int(Tpsi/h)):
        X = EMOTTimestepper(X, h, mu, sigma, rng)

    return X0 - X

def timeEvolution():
    N = 1000000
    mean = 2.0
    stdev = 2.0
    rng = rd.RandomState()
    X = rng.normal(mean, stdev, N)
 
    # Do 10 Euler-OT time steps
    h = 0.001
    T = 10.0
    n_steps = int(T / h)

    # Lists to store evolution of moments
    for n in range(n_steps):
        print('t =', n*h)
        X = EMOTTimestepper(X, h, mu, sigma, rng)

    # Plot the simulation results
    plt.hist(X, bins=int(np.sqrt(N)), density=True)
    plt.xlabel(r'$x$')
    plt.title('Steady-state Time-Evolution')
    plt.show()

def steadyState():
    N = 1000000
    mean = 1.0
    stdev = 1.0
    rng = rd.RandomState()
    X0 = np.sort(rng.normal(mean, stdev, size=N))

    # define drift and diffusion
    h = 0.001
    Tpsi = 0.1
    rdiff = 1000

    # Define Newton-Krylov parameters
    print('Starting Newton-Krylov...')
    f = lambda x: EMOTpsi(x, h, Tpsi, mu, sigma, rng)
    try:
        X_ss = opt.newton_krylov(f, X0, verbose=True, rdiff=rdiff, maxiter=20, line_search=None, method='gmres')
    except opt.NoConvergence as e:
        X_ss = e.args[0]

    # Plot the steady-state histogran
    plt.hist(X_ss, bins=int(np.sqrt(N)), density=True)
    plt.xlabel(r'$x$')
    plt.title('Steady-state Distribution Newton-Krylov')
    plt.show()

if __name__ == '__main__':
    steadyState()