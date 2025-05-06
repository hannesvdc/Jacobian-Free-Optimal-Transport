import numpy as np
import numpy.random as rd
import numpy.linalg as lg
import scipy.sparse.linalg as slg
import scipy.optimize as opt
import matplotlib.pyplot as plt

L = 10.0
rng = rd.RandomState()

def step(X, S, dS, chi, D, dt):
    # EM Step
    X = X + chi(S(X)) * dS(X) * dt + np.sqrt(2.0 * D * dt) * rng.normal(0.0, 1.0, size=X.size)
    
    # Reflective (Neumann) boundary conditions
    X = np.where(X < -L, 2 * (-L) - X, X)
    X = np.where(X > L, 2 * L - X, X)

    # Return OT of X
    return np.sort(X) 

def timestepper(X, S, dS, chi, D, dt, T, verbose=False):
    n_steps = int(T / dt)
    for n in range(n_steps):
        if verbose and n % 100 == 0:
            print('t =', n * dt)
        X = step(X, S, dS, chi, D, dt)
    return X

def psi(X0, S, dS, chi, D, dt, T):
    return X0 - timestepper(X0, S, dS, chi, D, dt, T)

def timeEvolution():
    # Physical functions defining the problem
    S = lambda x: np.tanh(x)
    dS = lambda x: 1.0 / np.cosh(x)**2
    chi = lambda s: 1 + 0.5 * s**2
    D = 0.1

    # Initial condition - standard normal Gaussian
    N = 10**6
    X0 = rng.normal(0.0, 1.0, size=N)

    # Do timestepping
    dt = 1.e-3
    T = 500.0
    X_inf = timestepper(X0, S, dS, chi, D, dt, T, verbose=True)

    # Analytic Steady-State for the given chi(S)
    x_array = np.linspace(-L, L, 1000)
    dist = np.exp( (S(x_array) + S(x_array)**3 / 6.0) / D)
    Z = np.trapz(dist, x_array)
    dist = dist / Z

    # Plot the particle histogram and compare it to the analytic steady-state
    plt.hist(X_inf, density=True, bins=int(np.sqrt(N)), label='Particles')
    plt.plot(x_array, dist, linestyle='--', label='Analytic Steady State')
    plt.xlabel('x')
    plt.ylabel(r'$\mu(x)$')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    timeEvolution()