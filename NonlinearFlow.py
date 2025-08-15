import math
import numpy as np
import matplotlib.pyplot as plt

b = lambda x: 1.0 + x**2
sigma = lambda x: 1.0 + x**2

def particleTimestepper(X, h, rng):
    return X + b(X) * h + sigma(X) * math.sqrt(h) * rng.normal(0.0, 1.0, len(X))

# Assumes X is sorted!
def velocityField(X, h, rng):
    Y = particleTimestepper(X, h, rng)
    Y = np.sort(Y)

    return (Y - X) / h

# Assumes X is sorted!
def averagedVelocityField(X, hmax, hmin, rng):
    h_values = np.logspace(np.log10(hmin), np.log10(hmax), num=10, base=10.0)
    velocity_field = np.zeros_like(X)
    for h in h_values:
        v_h = velocityField(X, h, rng)
        velocity_field += v_h

    return velocity_field / len(h_values)

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

def estimatePotentialEnergy():
    xmin = -4.0
    xmax = 8.0

    # Construct the basis functions at collocation points
    sigma = 0.3 
    pad = 3.0 * sigma

    M_main = 61
    collocation_points = np.linspace(xmin - pad, xmax + pad, M_main)
    collocation_points = np.concatenate([collocation_points, np.linspace(6.0, 8.6, 15)])  # densify right tail
    basis_functions = []
    nabla_basis_functions = []
    for p in collocation_points:
        rbf = lambda x, loc=p: np.exp(-(x - loc)**2 / (2.0 * sigma**2))
        nabla_rbf = lambda x, loc=p: -(x - loc) / sigma**2 * np.exp(-(x - loc)**2 / (2.0 * sigma**2))
        basis_functions.append(rbf)
        nabla_basis_functions.append(nabla_rbf)
    psi = lambda x: np.array([bf(x) for bf in basis_functions])
    dpsi = lambda x: np.array([dbf(x) for dbf in nabla_basis_functions])

    # Generate a uniform particle distribution
    N = 1000001
    X0 = np.linspace(xmin, xmax, N) # sorted
    hmin = 1.e-2
    hmax = 1.e-1
    print('X0', X0)

    # Approximate the velocity field in X
    rng = np.random.RandomState()
    v = averagedVelocityField(X0, hmax, hmin, rng)

    # Estimate the coefficients through least-squares analysis
    nabla_psi_vals = dpsi(X0)
    G = nabla_psi_vals @ nabla_psi_vals.T / N
    b = (nabla_psi_vals @ v[:,np.newaxis] / N)[:,0]
    lam = 1e-3 * np.max(np.diag(G))
    theta = np.linalg.solve(G + lam*np.eye(G.shape[0]), b)
    print('Theta values', theta)
    
    # Build the potential energy approximation and exponentiate it.
    U_parametric = lambda x: -np.dot(theta, psi(x))
    dU_parametric = lambda x: -np.dot(theta, dpsi(x))
    U_values = U_parametric(X0)
    U_values -= np.min(U_values)
    print('U values', psi(X0).shape, theta.shape, U_values.shape)

    # Build the invariant distribution and the exact potential
    p = lambda x: 4.0 / np.sinh(np.pi) * np.exp(2.0 * np.arctan(x)) / (1.0 + x**2)**2
    dp = lambda x: np.exp(2*np.arctan(x)) * (2 - 4*x) / (1 + x**2)**3
    U = lambda x: -np.log(p(x))
    dU = lambda x: -dp(x) / p(x)

    # Compare the analytic with estimated density
    plt.plot(X0, v, label='Stochastic OT-Estimated Velocity Field')
    plt.plot(X0, -dU_parametric(X0), label='Least-Squares Approximation of OT')
    plt.plot(X0, -dU(X0), label='Potential Energy Gradient')
    plt.xlabel(r'$x$')
    plt.legend()

    plt.figure()
    plt.plot(X0, U(X0), label=r'Effective Potential $U(x)$')
    plt.plot(X0, U_values, linestyle='--', label='Estimated Potential fron Velocity Field')
    plt.xlabel(r'$x$')
    plt.legend()

    plt.figure()
    mu_exact = np.exp(-U(X0))
    Z_exact = np.trapz(mu_exact, X0)
    mu_estimated = np.exp(-U_values)
    Z_estimated = np.trapz(mu_estimated, X0)
    plt.plot(X0, mu_exact / Z_exact, label='Exact Density')
    plt.plot(X0, mu_estimated / Z_estimated, label='Estimated Density')
    plt.xlabel(r'$x$')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    estimatePotentialEnergy()