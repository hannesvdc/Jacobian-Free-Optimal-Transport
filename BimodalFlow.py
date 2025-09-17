import numpy as np
import matplotlib.pyplot as plt

U = lambda x: 0.5 * (x**2 - 1.0)**2
dU = lambda x: 2.0 * (x**2 - 1.0) * x

def particleTimestepper(X, h, beta, rng):
    return X - h * dU(X) + np.sqrt(2.0 / beta * h) * rng.normal(0.0, 1.0, len(X))

# Assumes X is sorted!
def velocityField(X, h, beta, rng):
    Y = particleTimestepper(X, h, beta, rng)
    Y = np.sort(Y)

    return (Y - X) / h

# Assumes X is sorted!
def averagedVelocityField(X, hmax, hmin, beta, rng):
    h_values = np.logspace(np.log10(hmin), np.log10(hmax), num=10, base=10.0)
    velocity_field = np.zeros_like(X)
    for h in h_values:
        v_h = velocityField(X, h, beta, rng)
        velocity_field += v_h

    return velocity_field / len(h_values)

def estimateBimodalPotential():
    xmin = -4.0
    xmax = 4.0

    # Construct the basis functions at collocation points
    sigma = 0.2
    basis_functions = []#[lambda x: x**2, lambda x: x]
    nabla_basis_functions = []#[lambda x: 2*x, lambda x: 1.0 * np.ones_like(x)]
    collocation_points = np.linspace(xmin, xmax, 51)
    for p in collocation_points:
        rbf = lambda x, loc=p: np.exp(-(x - loc)**2 / (2.0 * sigma**2))
        nabla_rbf = lambda x, loc=p: -(x - loc) / sigma**2 * np.exp(-(x - loc)**2 / (2.0 * sigma**2))
        basis_functions.append(rbf)
        nabla_basis_functions.append(nabla_rbf)
    psi = lambda x: np.array([bf(x) for bf in basis_functions])
    dpsi = lambda x: np.array([dbf(x) for dbf in nabla_basis_functions])

    # Generate a uniform particle distribution
    N = 10000001
    X0 = np.linspace(xmin, xmax, N) # sorted
    hmin = 1.e-3
    hmax = 1.e-2
    beta = 1.e0

    # Approximate the velocity field in X
    rng = np.random.RandomState()
    v = averagedVelocityField(X0, hmax, hmin, beta, rng)

    # try one x-value
    x_try = -1.5
    print(psi(x_try), dpsi(x_try))

    # Estimate the coefficients through least-squares analysis
    nabla_psi_vals = dpsi(X0)
    G = nabla_psi_vals @ nabla_psi_vals.T / N
    b = (nabla_psi_vals @ v[:,np.newaxis] / N)[:,0]
    theta = np.linalg.solve(G, b)
    print('Theta values', theta)
    
    # Build the potential energy approximation and exponentiate it.
    U_parametric = lambda x: -np.dot(theta, psi(x))
    dU_parametric = lambda x: -np.dot(theta, dpsi(x))
    U_values = U_parametric(X0)
    U_values -= np.min(U_values)
    print('U values', psi(X0).shape, theta.shape, U_values.shape)

    # Compare the analytic with estimated density
    plt.plot(X0, v, label='Euler-OT Velocity Field')
    #plt.plot(X0, -dU_parametric(X0), label='Least-Squares Approximation of OT')
    plt.plot(X0, -dU(X0), linestyle='--', label='Analytic Velocity Field')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$v(x)$')
    plt.legend()

    plt.figure()
    plt.plot(X0, U(X0), label='Exact Potential')
    plt.plot(X0, U_values, linestyle='--', label='Estimated Potential')
    plt.xlabel(r'$x$')
    plt.legend()

    plt.figure()
    mu_exact = np.exp(-beta * U(X0))
    Z_exact = np.trapz(mu_exact, X0)
    mu_estimated = np.exp(-beta * U_values)
    Z_estimated = np.trapz(mu_estimated, X0)
    plt.plot(X0, mu_estimated / Z_estimated, label='Euler-OT Density')
    plt.plot(X0, mu_exact / Z_exact, linestyle='--', label='Exact Density')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\mu(x)$')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    estimateBimodalPotential()