import numpy as np
import matplotlib.pyplot as plt

from Density1DOptimizers import density_newton_krylov, reflected_hmc_from_tabulated_density, kde_1d_fft_neumann

L = 10.0
rng = np.random.RandomState()

def step(X : np.ndarray, S, dS, chi, D, dt) -> np.ndarray:
    # Check initial boundary conditions
    X = np.where(X < -L, 2 * (-L) - X, X)
    X = np.where(X > L, 2 * L - X, X)

    # EM Step
    X = X + chi(S(X)) * dS(X) * dt + np.sqrt(2.0 * D * dt) * rng.normal(0.0, 1.0, X.shape)
    
    # Reflective (Neumann) boundary conditions
    X = np.where(X < -L, 2 * (-L) - X, X)
    X = np.where(X > L, 2 * L - X, X)

    # Return OT of X
    return X

def timestepper(X : np.ndarray, S, dS, chi, D, dt, T, verbose=False) -> np.ndarray:
    n_steps = int(T / dt)
    for n in range(n_steps):
        if verbose and n % 100 == 0:
            print('t =', n * dt)
        X = step(X, S, dS, chi, D, dt)
    return X

def timeEvolution():
    # Physical functions defining the problem. 
    S = lambda x: np.tanh(x)
    dS = lambda x: 1.0 / np.cosh(x)**2
    chi = lambda s: 1 + 0.5 * s**2
    D = 0.1

    # Initial density: use a truncated Gaussian for now
    n_points = 101
    grid = np.linspace(-L, L, n_points)
    mean = 5.0
    stdev = 2.0
    mu0 = np.exp(-(grid - mean)**2 / (2.0 * stdev**2))
    mu0 /= (mu0.sum() * (grid[1] - grid[0]))   # renormalise, ∫μ0=1

    # Build the density-to-density timestepper
    N = 10**5
    mcmc_step_size = 2.0
    kde_bw = 0.25
    dt = 1.e-3
    T_psi = 1.0
    def dtod_timestepper(mu):
        particles = reflected_hmc_from_tabulated_density(grid, mu, N, mcmc_step_size, rng)
        new_particles = timestepper(particles, S, dS, chi, D, dt, T_psi)
        return kde_1d_fft_neumann(new_particles, grid, kde_bw)
    
    # Do timestepping up to 300 seconds
    T = 300.0
    n_steps = int(T / T_psi)
    mu = np.copy(mu0)
    for n in range(n_steps):
        print('t =', n*T_psi)
        mu = dtod_timestepper(mu)
    print('t =', T)

    # Plot the initial and final density, as well as the true steady-state distribution
    x_array = np.linspace(-L, L, 1001)
    analytic_ss = np.exp( (S(x_array) + S(x_array)**3 / 6.0) / D)
    Z = np.trapz(analytic_ss, x_array)
    plt.plot(grid, mu0, label='Initial Density')
    plt.plot(grid, mu, label='Time-Evolution Density')
    plt.plot(x_array, analytic_ss / Z, label='Analytic Steady-State')
    plt.legend()
    plt.grid()
    plt.show()


def calculateSteadyState():
    # Physical functions defining the problem. 
    S = lambda x: np.tanh(x)
    dS = lambda x: 1.0 / np.cosh(x)**2
    chi = lambda s: 1 + 0.5 * s**2
    D = 0.1

    # Initial density: use a truncated Gaussian for now
    n_points = 101
    grid = np.linspace(-L, L, n_points)
    mean = 5.0
    stdev = 2.0
    mu0 = np.exp(-(grid - mean)**2 / (2.0 * stdev**2))
    mu0 /= (mu0.sum() * (grid[1] - grid[0]))   # renormalise, ∫μ0=1

    # Build a wrapper around the particle time stepper
    dt = 1.e-3
    T_psi = 1.0
    particle_timestepper = lambda X: timestepper(X, S, dS, chi, D, dt, T_psi)

    # Newton-Krylov optimzer with parameters
    N = 10**5
    mcmc_step_size = 2.0
    kde_bw = 0.25

    # Do Newton-Krylov optmization
    maxiter = 50
    rdiff = 1.e-3
    mu_inf, losses = density_newton_krylov(mu0, grid, particle_timestepper, maxiter, rdiff, N, mcmc_step_size, kde_bw, store_directory=None)

    # Plot the initial and final density, as well as the true steady-state distribution
    x_array = np.linspace(-L, L, 1001)
    analytic_ss = np.exp( (S(x_array) + S(x_array)**3 / 6.0) / D)
    Z = np.trapz(analytic_ss, x_array)
    plt.plot(grid, mu0, label='Initial Density')
    plt.plot(grid, mu_inf, label='Newton-Krylov Density')
    plt.plot(x_array, analytic_ss / Z, label='Analytic Steady-State')
    plt.legend()
    plt.grid()
    plt.show()

def parseArguments():
    import argparse
    parser = argparse.ArgumentParser(description="Run the Bimodal PDE simulation.")
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        dest='experiment',
        help="Specify the experiment to run (e.g., 'evolution', 'test', or 'steady-state')."
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArguments()

    if args.experiment == 'evolution':
        timeEvolution()
    elif args.experiment == 'steady-state':
        calculateSteadyState()
