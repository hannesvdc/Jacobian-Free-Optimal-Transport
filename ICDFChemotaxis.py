import math
import numpy as np
import scipy.optimize as opt
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

from ICDF1DOptimizers import icdf_newton_krylov, icdf_on_percentile_grid, particles_from_icdf

L = 10.0
EPS = 1.e-10

def step(X : np.ndarray, S, dS, chi, D, dt, rng) -> np.ndarray:
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

def particle_timestepper(X : np.ndarray, S, dS, chi, D, dt, T, rng, verbose=False) -> np.ndarray:
    n_steps = int(T / dt)
    for n in range(n_steps):
        if verbose and n % 100 == 0:
            print('t =', n * dt)
        X = step(X, S, dS, chi, D, dt, rng)
    return X

def invertCDF(cdf, cdf_grid, percentile_grid):
    spline = PchipInterpolator(cdf_grid, cdf, extrapolate=True)

    particles = np.zeros_like(percentile_grid)
    for k, p in enumerate(percentile_grid):
        j = np.searchsorted(cdf, p)
        if j == len(cdf_grid):
            print('Index error', j, p, cdf[-1])
        xl, xr = cdf_grid[j-1], cdf_grid[j]

        if j == len(cdf_grid)-1 and spline(xr) < p:
            particles[k] = cdf_grid[-1]
            continue

        # solve F(x) - p = 0 on [xl,xr]
        root = opt.brentq(lambda x: spline(x) - p, xl, xr, xtol=1e-10)
        particles[k] = root

    return particles

def timeEvolution():
    rng = np.random.RandomState()

    # Physical functions defining the problem. 
    S = lambda x: np.tanh(x)
    dS = lambda x: 1.0 / np.cosh(x)**2
    chi = lambda s: 1 + 0.5 * s**2
    D = 0.1

    # Initial density: use a truncated Gaussian for now
    N = 10000
    n_points = 1000
    percentile_grid = (np.arange(n_points) + 0.5) / n_points
    mean = 2.0
    stdev = 2.0
    particles = rng.normal(mean, stdev, N)
    icdf0 = icdf_on_percentile_grid(particles, percentile_grid)

    # Build the density-to-density timestepper
    boundary = ((0.0, -L), (1.0, L))
    N = 10**5
    dt = 1.e-3
    T_psi = 1.0
    def icdf_timestepper(icdf):
        particles = particles_from_icdf(percentile_grid, icdf, N, boundary=boundary)
        new_particles = particle_timestepper(particles, S, dS, chi, D, dt, T_psi, rng)
        icdf_new = icdf_on_percentile_grid(new_particles, percentile_grid)
        return icdf_new
    
    # Do ICDF timestepping up to 500 seconds
    T = 500.0
    n_steps = int(T / T_psi)
    icdf = np.copy(icdf0)
    for n in range(n_steps):
        print('t =', n*T_psi)
        icdf = icdf_timestepper(icdf)
    print('t =', T)
    samples_from_icdf = particles_from_icdf(percentile_grid, icdf, N, boundary)

    # Plot the initial and final density, as well as the true steady-state distribution
    dx = 2.0 * L / 1000
    grid = np.linspace(-L, L, 1001)
    analytic_dist = np.exp( (S(grid) + S(grid)**3 / 6.0) / D)
    Z_dist = np.trapz(analytic_dist, grid)
    analytic_dist /= Z_dist
    analytic_cdf = analytic_dist.cumsum() * dx
    analytic_cdf /= analytic_cdf[-1]
    analytic_icdf = invertCDF(analytic_cdf, grid, percentile_grid)

    # Plot the ICDFs first
    plt.plot(percentile_grid, analytic_icdf, label='Analytic ICDF')
    plt.plot(percentile_grid, icdf, label="ICDF Timestepper")
    plt.xlabel('percentiles')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.hist(samples_from_icdf, density=True, bins=int(math.sqrt(N)), label='Particles from ICDF Timestepper')
    plt.plot(grid, analytic_dist, label='Analytic Steady-State Distribution')
    plt.legend()
    plt.grid()
    plt.show()

def calculateSteadyState():
    rng = np.random.RandomState()

    # Physical functions defining the problem. 
    S = lambda x: np.tanh(x)
    dS = lambda x: 1.0 / np.cosh(x)**2
    chi = lambda s: 1 + 0.5 * s**2
    D = 0.1

    # Initial density: use a truncated Gaussian for now
    N = 10**6
    n_points = 100
    percentile_grid = (np.arange(n_points) + 0.5) / n_points
    mean = 5.0
    stdev = 2.0
    particles = rng.normal(mean, stdev, N)
    icdf0 = icdf_on_percentile_grid(particles, percentile_grid)

    # Build a wrapper around the particle time stepper
    dt = 1.e-3
    T_psi = 1.0
    timestepper = lambda X: particle_timestepper(X, S, dS, chi, D, dt, T_psi, rng)

    # Newton-Krylov optimzer with parameters. All parameter values were tested using time evolution
    boundary = ((0.0, -L), (1.0, L))
    maxiter = 20
    rdiff = 10**(-1)
    icdf_inf, losses = icdf_newton_krylov(icdf0, percentile_grid, timestepper, maxiter, rdiff, N, boundary)
    samples_from_icdf = particles_from_icdf(percentile_grid, icdf_inf, N, boundary)
    print(samples_from_icdf)

    # Plot the initial and final density, as well as the true steady-state distribution
    dx = 2.0 * L / 1000
    grid = np.linspace(-L, L, 1001)
    analytic_dist = np.exp( (S(grid) + S(grid)**3 / 6.0) / D)
    Z_dist = np.trapz(analytic_dist, grid)
    analytic_dist /= Z_dist
    analytic_cdf = analytic_dist.cumsum() * dx
    analytic_cdf /= analytic_cdf[-1]
    analytic_icdf = invertCDF(analytic_cdf, grid, percentile_grid)
    samples_from_analytic_icdf = particles_from_icdf(percentile_grid, analytic_icdf, N, boundary)
    analytic_icdf = np.concatenate(([-L], analytic_icdf, [L]))

    spline = PchipInterpolator(np.concatenate(([0.0], percentile_grid, [1.0])), np.concatenate(([-L], icdf_inf, [L])), extrapolate=False)
    spline_grid = np.linspace(-L, L, 10001)
    spline_values = spline(spline_grid)

    # Plot the ICDFs first
    icdf0 = np.concatenate(([-L], icdf0, [L]))
    icdf_inf = np.concatenate(([-L], icdf_inf, [L]))
    percentile_grid = np.concatenate(([0.0], percentile_grid, [1.0]))
    plt.plot(percentile_grid, icdf0, label='Initial ICDF')
    plt.plot(percentile_grid, analytic_icdf, label='Analytic ICDF')
    plt.plot(percentile_grid, icdf_inf, label="ICDF Timestepper")
    #plt.plot(spline_grid, spline_values, label='Spline Interpolation of ICDF')
    plt.xlabel('percentiles')
    plt.grid()
    plt.legend()

    # Also plot the particles
    plt.figure()
    plt.hist(samples_from_analytic_icdf, density=True, bins=int(math.sqrt(N)), label='Particles from ICDF Timestepper')
    plt.plot(grid, analytic_dist, label='Analytic Steady-State Distribution')
    plt.legend()
    plt.grid()

    # Plot the losses
    plt.figure()
    plt.semilogy(np.arange(len(losses)), losses)
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.title('Newton-Krylov Loss')
    plt.show()

def testICDFSampling():

    # Physical functions defining the problem. 
    S = lambda x: np.tanh(x)
    D = 0.1

    N = 10000
    n_points = 100
    percentile_grid = (np.arange(n_points) + 0.5) / n_points
    boundary = ((0.0, -L), (1.0, L))

    # Calculate the analytic density
    dx = 2.0 * L / 1000
    grid = np.linspace(-L, L, 1001)
    analytic_dist = np.exp( (S(grid) + S(grid)**3 / 6.0) / D)
    Z_dist = np.trapz(analytic_dist, grid)
    analytic_dist /= Z_dist
    analytic_cdf = analytic_dist.cumsum() * dx
    analytic_cdf /= analytic_cdf[-1]
    analytic_icdf = invertCDF(analytic_cdf, grid, percentile_grid)
    particles_from_analytic_icdf = particles_from_icdf(percentile_grid, analytic_icdf, N, boundary)
    analytic_icdf = np.concatenate(([-L], analytic_icdf, [L]))

    # Plot the ICDFs first
    percentile_grid = np.concatenate(([boundary[0][0]], percentile_grid, [boundary[1][0]]))
    plt.plot(percentile_grid, analytic_icdf, label='Analytic ICDF')
    plt.xlabel('percentiles')
    plt.title('Sampling Test')
    plt.grid()
    plt.legend()

    # Also plot the particles
    plt.figure()
    plt.hist(particles_from_analytic_icdf, density=True, bins=int(math.sqrt(N)), label='Particles from ICDF Timestepper')
    plt.plot(grid, analytic_dist, label='Analytic Steady-State Distribution')
    plt.title('Sampling Test')
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
    elif args.experiment == 'test-sampling':
        testICDFSampling()