import math
import numpy as np
import scipy.optimize as opt
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

from ICDF1DOptimizers import icdf_newton_krylov, icdf_on_percentile_grid, particles_from_icdf

L = 4.0
U = lambda x: 0.5 * (x**2 - 1.0)**2
dU = lambda x: 2.0 * (x**2 - 1.0) * x

def oneStep(X, h, beta, rng):
    return X - h * dU(X) + np.sqrt(2.0 / beta * h) * rng.normal(0.0, 1.0, len(X))

def particleTimestepper(X, h, T, beta, rng):
    nsteps = int(T / h)
    for n in range(nsteps):
        X = oneStep(X, h, beta, rng)
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
    beta = 1.0

    # Initial density: use a truncated Gaussian for now
    N = 100000
    n_points = 1000
    percentile_grid = (np.arange(n_points) + 0.5) / n_points
    mean = 0.0
    stdev = 1.0
    particles = rng.normal(mean, stdev, N)
    icdf0 = icdf_on_percentile_grid(particles, percentile_grid)

    # Build the density-to-density timestepper
    boundary = ((0.0, -L), (1.0, L))
    N = 10**5
    dt = 1.e-3
    T_psi = 1.0
    def icdf_timestepper(icdf):
        particles = particles_from_icdf(percentile_grid, icdf, N, boundary=boundary)
        new_particles = particleTimestepper(particles, dt, T_psi, beta, rng)
        icdf_new = icdf_on_percentile_grid(new_particles, percentile_grid)
        return icdf_new
    
    # Do ICDF timestepping up to 500 seconds
    T = 100.0
    n_steps = int(T / T_psi)
    icdf = np.copy(icdf0)
    for n in range(n_steps):
        print('t =', n*T_psi)
        icdf = icdf_timestepper(icdf)
    print('t =', T)
    samples_from_icdf = particles_from_icdf(percentile_grid, icdf, N, boundary)

    # Build the analytic distribution and icdf
    x_grid = np.linspace(-L, L, 1001)
    analytic_dist = np.exp(-beta * U(x_grid))
    analytic_dist /= np.trapz(analytic_dist, x_grid)
    analytic_cdf = np.cumsum(analytic_dist)
    analytic_cdf /= analytic_cdf[-1]
    analytic_icdf = invertCDF(analytic_cdf, x_grid, percentile_grid)

    # Plot the ICDFs first
    plt.plot(percentile_grid, analytic_icdf, label='Analytic ICDF')
    plt.plot(percentile_grid, icdf0, label='Initial ICDF')
    plt.plot(percentile_grid, icdf, '--', label=r"ICDF at time $T=100$")
    plt.xlabel(r'$p$')
    plt.ylabel('ICDF')
    #plt.grid()
    plt.legend()

    plt.figure()
    plt.hist(samples_from_icdf, density=True, bins=int(math.sqrt(N)), label='Particles from Evolved ICDF')
    plt.plot(x_grid, analytic_dist, label='Analytic Steady-State Density')
    plt.xlabel(r"$x$")
    plt.legend()
    #plt.grid()
    plt.show()

def calculateSteadyState():
    rng = np.random.RandomState()
    beta = 1.0

    # Initial density: use a truncated Gaussian for now
    N = 10**6
    n_points = 1000
    percentile_grid = (np.arange(n_points) + 0.5) / n_points
    mean = 0.0
    stdev = 1.0
    particles = rng.normal(mean, stdev, N)
    icdf0 = icdf_on_percentile_grid(particles, percentile_grid)

    # Solve for the invariant ICDF using Newton-Krylov
    dt = 1.e-3
    boundary = ((0.0, -L), (1.0, L))
    rdiff = 1e-1
    T_psi = 1.0
    maxiter = 25
    icdf_inf, losses = icdf_newton_krylov(icdf0, percentile_grid, lambda X : particleTimestepper(X, dt, T_psi, beta, rng), maxiter, rdiff, N, boundary)
    samples_from_icdfinf = particles_from_icdf(percentile_grid, icdf_inf, N, boundary)

    # Build the analytic distribution and icdf
    x_grid = np.linspace(-L, L, 1001)
    analytic_dist = np.exp(-beta * U(x_grid))
    analytic_dist /= np.trapz(analytic_dist, x_grid)
    analytic_cdf = np.cumsum(analytic_dist)
    analytic_cdf /= analytic_cdf[-1]
    analytic_icdf = invertCDF(analytic_cdf, x_grid, percentile_grid)
    
    plot_percentile_grid = np.concatenate(([0.0], percentile_grid, [1.0]))
    plot_analytic_icdf = np.concatenate(([-L], analytic_icdf, [L]))
    plot_icdf_inf = np.concatenate(([-L], icdf_inf, [L]))

    # Plot the ICDFs first
    plt.plot(plot_percentile_grid, plot_analytic_icdf, linestyle='dashdot', label='Analytic ICDF')
    plt.plot(percentile_grid, icdf0, label='Initial ICDF')
    plt.plot(plot_percentile_grid, plot_icdf_inf+0.02, '--', label="ICDF by Newton-Krylov")
    plt.xlabel(r'$p$')
    plt.ylabel('ICDF')
    plt.legend()

    plt.figure()
    plt.hist(samples_from_icdfinf, density=True, bins=int(math.sqrt(N)), label='Particles from Evolved ICDF')
    plt.plot(x_grid, analytic_dist, label='Analytic Steady-State Density')
    plt.xlabel(r"$x$")
    plt.legend()

    # Plot the losses
    plt.figure()
    plt.semilogy(np.arange(len(losses)), losses, label=r'$\Psi\left((F^{-1})^{(k)}\right)$')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.show()
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