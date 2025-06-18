import numpy as np
import matplotlib.pyplot as plt

from CDF1DOptimizers import cdf_newton_krylov, empirical_cdf_on_grid, particles_from_cdf

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

def timeEvolution():
    rng = np.random.RandomState()

    # Physical functions defining the problem. 
    S = lambda x: np.tanh(x)
    dS = lambda x: 1.0 / np.cosh(x)**2
    chi = lambda s: 1 + 0.5 * s**2
    D = 0.1

    # Initial density: use a truncated Gaussian for now
    n_points = 1001
    grid = np.linspace(-L, L, n_points)
    dx = 2.0 * L / (n_points - 1)
    mean = 2.0
    stdev = 2.0
    mu_0 = np.exp(-(grid - mean)**2 / (2.0 * stdev**2)) / np.sqrt(2.0 * np.pi * stdev**2)
    Z = np.trapz(mu_0, grid)
    cdf_0 = mu_0.cumsum() * dx / Z
    cdf_0 = cdf_0 / cdf_0[-1] # Rescale to ensure final value is 1
    cdf_0 = np.where(cdf_0 < EPS, 0.0, cdf_0)

    # Build the density-to-density timestepper
    N = 10**5
    dt = 1.e-3
    T_psi = 1.0
    def cdf_timestepper(cdf):
        particles = particles_from_cdf(grid, cdf, N)
        new_particles = particle_timestepper(particles, S, dS, chi, D, dt, T_psi, rng, verbose=False)
        return empirical_cdf_on_grid(new_particles, grid)
    
    # Do timestepping up to 500 seconds
    T = 500.0
    n_steps = int(T / T_psi)
    cdf = np.copy(cdf_0)
    for n in range(n_steps):
        print('t =', n*T_psi)
        cdf = cdf_timestepper(cdf)
    print('t =', T)

    # Plot the initial and final density, as well as the true steady-state distribution
    analytic_dist = np.exp( (S(grid) + S(grid)**3 / 6.0) / D)
    Z_dist = np.trapz(analytic_dist, grid)
    analytic_dist /= Z_dist
    analytic_cdf = analytic_dist.cumsum() * dx
    plt.plot(grid, mu_0, label='Initial Density')
    plt.plot(grid, cdf_0, label='Initial CDF')
    plt.plot(grid, cdf, label='Time-Evolution CDF')
    plt.plot(grid, analytic_cdf, label='Analytic Steady-State CDF')
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
    n_points = 1001
    grid = np.linspace(-L, L, n_points)
    dx = 2.0 * L / (n_points - 1)
    mean = 2.0
    stdev = 2.0
    mu_0 = np.exp(-(grid - mean)**2 / (2.0 * stdev**2)) / np.sqrt(2.0 * np.pi * stdev**2)
    Z = np.trapz(mu_0, grid)
    cdf_0 = mu_0.cumsum() * dx / Z
    cdf_0 = cdf_0 / cdf_0[-1] # Rescale to ensure final value is 1
    cdf_0 = np.where(cdf_0 < EPS, 0.0, cdf_0)

    # Build a wrapper around the particle time stepper
    dt = 1.e-3
    T_psi = 1.0
    timestepper = lambda X: particle_timestepper(X, S, dS, chi, D, dt, T_psi, rng)

    # Newton-Krylov optimzer with parameters. All parameter values were tested using time evolution
    N = 10**5
    maxiter = 100
    rdiff = 10**(-1)
    cdf_inf, losses = cdf_newton_krylov(cdf_0, grid, timestepper, maxiter, rdiff, N)

    # Plot the initial and final density, as well as the true steady-state distribution
    analytic_dist = np.exp( (S(grid) + S(grid)**3 / 6.0) / D)
    Z_dist = np.trapz(analytic_dist, grid)
    analytic_dist /= Z_dist
    analytic_cdf = analytic_dist.cumsum() * dx
    analytic_cdf /= analytic_cdf[-1]
    plt.plot(grid, cdf_0, label='Initial CDF')
    plt.plot(grid, analytic_cdf, label='Analytic Steady-State CDF')
    plt.plot(grid, cdf_inf, linestyle='--', label='Newton-Krylov CDF')
    plt.legend()
    plt.grid()

    # Plot the losses
    plt.figure()
    plt.semilogy(np.arange(len(losses)), losses)
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.title('Newton-Krylov Loss')
    plt.show()

def averageSteadyState(job_id : int):
    rng_job = np.random.RandomState(seed=job_id)

    # Physical functions defining the problem. 
    S = lambda x: np.tanh(x)
    dS = lambda x: 1.0 / np.cosh(x)**2
    chi = lambda s: 1 + 0.5 * s**2
    D = 0.1

    # Initial density: use a truncated Gaussian for now
    n_points = 1001
    grid = np.linspace(-L, L, n_points)
    dx = 2.0 * L / (n_points - 1)
    mean = 2.0
    stdev = 2.0
    mu_0 = np.exp(-(grid - mean)**2 / (2.0 * stdev**2)) / np.sqrt(2.0 * np.pi * stdev**2)
    Z = np.trapz(mu_0, grid)
    cdf_0 = mu_0.cumsum() * dx / Z
    cdf_0 = cdf_0 / cdf_0[-1] # Rescale to ensure final value is 1
    cdf_0 = np.where(cdf_0 < EPS, 0.0, cdf_0)

    # Build a wrapper around the particle time stepper
    dt = 1.e-3
    T_psi = 1.0
    timestepper = lambda X: particle_timestepper(X, S, dS, chi, D, dt, T_psi, rng_job)

    # Newton-Krylov optimzer with parameters. All parameter values were tested using time evolution
    N = 10**5
    maxiter = 100
    rdiff = 10**(-1)
    cdf_inf, losses = cdf_newton_krylov(cdf_0, grid, timestepper, maxiter, rdiff, N)

    # Store the losses for later analysis and plotting
    store_directory = './Results/slurm/'
    filename = f'CDF_1D_NK_job={job_id}.npy'
    np.save(store_directory + filename, np.array(losses))

def plotAverageConvergenceRate():
    store_directory = './Results/slurm/'
    maxiter = 100
    n_jobs = 100
    losses = np.empty((n_jobs, maxiter+1))
    for job_id in range(n_jobs):
        filename = f'CDF_1D_NK_job={job_id}.npy'
        job_losses = np.load(store_directory + filename)
        losses[job_id, :] = job_losses
    losses = losses[:,1:]
    log_losses = np.log(losses)

    # Compute the average losses
    average_losses = np.average(log_losses, axis=0)
    averaged_log_reduction_rates = average_losses[5:61] - average_losses[4:60]
    log_reduction_rate = np.average(averaged_log_reduction_rates)
    reduction_rate = np.exp(log_reduction_rate)
    print('Log Reduction Rate', log_reduction_rate)
    print('Reduction rate', reduction_rate)

    # Plot both
    plt.plot(np.arange(maxiter), average_losses, label='Averaged Log Residual')
    plt.plot(np.arange(maxiter) + 1.0, -0.5 + (np.arange(maxiter) + 1.0) * log_reduction_rate, linestyle='dashed')
    plt.xlabel('Iteration (Epoch)')
    plt.legend()

    # Plot both
    plt.figure()
    plt.semilogy(np.arange(maxiter), np.exp(average_losses), label='Averaged Residual')
    plt.semilogy(np.arange(maxiter) + 1.0, 0.7*reduction_rate**(np.arange(maxiter) + 1.0), linestyle='dashed', label=r'Linear Rate $|\psi(F_{k+1})| \ / \ |\psi(F_k)|$')
    plt.title(r'Newton-Krylov Residual $|\psi(F_k)|$')
    plt.xlabel('Iteration (Epoch)')
    plt.legend()

    # Plot all convergence of psi
    plt.figure()
    plt.plot(np.arange(maxiter), log_losses.transpose(), color='tab:blue')
    plt.title('All Log Residuals')
    plt.xlabel('Iteration (Epoch)')
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
    parser.add_argument(
        '--job_id',
        type=int,
        required=False,
        dest='job_id'
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArguments()

    if args.experiment == 'evolution':
        timeEvolution()
    elif args.experiment == 'steady-state':
        calculateSteadyState()
    elif args.experiment == 'average-steady-state':
        averageSteadyState(args.job_id)
    elif args.experiment == 'plot-averaged-residual':
        plotAverageConvergenceRate()
