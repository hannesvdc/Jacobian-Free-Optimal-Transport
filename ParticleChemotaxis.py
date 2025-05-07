import numpy as np
import numpy.random as rd
import numpy.linalg as lg
import scipy.sparse.linalg as slg
import scipy.optimize as opt
import matplotlib.pyplot as plt

import concurrent
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import as_completed
import pandas as pd

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

def psi(X0, S, dS, chi, D, dt, T, verbose=False):
    return X0 - timestepper(X0, S, dS, chi, D, dt, T, verbose=verbose)

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

def steadyState():
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
    T_psi = 1.0
    rdiff = 10.0 / np.sqrt(N)
    print('rdiff', rdiff)
    F = lambda mu: psi(mu, S, dS, chi, D, dt, T_psi, verbose=True)
    X_ss = opt.newton_krylov(F, X0, rdiff=rdiff, f_tol=1.e-12, verbose=True)

    # Analytic Steady-State for the given chi(S)
    x_array = np.linspace(-L, L, 1000)
    dist = np.exp( (S(x_array) + S(x_array)**3 / 6.0) / D)
    Z = np.trapz(dist, x_array)
    dist = dist / Z

     # Plot final distribution
    plt.hist(X_ss, density=True, bins=int(np.sqrt(N)), label='Newton-Krylov')
    plt.plot(x_array, dist, linestyle='--', label='Analytic Steady State')
    plt.xlabel('x')
    plt.ylabel(r'$\mu(x)$')
    plt.title('1D Drift-Diffusion with Simple Chemotactic Drift')
    plt.grid(True)
    plt.legend()
    plt.show()

def steadyStateBash():
    # Physical functions defining the problem
    S = lambda x: np.tanh(x)
    dS = lambda x: 1.0 / np.cosh(x)**2
    chi = lambda s: 1 + 0.5 * s**2
    D = 0.1
    dt = 1.e-3
    T_psi = 1.0
    def calculateSteadyState(fd_eps, N):
        X0 = X0 = rng.normal(0.0, 1.0, size=N)
        F = lambda mu: psi(mu, S, dS, chi, D, dt, T_psi, verbose=True)
        try:
            X_ss = opt.newton_krylov(F, X0, rdiff=fd_eps, f_tol=1.e-12, verbose=True)
        except opt.NoConvergence as e:
            X_ss = e.args[0]
        F_value = lg.norm(F(X_ss))
        return {'fd_eps': fd_eps, 'N': N, 'F_value': F_value}
    
    # Define parameter grid
    fd_eps_values = [1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5]
    N_values = [10**3, 10**4, 10**5, 10**6]
    param_combinations = [(eps, N) for eps in fd_eps_values for N in N_values]
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks
        future_to_params = {executor.submit(calculateSteadyState, a, b): (a, b) for a, b in param_combinations}
        
        for future in as_completed(future_to_params):
            data = future.result()
            results.append( data)

    # Create pandas dataframe
    df = pd.DataFrame(results)
    df.to_csv("particle_chemotaxis_N_fdeps.csv", index=False)

def parseArguments():
    import argparse
    parser = argparse.ArgumentParser(description="Run the Bimodal PDE simulation.")
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        dest='experiment',
        help="Specify the experiment to run (e.g., 'timeEvolution', 'steady-state', 'arnoldi')."
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArguments()
    if args.experiment == 'evolution':
        timeEvolution()
    elif args.experiment == 'steady-state':
        steadyState()
    elif args.experiment == 'steady-state-bash':
        steadyStateBash()
    else:
        print("This experiment is not supported. Choose either 'evolution', 'steady-state' or 'arnoldi'.")