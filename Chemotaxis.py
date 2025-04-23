import numpy as np
import matplotlib.pyplot as plt

D = 0.1
L = 10.0
N = 1000
x_array = np.linspace(-L, L, N)
dx = 2.0 * L / (N-1)

def step(mu, S, chi, dt):
    # Midpoints for flux evaluation
    x_half = 0.5 * (x_array[1:] + x_array[:-1])
    S_half = S(x_half)
    chi_half = chi(S_half)

    # Gradients at midpoints
    dmu_dx = (mu[1:] - mu[:-1]) / dx
    dS_dx = (S(x_array[1:]) - S(x_array[:-1])) / dx

    # Flux at interfaces at x_{i + 1/2}
    mu_avg = 0.5 * (mu[1:] + mu[:-1])
    flux = D * dmu_dx - mu_avg * chi_half * dS_dx

    # Divergence of flux at centers
    dflux_dx = np.zeros_like(mu)
    dflux_dx[1:-1] = (flux[1:] - flux[:-1]) / dx

    # Explicit Euler step
    mu[1:-1] += dt * dflux_dx[1:-1]

    # Dirichlet boundary conditions and normalize to a density
    mu[0] = mu[-1] = 0
    mu = np.maximum(mu, 0)
    mu = mu / np.trapz(mu, x_array)

    return mu

def timestepper(mu, S, chi, dt, T):
    n_steps = int(T / dt)
    for n in range(n_steps):
        mu = step(mu, S, chi, dt)
    return mu

def timeEvolution():
    # Physical functions defining the problem
    S = lambda x: np.tanh(x)
    chi = lambda s: 1 + 0.5 * s**2  

    # Initial condition
    mu0 = np.exp(-x_array**2 / 0.5**2)
    mu0 = mu0 / np.trapz(mu0, x_array) 

    # Do timestepping
    dt = 1.e-3
    T = 10.0
    mu_inf = timestepper(mu0, S, chi, dt, T)

    # Analytic Steady-State for the given chi(S)
    dist = np.exp( (S(x_array) + S^3(x_array) / 6.0) / D)
    Z = np.trapz(dist, x_array)
    dist = dist / Z

    # Plot final distribution
    plt.plot(x_array, mu_inf, label='Time Evolution')
    plt.plot(x_array, dist, label='Analytic Steady State')
    plt.xlabel('x')
    plt.ylabel(r'$\mu(x)$')
    plt.title('1D Drift-Diffusion with Chemotactic Drift')
    plt.grid(True)
    plt.legend()
    plt.show()

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
    else:
        print("This experiment is not supported. Choose either 'evolution', 'steady-state' or 'arnoldi'.")