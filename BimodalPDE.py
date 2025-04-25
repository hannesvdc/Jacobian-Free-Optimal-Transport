import numpy as np
import numpy.random as rd
import scipy.sparse.linalg as slg
import scipy.optimize as opt
import matplotlib.pyplot as plt

N = 1000
left = -5.0
right = 5.0
x_array = np.linspace(left, right, N)
dx = (right - left) / (N - 1)

rng = rd.RandomState()

def step(p, mu, sigma, dt, eps=None):
    mu_vals = mu(x_array)
    sigma_vals = sigma(x_array)

    # Drift term (upwind)
    mu_p = mu_vals * p
    d_mu_p_dx = np.zeros_like(p)

    # Boolean masks for drift direction
    upwind_right = mu_vals[1:-1] >= 0
    upwind_left = ~upwind_right

    # Backward difference (mu > 0)
    d_mu_p_dx[1:-1][upwind_right] = (
        mu_p[1:-1][upwind_right] - mu_p[:-2][upwind_right]
    ) / dx

    # Forward difference (mu < 0)
    d_mu_p_dx[1:-1][upwind_left] = (
        mu_p[2:][upwind_left] - mu_p[1:-1][upwind_left]
    ) / dx

    # Diffusion term (central difference)
    d2p_dx2 = np.zeros_like(p)
    d2p_dx2[1:-1] = (p[2:] - 2 * p[1:-1] + p[:-2]) / dx**2

    dpdt = -d_mu_p_dx + 0.5 * sigma_vals**2 * d2p_dx2
    p = p + dt * dpdt
    p[0] = p[-1] = 0  # Dirichlet BCs

    # Add noise when eps != None
    if eps is not None:
        noise = np.sqrt(dt) * eps * rng.normal(0.0, scale=1.0, size=p.size)
        noise -= np.mean(noise)
        p = p + p * noise # Scale noise with p to get rid of noise domination

    # Sanitize the output to represent a density
    p = np.maximum(p, 0)
    p /= np.trapz(p, x_array)
    return p

def timestepper(p, mu, sigma, dt, T, eps=None, verbose=False):
    n_steps = int(T / dt)
    for n in range(n_steps):
        if verbose and n % 1000 == 0:
            print('t =', n * dt)
        p = step(p, mu, sigma, dt, eps)
    return p

def psi(p, mu, sigma, dt, T, eps=None):
    return p - timestepper(p, mu, sigma, dt, T, eps)

def timeEvolution():
    beta = 2.0
    V = lambda x: 0.5* (x**2 - 1)**2
    mu = lambda x: - 2 * (x**2 - 1) * x # drift = - \nabla V(x)
    sigma = lambda x: np.sqrt(2.0 / beta)

    # Uniform initial condition
    std = 0.1
    p0 = np.exp( - x_array**2 / (2.0 * std**2) ) / np.sqrt(2.0 * np.pi * std**2)
    p0 /= np.trapz(p0, x_array)
    assert np.abs(np.trapz(p0, x_array) - 1.0) < 1.e-12

    # Time stepping
    dt = 1.e-5
    T = 10.0
    p_inf = timestepper(p0, mu, sigma, dt, T, verbose=True)

    # Plot the solution
    dist = lambda x: np.exp(-beta * V(x))
    Z = np.trapz(dist(x_array), x_array)
    plt.plot(x_array, p_inf, label='Time Evolution')
    plt.plot(x_array, dist(x_array) / Z, label='Exact Distribution')
    plt.xlabel(r'$x$')
    plt.legend()
    plt.show()

def steadyState(_return=False):
    beta = 2.0
    V = lambda x: 0.5* (x**2 - 1)**2
    mu = lambda x: - 2 * (x**2 - 1) * x # drift = - \nabla V(x)
    sigma = lambda x: np.sqrt(2.0 / beta)

    # Uniform initial condition
    std = 0.1
    p0 = np.exp( - x_array**2 / (2.0 * std**2) ) / np.sqrt(2.0 * np.pi * std**2)
    p0 = p0 / np.trapz(p0, x_array)

    # Solve Newton - Krylov
    dt = 1.e-5
    T_psi = 0.1
    F = lambda p: psi(p, mu, sigma, dt, T_psi)
    p_ss = opt.newton_krylov(F, p0, f_tol=1.e-14, verbose=True)
    if _return:
        return p_ss

    # Plot the solution
    dist = lambda x: np.exp(-beta * V(x))
    Z = np.trapz(dist(x_array), x_array)
    plt.plot(x_array, p_ss, label='Newton-Krylov Steady-State')
    plt.plot(x_array, dist(x_array) / Z, label='Exact Distribution')
    plt.xlabel(r'$x$')
    plt.legend()
    plt.show()

def noiseSteadyState():
    beta = 2.0
    V = lambda x: 0.5* (x**2 - 1)**2
    mu = lambda x: - 2 * (x**2 - 1) * x # drift = - \nabla V(x)
    sigma = lambda x: np.sqrt(2.0 / beta)

    # Uniform initial condition
    std = 0.1
    p0 = np.exp( - x_array**2 / (2.0 * std**2) ) / np.sqrt(2.0 * np.pi * std**2)
    p0 = p0 / np.trapz(p0, x_array)

    # Invariant Bimodal Distribution
    dist = lambda x: np.exp(-beta * V(x))
    Z = np.trapz(dist(x_array), x_array)
    dist_vals = dist(x_array) / Z
    plt.plot(x_array, dist_vals, label='Exact Distribution')

    # Noise size
    eps_list = [1.e-4, 1.e-5, 1.e-6, 1.e-7, 1.e-8, 1.e-9, 1.e-10, 1.e-11]
    errors = []
    for eps in eps_list:
        print('eps =', eps)

        # Solve Newton - Krylov
        dt = 1.e-5
        T_psi = 0.1
        F = lambda p: psi(p, mu, sigma, dt, T_psi, eps=eps)
        try:
            p_ss = opt.newton_krylov(F, p0, f_tol=1.e-14, maxiter=50, verbose=True)
        except opt.NoConvergence as e:
            p_ss = e.args[0]
        p_ss[p_ss <= 0.0] = 1.e-5
        p_ss /= np.trapz(p_ss, x_array)

        # Compute the KL Divergence between p_ss and dist
        kl_div = np.trapz(p_ss * np.log(p_ss / dist_vals), x_array)
        errors.append(kl_div)

        # Plot
        plt.plot(x_array, p_ss, label=f"Noise Level {eps}")

    # Plot the solution
    plt.xlabel(r'$x$')
    plt.title('Steady-State after 50 Newton-Krylov Iterations')
    plt.legend()

    plt.figure()
    plt.loglog(eps_list, errors)
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel('KLD')
    plt.title(r'Kullback-Leibler Divergence versus Noise Level $\varepsilon$')

    plt.show()

def arnoldi():
    beta = 2.0
    V = lambda x: 0.5* (x**2 - 1)**2
    mu = lambda x: - 2 * (x**2 - 1) * x # drift = - \nabla V(x)
    sigma = lambda x: np.sqrt(2.0 / beta)
    p_ss = steadyState(_return=True)

    dt = 1.e-5
    T_psi = 0.1
    epsilon = 1.e-8
    def Dpsi_v(v):
        """
        Compute the matrix-vector product Dpsi * v using finite differences.
        """
        return (psi(p_ss + epsilon * v, mu, sigma, dt, T_psi) - psi(p_ss, mu, sigma, dt, T_psi)) / epsilon

    # Define the LinearOperator for Dpsi
    n = len(p_ss)
    Dpsi = slg.LinearOperator((n, n), matvec=Dpsi_v, dtype=np.float64)

    # Compute the leading eigenvalues using eigs
    k = 10
    print('\nComputing Eigenvalues...')
    eigenvalues, _ = slg.eigs(Dpsi, k=k, which='SM', return_eigenvectors=True)
    eigenvalues = 1.0 - eigenvalues # Mapping from psi to timestepper

    # Plot in the complex plane
    plt.scatter(eigenvalues.real, eigenvalues.imag, color='blue', marker='x', label='Eigenvalues')
    theta = np.linspace(0, 2 * np.pi, 500)
    unit_circle = np.exp(1j * theta)
    plt.plot(unit_circle.real, unit_circle.imag, color='red', linestyle='--')

    # Add labels and grid
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Fokker-Planck Eigenvalues')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

def parseArguments():
    import argparse
    parser = argparse.ArgumentParser(description="Run the Bimodal PDE simulation.")
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        dest='experiment',
        help="Specify the experiment to run (e.g., 'timeEvolution', 'steady-state', 'arnoldi', 'noise')."
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArguments()
    if args.experiment == 'evolution':
        timeEvolution()
    elif args.experiment == 'steady-state':
        steadyState()
    elif args.experiment == 'arnoldi':
        arnoldi()
    elif args.experiment == 'noise':
        noiseSteadyState()
    else:
        print("This experiment is not supported. Choose either 'evolution', 'steady-state', 'arnoldi' or 'noise'.")