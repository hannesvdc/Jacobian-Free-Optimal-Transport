import numpy as np
import numpy.random as rd
import numpy.linalg as lg
import scipy.sparse.linalg as slg
import scipy.optimize as opt
import matplotlib.pyplot as plt

L = 500
N = 1001
x_array = np.linspace(-L, L, N)
dx = 2.0 * L / (N - 1)

rng = rd.RandomState()

D_B = 440.0
chi_0 = 3800.0
B_h = 6.8
K_i = 1.0
def A_func(x):
    return np.exp(-x**2 / (2 * 50**2))

# step function
def step(B, dt, eps=None):
    # Source Term A
    S = np.log(1.0 + A_func(x_array) / K_i)
    dS_half = (S[1:] - S[:-1]) / dx

    # The solution B in the midpoints
    B_half = 0.5 * (B[:-1] + B[1:]) # Size (1000,) = number of midpoints

    # Compute the fluxes in the midpoints (size 1000)
    flux_half = chi_0 * (1.0 + B_half / B_h)**(-1) * B_half * dS_half

    # Full advection term in the 999 interior points
    adv = (flux_half[1:] - flux_half[:-1]) / dx

    # Diffusion term (centered second derivative)
    d2B_dx2 = np.zeros_like(B)
    d2B_dx2[1:-1] = (B[2:] - 2 * B[1:-1] + B[:-2]) / dx**2
    diffusion = D_B * d2B_dx2[1:-1]

    # Combine and update
    dB_dt = diffusion - adv
    B[1:-1] += dt * dB_dt

    # Add noise when eps != None
    if eps is not None:
        noise = np.sqrt(dt) * eps * rng.normal(0.0, scale=1.0, size=B.size)
        noise -= np.mean(noise)
        B = B + B * noise # Scale noise with B to get rid of noise domination

    # Neumann BCs
    B[0] = B[1]
    B[-1] = B[-2]

    # Clamp negative values and renormalize
    B = np.maximum(B, 0)
    B /= np.trapz(B, x_array)

    return B

def timestepper(B, dt, T, verbose=False, eps=None):
    n_steps = int(T / dt)
    for n in range(n_steps):
        if verbose and n % 1000 == 0:
            print('t =', n * dt)
        B = step(B, dt, eps=eps)
    return B

def psi(B0, dt, T, eps=None):
    return B0 - timestepper(B0, dt, T, verbose=False, eps=eps)

def timeEvolution(_return=False):
    # Initial condition for B(x, 0)
    B0 = np.exp(-x_array**2 / (2 * 100**2))
    B0 /= np.trapz(B0, x_array)

    # Do timestepping
    dt = 1.e-3
    T = 1000
    B_inf = timestepper(B0, dt, T, verbose=True)
    if _return:
        return B_inf

    plt.plot(x_array, B_inf, label='Final B(x)')
    plt.plot(x_array, B0, label='B0(x)')
    plt.xlabel('x')
    plt.ylabel('B(x)')
    plt.title('Chemotaxis Model')
    plt.grid(True)
    plt.legend()
    plt.show()

def steadyState(_return=False):
    # Initial condition for B(x, 0)
    B0 = np.exp(-x_array**2 / (2 * 100**2))
    B0 /= np.trapz(B0, x_array)

    # Do Newton-Krylov
    dt = 1.e-3
    T_psi = 1.0
    F = lambda mu: psi(mu, dt, T_psi)
    B_ss = opt.newton_krylov(F, B0, verbose=True)
    if _return:
        return B_ss
    
    # Do timestepping for reference
    B_inf = timeEvolution(_return=True)

    # Plot final distribution
    plt.plot(x_array, B_ss, label='Newton-Krylov')
    plt.plot(x_array, B_inf, linestyle='--', label='Time Evolution')
    plt.xlabel('x')
    plt.ylabel(r'$\mu(x)$')
    plt.title('1D Drift-Diffusion with Advanced Chemotactic Drift')
    plt.grid(True)
    plt.legend()
    plt.show()

def arnoldi():
    B_ss = steadyState(_return=True)

    # Setup the Jacobian matrix
    dt = 1.e-3
    T_psi = 1.0
    epsilon = 1.e-8
    def Dpsi_v(v):
        """
        Compute the matrix-vector product Dpsi * v using finite differences.
        """
        return (timestepper(B_ss + epsilon * v, dt, T_psi) - timestepper(B_ss, dt, T_psi)) / epsilon
    Dpsi = slg.LinearOperator((N, N), matvec=Dpsi_v, dtype=np.float64)

    # Compute the leading eigenvalues using Arnoldi
    k = 10
    print('\nComputing Eigenvalues...')
    eigenvalues, _ = slg.eigs(Dpsi, k=k, which='LM', return_eigenvectors=True)

    # Compute the eigenvalues using the QR method
    dpsi_mat = np.zeros((N,N))
    for n in range(N):
        dpsi_mat[:,n] = Dpsi_v(np.eye(N)[:,n])
    eigenvalues_qr = np.flip(np.sort(lg.eigvals(dpsi_mat)))[0:k]

    # Plot in the complex plane
    plt.scatter(eigenvalues.real, eigenvalues.imag, color='blue', marker='o', label='Eigenvalues Arnoldi')
    plt.scatter(eigenvalues_qr.real, eigenvalues_qr.imag, color='tab:orange', marker='x', label='Eigenvalues QR')
    theta = np.linspace(0, 2 * np.pi, 500)
    unit_circle = np.exp(1j * theta)
    plt.plot(unit_circle.real, unit_circle.imag, color='red', linestyle='--')

    # Add labels and grid
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('1D Drift-Diffusion with Advanced Chemotactic Drift')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

def noiseSteadyState():
    # Initial condition for B(x, 0)
    B0 = np.exp(-x_array**2 / (2 * 100**2))
    B0 /= np.trapz(B0, x_array)

    # Noise Amplitude
    eps_list = [1.e-12, 1.e-11, 1.e-10, 1.e-9, 1.e-8, 1.e-7, 1.e-6, 1.e-5, 1.e-4]
    errors = []

    # Do eps = 0 as ou analytic, exact, solution
    dt = 1.e-3
    T_psi = 1.0
    F = lambda mu: psi(mu, dt, T_psi, eps=None)
    try:
        dist = opt.newton_krylov(F, B0, f_tol=0.0, maxiter=20, verbose=True)
    except opt.NoConvergence as e:
        dist = e.args[0]

    for eps in eps_list:
        print('eps =', eps)

        F = lambda mu: psi(mu, dt, T_psi, eps=eps)
        try:
            B_ss = opt.newton_krylov(F, B0, f_tol=0.0, maxiter=20, verbose=True)
        except opt.NoConvergence as e:
            B_ss = e.args[0]
        B_ss[B_ss <= 0.0] = 1.e-10
        B_ss /= np.trapz(B_ss, x_array)

        # Compute the KL Divergence between p_ss and dist
        kl_div = np.trapz(B_ss * np.log(B_ss / dist), x_array)
        errors.append(kl_div)
        print('KL Divergence:', kl_div)

        # Plot
        plt.plot(x_array, B_ss, label=rf"$\varepsilon = {eps}$")

    # Plot the solution
    plt.xlabel(r'$x$')
    plt.title('Steady-State after 50 Newton-Krylov Iterations')
    plt.legend()

    plt.figure()
    plt.loglog(np.flip(np.array(eps_list)), np.flip(np.array(errors)))
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel('KLD')
    plt.title(r'Kullback-Leibler Divergence versus Noise Level $\varepsilon$')

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
        print("This experiment is not supported. Choose either 'evolution', 'steady-state', 'arnoldi', or 'noise'.")