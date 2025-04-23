import numpy as np
import numpy.linalg as lg
import scipy.sparse.linalg as slg
import scipy.optimize as opt
import matplotlib.pyplot as plt

L = 500
N = 1001
x_array = np.linspace(-L, L, N)
dx = 2.0 * L / (N - 1)

D_B = 440.0
chi_0 = 3800.0
B_h = 6.8
K_i = 1.0

# --- Source A(x): define as a function ---
def A_func(x):
    return np.exp(-x**2 / (2 * 50**2))

# step function
def step(B, dt):
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

    # Neumann BCs
    B[0] = B[1]
    B[-1] = B[-2]

    # Clamp negative values and renormalize
    B = np.maximum(B, 0)
    B /= np.trapz(B, x_array)

    return B

def timestepper(B, dt, T, verbose=False):
    n_steps = int(T / dt)
    for n in range(n_steps):
        if verbose and n % 1000 == 0:
            print('t =', n * dt)
        B = step(B, dt)
    return B

def psi(B0, dt, T):
    return B0 - timestepper(B0, dt, T)

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
    plt.title('1D Drift-Diffusion with Chemotactic Drift')
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
        return (psi(B_ss + epsilon * v, dt, T_psi) - psi(B_ss, dt, T_psi)) / epsilon
    Dpsi = slg.LinearOperator((N, N), matvec=Dpsi_v, dtype=np.float64)

    # Compute the leading eigenvalues using Arnoldi
    k = 10
    print('\nComputing Eigenvalues...')
    eigenvalues, _ = slg.eigs(Dpsi, k=k, which='SM', return_eigenvectors=True)
    eigenvalues = 1.0 - eigenvalues # Mapping from psi to timestepper

    # Compute the eigenvalues using the QR method
    dpsi_mat = np.zeros((N,N))
    for n in range(N):
        dpsi_mat[:,n] = Dpsi_v(np.eye(N)[:,n])
    eigenvalues_qr = np.flip(np.sort(1.0 - lg.eigvals(dpsi_mat)))[0:10]

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
    plt.title('1D Drift-Diffusion with Chemotactic Drift')
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
    elif args.experiment == 'arnoldi':
        arnoldi()
    else:
        print("This experiment is not supported. Choose either 'evolution', 'steady-state' or 'arnoldi'.")