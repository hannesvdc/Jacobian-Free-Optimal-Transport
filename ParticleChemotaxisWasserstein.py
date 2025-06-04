import os
import torch as pt
import math
from geomloss import SamplesLoss
import matplotlib.pyplot as plt

import Wasserstein1DOptimizers as wopt

L = 10.0

# Simple Brownian MCMC sampler
def sampleInvariantMCMC(mu, N):
    x = pt.tensor([0.0])
    dt = 10.0
    samples = []
    n_accepted = 0.0
    for n in range(N):
        y = x + math.sqrt(2.0 * dt) * pt.normal(0.0, 1.0, (1,))
        if y < -L:
            y = -2*L - x
        elif y > L:
            y = 2*L - y
        acc = mu(y) / mu(x)
        if pt.rand((1,)) <= acc:
            n_accepted += 1
            x = y
        samples.append(x[0].item())

    print('Acceptance Rate', n_accepted / N)
    return samples

def step(X, S, dS, chi, D, dt, device, dtype):
    # Check initial boundary conditions
    X = pt.where(X < -L, 2 * (-L) - X, X)
    X = pt.where(X > L, 2 * L - X, X)

    # EM Step
    X = X + chi(S(X)) * dS(X) * dt + math.sqrt(2.0 * D * dt) * pt.normal(0.0, 1.0, X.shape, device=device, dtype=dtype)
    
    # Reflective (Neumann) boundary conditions
    X = pt.where(X < -L, 2 * (-L) - X, X)
    X = pt.where(X > L, 2 * L - X, X)

    # Return OT of X
    return X

def timestepper(X, S, dS, chi, D, dt, T, device, dtype, verbose=False):
    n_steps = int(T / dt)
    for n in range(n_steps):
        if verbose and n % 100 == 0:
            print('t =', n * dt)
        X = step(X, S, dS, chi, D, dt, device, dtype)
    return X

def timeEvolution():
    device = pt.device("mps")
    dtype = pt.float32

    # Physical functions defining the problem
    S = lambda x: pt.tanh(x)
    dS = lambda x: 1.0 / pt.cosh(x)**2
    chi = lambda s: 1 + 0.5 * s**2
    D = 0.1

    # Initial condition - standard normal Gaussian
    N = 10**6
    X0 = pt.normal(0.0, 1.0, (N,1), device=device, dtype=dtype, requires_grad=False)

    # Do timestepping
    dt = 1.e-3
    T = 500.0
    X_inf = timestepper(X0, S, dS, chi, D, dt, T, device, dtype, verbose=True)

    # Analytic Steady-State for the given chi(S)
    x_array = pt.linspace(-L, L, 1000)
    dist = pt.exp( (S(x_array) + S(x_array)**3 / 6.0) / D)
    Z = pt.trapz(dist, x_array)
    dist = dist / Z

    # Plot the particle histogram and compare it to the analytic steady-state
    plt.hist(X_inf.cpu().numpy(), density=True, bins=int(math.sqrt(N)), label='Particles')
    plt.plot(x_array.numpy(), dist.numpy(), linestyle='--', label='Analytic Steady State')
    plt.xlabel('x')
    plt.ylabel(r'$\mu(x)$')
    plt.grid(True)
    plt.legend()
    plt.show()

def test_w2_helpers():
    """Compute ½ W₂² and grad for one mini-batch and print diagnostics."""
    device = pt.device("mps")
    dtype = pt.float32

    # Physical functions defining the problem
    S = lambda x: pt.tanh(x)
    dS = lambda x: 1.0 / pt.cosh(x)**2
    chi = lambda s: 1 + 0.5 * s**2
    D = 0.1
    dist = lambda x: pt.exp( (S(x) + S(x)**3 / 6.0) / D)

    # Initial distribution of particles (standard normal Gaussian)
    N = 10**4
    X0 = sampleInvariantMCMC(dist, N)

    # Biuld the timestepper
    dt = 1.e-3
    T_psi = 1.0
    stepper = lambda X: timestepper(X, S, dS, chi, D, dt, T_psi, device=device, dtype=dtype)

    # loss only
    batch_size = N
    loss_val = wopt._call_loss(X0, stepper, batch_size, device)
    print(f"½ W₂²  (call_loss)     : {loss_val.item():.3e}")

    # Plot histogram of the push-forward versus X
    with pt.no_grad():
        y_plot = stepper(X0)
        plt.hist(X0.cpu().numpy(), bins=80, alpha=0.5, label="X(t)", density=True)
        plt.hist(y_plot.cpu().numpy(), bins=80, alpha=0.5, label="φ_T(X)", density=True)
        plt.xlim((-10, 10))
        plt.title("Mini-batch before vs. after φ_T")
        plt.legend()
        plt.show()

def calculateSteadyState():
    device = pt.device("mps")
    dtype = pt.float32
    store_directory = "./Results/"

    # Physical functions defining the problem
    S = lambda x: pt.tanh(x)
    dS = lambda x: 1.0 / pt.cosh(x)**2
    chi = lambda s: 1 + 0.5 * s**2
    D = 0.1

    # Initial condition - Gaussian (mean 5, stdev 2)
    N = 10**4
    X0 = pt.normal(5.0, 2.0, (N,1), device=device, dtype=dtype, requires_grad=False)
    X0 = pt.where(X0 < -L, 2 * (-L) - X0, X0)
    X0 = pt.where(X0 > L, 2 * L - X0, X0)

    # Build the timestepper function
    dt = 1.e-3
    T_psi = 1.0
    stepper = lambda X: timestepper(X, S, dS, chi, D, dt, T_psi, device=device, dtype=dtype)

    # Do optimization to find the steady-state particles
    batch_size = N
    lr = 1.e-1
    epochs = 15000
    X_inf, losses, grad_norms = wopt.wasserstein_adam(X0, stepper, epochs, batch_size, lr, device, store_directory=store_directory)

    # Analytic Steady-State for the given chi(S)
    x_array = pt.linspace(-L, L, 1000)
    dist = pt.exp( (S(x_array) + S(x_array)**3 / 6.0) / D)
    Z = pt.trapz(dist, x_array)
    dist = dist / Z

    # Plot the loss as a function of the batch / epoch number as well as a histogram of the final particles
    batch_counter = pt.linspace(0.0, epochs, len(losses))
    plt.semilogy(batch_counter, losses, label='Wasserstein Loss')
    plt.semilogy(batch_counter, grad_norms, label='Wasserstein Loss Gradient')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.hist(X_inf.numpy(), density=True, bins=int(math.sqrt(N)), label='Adam Particles')
    plt.plot(x_array.numpy(), dist.numpy(), linestyle='--', label='Analytic Steady State')
    plt.xlabel('x')
    plt.ylabel(r'$\mu(x)$')
    plt.grid(True)
    plt.legend()
    
    plt.show()

def plotSteadyState():
    store_directory = "./Results/"
    filename = os.path.join(store_directory, "particles_wasserstein_adam.pt")
    particles = pt.load(filename)
    N = len(particles)

    losses_filename = os.path.join(store_directory, "wasserstein_adam_losses.pt")
    losses_and_grads = pt.load(losses_filename)
    if losses_and_grads.shape[0] < losses_and_grads.shape[1]:
        losses = losses_and_grads[0,:]
        grad_norms = losses_and_grads[1,:]
    else:
        losses = losses_and_grads[:,0]
        grad_norms = losses_and_grads[:,1]
    epochs = len(losses)

    # Get Samples from the Initial condition - Gaussian (mean 5, stdev 2) - for Plotting Purposes
    N = 10**4
    X0 = pt.normal(5.0, 2.0, (N,1), requires_grad=False)
    X0 = pt.where(X0 < -L, 2 * (-L) - X0, X0)
    X0 = pt.where(X0 > L, 2 * L - X0, X0)

    # Analytic Steady-State for the given chi(S)
    S = lambda x: pt.tanh(x)
    D = 0.1
    x_array = pt.linspace(-L, L, 1000)
    dist = pt.exp( (S(x_array) + S(x_array)**3 / 6.0) / D)
    Z = pt.trapz(dist, x_array)
    dist = dist / Z

    # Plot the loss as a function of the batch / epoch number as well as a histogram of the final particles
    batch_counter = pt.linspace(0.0, epochs, len(losses))
    plt.semilogy(batch_counter, losses, label=r'$\frac{1}{2}W_2^2(X, \phi_T(X))$')
    plt.semilogy(batch_counter, grad_norms, label=r'$\nabla_X \frac{1}{2}W_2^2(X, \phi_T(X))$')
    milestones = [2_500, 5_000, 7_500, 10_000, 12_500]
    lrs = [1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6]
    y_top = losses.max().item()          # highest point on the log plot
    y_txt = y_top * 0.8   
    for m, lr in zip(milestones, lrs):
        plt.axvline(m, color='gray', linewidth=5, ls='--', lw=0.8, alpha=0.9)
        plt.text(m, y_txt, f"lr={lr:g}", ha="right", va="bottom", color="black", fontsize=8)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Wasserstein Loss and Gradient')
    plt.grid(True)
    plt.legend()

    # Plot the loss as a function of the batch / epoch number as well as a histogram of the final particles
    plt.figure()
    plt.hist(X0, density=True, bins=int(math.sqrt(N)), alpha=0.6, color='tab:blue', label='Initial Particles')
    plt.hist(particles, density=True, bins=int(math.sqrt(N)), alpha=0.6, color='tab:orange', label='Adam Optimized Particles')
    plt.plot(x_array, dist, linestyle='--', linewidth=2, color='tab:red', label='Analytic Steady State')
    plt.xlabel('x')
    plt.ylabel(r'$\mu(x)$')
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
    elif args.experiment == 'plot-steady-state':
        plotSteadyState()
    elif args.experiment == 'test':
        test_w2_helpers()
    else:
        print("This experiment is not supported. Choose either 'evolution', 'test', or 'steady-state'.")