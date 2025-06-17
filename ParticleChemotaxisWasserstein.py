import os
import gc

import math
import torch as pt
import numpy as np
import scipy.optimize as opt
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

def step(X : pt.Tensor, S, dS, chi, D, dt, device, dtype) -> pt.Tensor:
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

def timestepper(X : pt.Tensor, S, dS, chi, D, dt, T, device, dtype, verbose=False) -> pt.Tensor:
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

    # Initial distribution of particles (Invariant Measure)
    N = 10**4
    X0_list = sampleInvariantMCMC(dist, N)
    X0 = pt.tensor(X0_list).reshape((N,1)).to(device=device, dtype=dtype)

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

def calculateSteadyStateAdam():
    device = pt.device("mps")
    dtype = pt.float32
    store_directory = "./Results/"

    # Physical functions defining the problem
    S = lambda x: pt.tanh(x)
    dS = lambda x: 1.0 / pt.cosh(x)**2
    chi = lambda s: 1 + 0.5 * s**2
    D = 0.1

    # Initial condition: Gaussian (mean 5, stdev 2) with correct boundary conditions
    N = 10**5
    X0 = pt.normal(5.0, 2.0, (N,1), device=device, dtype=dtype, requires_grad=False)
    X0 = pt.where(X0 < -L, 2 * (-L) - X0, X0)
    X0 = pt.where(X0 > L, 2 * L - X0, X0)

    # Build the timestepper function
    dt = 1.e-3
    T_psi = 1.0
    stepper = lambda X: timestepper(X, S, dS, chi, D, dt, T_psi, device=device, dtype=dtype)

    # Do optimization to find the steady-state particles
    # These parameters work really well!! - 150s of total integration time 
    # versus 500s for regular timestepping (see time_evolution code).
    batch_size = N
    lr = 1.e-1
    lr_decrease_factor = 0.1
    lr_decrease_step = 100
    n_lrs = 3
    epochs = n_lrs * lr_decrease_step
    X_inf, losses, grad_norms = wopt.wasserstein_adam(X0, stepper, epochs, batch_size, lr, lr_decrease_factor, lr_decrease_step, device, store_directory=store_directory)

    # Analytic Steady-State for the given chi(S)
    x_array = pt.linspace(-L, L, 1000)
    dist = pt.exp( (S(x_array) + S(x_array)**3 / 6.0) / D)
    Z = pt.trapz(dist, x_array)
    dist = dist / Z

    # Clean up memory
    gc.collect()

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

def calculateSteadyStateNewtonKrylov():
    """
    This function uses scipy's newton_krylov to solve \\nabla_X|A 1/2 W_2^2(X, \\phi_T(X)) = 0. 
    This is obviously equivalent to minimizing W_2^2(X, \\phi_T(X)). We only use the
    preconditioned gradient - just as with the Adam optimizer.

    Because of scipy, this is a numpy-only function. It internally calls w2_loss_1d in pytorch
    language, but we need to explicitely take care of conversions. Also, this is a CPU function
    because numpy does not provide support for the Apple Neural Engine.
    """
    store_directory = "./Results/"

    # Physical functions defining the problem. These have to be torch functions 
    # because they are called in the timesteper.
    S = lambda x: pt.tanh(x)
    dS = lambda x: 1.0 / pt.cosh(x)**2
    chi = lambda s: 1 + 0.5 * s**2
    D = 0.1

    # Initial condition: Gaussian (mean 5, stdev 2) with correct boundary conditions. Numpy.
    N = 10**5
    x0 = np.random.normal(5.0, 2.0, N)
    x0 = np.where(x0 < -L, 2 * (-L) - x0, x0)
    x0 = np.where(x0 > L, 2 * L - x0, x0)

    # Define the Newton-Krylov objective function F(x). Input and output are numpy arrays!
    # First build the timestepper which takes in torch tensors!
    dt = 1.e-3
    T_psi = 1.0
    burnin_T = None
    device = pt.device('cpu')
    dtype = pt.float64
    def stepper(X : pt.Tensor, T : float = T_psi) -> pt.Tensor:
        return timestepper(X, S, dS, chi, D, dt, T, device=device, dtype=dtype)
    rdiff = 1.e-1 # the epsilon parameter
    maxiter = 50
    x_inf, losses, grad_norms = wopt.wasserstein_newton_krylov(x0, stepper, maxiter, rdiff, burnin_T, device, dtype, store_directory=None)

    # Plot the steady-state and the analytic steady-state
    x_array = pt.linspace(-L, L, 1000)
    dist = pt.exp( (S(x_array) + S(x_array)**3 / 6.0) / D)
    dist = dist / pt.trapz(dist, x_array)

    plt.figure()
    plt.hist(x0, density=True, bins=int(math.sqrt(N)), alpha=0.6, color='tab:blue', label='Initial Particles')
    plt.hist(x_inf, density=True, bins=int(math.sqrt(N)), alpha=0.6, color='tab:orange', label='Newton-Krylov Optimized Particles')
    plt.plot(x_array, dist, linestyle='--', linewidth=2, color='tab:red', label='Analytic Steady State')
    plt.xlabel('x')
    plt.ylabel(r'$\mu(x)$')
    plt.grid(True)
    plt.legend()
    
     # Plot the loss as a function of the batch / epoch number as well as a histogram of the final particles
    plt.figure()
    iterations = 1 + np.arange(len(losses))
    plt.semilogy(iterations, losses, label='Newton-Krylov Reidual')
    plt.semilogy(iterations, grad_norms, label='Newton-Krylov Residual Gradient')
    plt.xlabel('Iteration')
    plt.grid(True)
    plt.legend()

    plt.show()

def findOptimalNKParameters():
    store_directory = "./Results/"

    # Physical functions defining the problem. These have to be torch functions 
    # because they are called in the timesteper.
    S = lambda x: pt.tanh(x)
    dS = lambda x: 1.0 / pt.cosh(x)**2
    chi = lambda s: 1 + 0.5 * s**2
    D = 0.1

    # First build the timestepper which takes in torch tensors!
    dt = 1.e-3
    T_psi = 1.0
    burnin_T = 10 * dt
    device = pt.device('cpu')
    dtype = pt.float64
    maxiter = 50
    def stepper(X : pt.Tensor, T : float = T_psi) -> pt.Tensor:
        return timestepper(X, S, dS, chi, D, dt, T, device=device, dtype=dtype)

    # Initial condition: Gaussian (mean 5, stdev 2) with correct boundary conditions. Numpy.
    N_array = [10**4, 2*10**4, 4*10**4, 8*10**4, 10**5]
    rdiff_array = [1.e-5, 1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.0, 10.0]
    for N in N_array:
        for rdiff in rdiff_array:
            print('N =', N, ' rdiff =', rdiff)

            x0 = np.random.normal(5.0, 2.0, N)
            x0 = np.where(x0 < -L, 2 * (-L) - x0, x0)
            x0 = np.where(x0 > L, 2 * L - x0, x0)

            x_inf, losses, grad_norms = wopt.wasserstein_newton_krylov(x0, stepper, maxiter, rdiff, burnin_T, device, dtype, store_directory)

            # Apply boundary conditions just to be sure.
            x_inf = np.where(x_inf < -L, 2 * (-L) - x_inf, x_inf)
            x_inf = np.where(x_inf > L, 2 * L - x_inf, x_inf)

            # Store the loss and grad_norm history
            if store_directory is not None:
                filename = os.path.join(store_directory or ".", f"wasserstein_newton_krylov_losses_eps={rdiff}_N={N}.npy")
                data = np.stack((np.array(losses), np.array(grad_norms)), axis=0)
                np.save(filename, data)

def plotOptimalNKParameters():
    store_directory = "./Results/"

    # Initialize results matrix
    N_array = [10**4, 2*10**4, 4*10**4, 8*10**4, 10**5]
    rdiff_array = [1.e-5, 1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.0, 10.0]
    loss_surface = np.full((len(N_array), len(rdiff_array)), np.nan)
    grad_surface = np.full((len(N_array), len(rdiff_array)), np.nan)

    # Load data
    for i, N in enumerate(N_array):
        for j, rdiff in enumerate(rdiff_array):
            filename = os.path.join(store_directory, f"wasserstein_newton_krylov_losses_eps={rdiff}_N={N}.npy")
            if os.path.exists(filename):
                data = np.load(filename)
                final_loss = data[0, -1]  # first row = losses
                final_grad = data[1, -1]
                loss_surface[i, j] = final_loss
                grad_surface[i, j] = final_grad
            else:
                print(f"File not found: {filename}")

    # Convert to log10 for plotting (avoid log(0) issues)
    log_Ns = np.log10(N_array)
    log_rdiffs = np.log10(rdiff_array)
    log_loss = np.log10(loss_surface)
    log_grad = np.log10(grad_surface)

    # Plot
    c_loss = plt.pcolormesh(log_rdiffs, log_Ns, log_loss, shading='auto', cmap='viridis')
    plt.colorbar(c_loss)
    plt.xlabel(r"$\log_{10}(\varepsilon)$")
    plt.ylabel(r"$\log_{10}(N)$")
    plt.title("Final Wasserstein Losses")
    plt.xticks(log_rdiffs, labels=[f"{r:.0e}" for r in rdiff_array])
    plt.yticks(log_Ns, labels=[str(N) for N in N_array])
    plt.tight_layout()

    plt.figure()
    c_grad = plt.pcolormesh(log_rdiffs, log_Ns, log_grad, shading='auto', cmap='viridis')
    plt.colorbar(c_grad)
    plt.xlabel(r"$\log_{10}(\varepsilon)$")
    plt.ylabel(r"$\log_{10}(N)$")
    plt.title("Final Wasserstein Gradient Norms")
    plt.xticks(log_rdiffs, labels=[f"{r:.0e}" for r in rdiff_array])
    plt.yticks(log_Ns, labels=[str(N) for N in N_array])
    plt.tight_layout()
    plt.show()

def plotAdamSteadyState():
    store_directory = "./Results/"
    filename = os.path.join(store_directory, "particles_wasserstein_adam.pt")
    particles = pt.load(filename, weights_only=True)
    N = len(particles)

    losses_filename = os.path.join(store_directory, "wasserstein_adam_losses.pt")
    losses_and_grads = pt.load(losses_filename, weights_only=True)
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
    lrs = [1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6]
    y_top = losses.max().item()          # highest point on the log plot
    y_txt = y_top * 0.8   
    #for m, lr in zip(milestones, lrs):
    #    plt.axvline(m, color='gray', linewidth=5, ls='--', lw=0.8, alpha=0.9)
    #    plt.text(m, y_txt, f"lr={lr:g}", ha="right", va="bottom", color="black", fontsize=8)
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
    parser.add_argument(
        '--optimizer',
        type=str,
        required=False,
        dest='optimizer',
        help="Specify the optimizer to use, e.g. adam or newton_krylov."
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArguments()
    if args.experiment == 'evolution':
        timeEvolution()
    elif args.experiment == 'steady-state':
        if args.optimizer == 'adam':
            calculateSteadyStateAdam()
        elif args.optimizer == 'newton_krylov':
            calculateSteadyStateNewtonKrylov()
        else:
            print('This optimizer is not supported.')
    elif args.experiment == 'optimal_parameters':
        findOptimalNKParameters()
    elif args.experiment == 'plot_optimal_parameters':
        plotOptimalNKParameters()
    elif args.experiment == 'plot-steady-state':
        if args.optimizer == 'adam':
            plotAdamSteadyState()
        else:
            print('Choose an optimizer who\'s steady state to show')
    elif args.experiment == 'test':
        test_w2_helpers()
    else:
        print("This experiment is not supported. Choose either 'evolution', 'test', or 'steady-state'.")