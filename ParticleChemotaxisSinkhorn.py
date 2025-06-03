import os
import torch as pt
import math
from geomloss import SamplesLoss
import matplotlib.pyplot as plt

import SinkhornOptimizers as sopt

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
    print(x_array)
    print(dist)

    # Plot the particle histogram and compare it to the analytic steady-state
    plt.hist(X_inf.cpu().numpy(), density=True, bins=int(math.sqrt(N)), label='Particles')
    plt.plot(x_array.numpy(), dist.numpy(), linestyle='--', label='Analytic Steady State')
    plt.xlabel('x')
    plt.ylabel(r'$\mu(x)$')
    plt.grid(True)
    plt.legend()
    plt.show()

def steadyStateSinkhorn(optimizer):
    device = pt.device("mps")
    dtype = pt.float32
    store_directory = "/Users/hannesvdc/Research/Projects/Jacobian-Free-Optimal-Transport/Results/"

    # Physical functions defining the problem
    S = lambda x: pt.tanh(x)
    dS = lambda x: 1.0 / pt.cosh(x)**2
    chi = lambda s: 1 + 0.5 * s**2
    D = 0.1

    # Initial condition - Gaussian (mean 0, stdev 1)
    N = 10**4
    X0 = pt.normal(0.0, 1.0, (N,1), device=device, dtype=dtype, requires_grad=False)

    # Build the timestepper function
    dt = 1.e-3
    T_psi = 1.0
    stepper = lambda X: timestepper(X, S, dS, chi, D, dt, T_psi, device=device, dtype=dtype)

    # Do optimization to find the steady-state particles
    batch_size = 10000
    lr = 1.0
    replicas = 10
    if optimizer == 'SGD':
        epochs = 5000
        X_inf, losses = sopt.sinkhorn_sgd(X0, stepper, epochs, batch_size, lr, replicas, device=device, store_directory=store_directory)
    elif optimizer == 'Adam':
        epochs = 1500
        X_inf, losses, grad_norms = sopt.sinkhorn_adam(X0, stepper, epochs, batch_size, lr, replicas, device=device, store_directory=store_directory)

    # Analytic Steady-State for the given chi(S)
    x_array = pt.linspace(-L, L, 1000)
    dist = pt.exp( (S(x_array) + S(x_array)**3 / 6.0) / D)
    Z = pt.trapz(dist, x_array)
    dist = dist / Z

    # Plot the loss as a function of the batch / epoch number as well as a histogram of the final particles
    batch_counter = pt.linspace(0.0, epochs, len(losses))
    plt.semilogy(batch_counter, losses, label='Sinkhorn Divergence')
    plt.semilogy(batch_counter, grad_norms, label='Sinkhorn Divergence Gradient')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.hist(X_inf, density=True, bins=int(math.sqrt(N)), label='Particles')
    plt.plot(x_array, dist, linestyle='--', label='Analytic Steady State')
    plt.xlabel('x')
    plt.ylabel(r'$\mu(x)$')
    plt.grid(True)
    plt.legend()
    
    plt.show()

def plotSinkhornSteadyState(optimizer):
    store_directory = "./Results/"
    filename = os.path.join(store_directory, "particles_adam.pt")
    particles = pt.load(filename)
    N = len(particles)

    losses_filename = os.path.join(store_directory, "sinkhorn_adam_losses.pt")
    losses_and_grads = pt.load(losses_filename)
    if losses_and_grads.shape[0] < losses_and_grads.shape[1]:
        losses = losses_and_grads[0,:]
        grad_norms = losses_and_grads[1,:]
    else:
        losses = losses_and_grads[:,0]
        grad_norms = losses_and_grads[:,1]
    epochs = len(losses)

    # Get initial distribution for plotting purposes
    N = 10**4
    X0 = pt.normal(0.0, 1.0, (N,), requires_grad=False)

    # Analytic Steady-State for the given chi(S)
    S = lambda x: pt.tanh(x)
    D = 0.1
    x_array = pt.linspace(-L, L, 1000)
    dist = pt.exp( (S(x_array) + S(x_array)**3 / 6.0) / D)
    Z = pt.trapz(dist, x_array)
    dist = dist / Z

    # Plot the loss as a function of the batch / epoch number as well as a histogram of the final particles
    batch_counter = pt.linspace(0.0, epochs, len(losses))
    plt.semilogy(batch_counter, losses, label='Sinkhorn Divergence')
    plt.semilogy(batch_counter, grad_norms, label='Sinkhorn Divergence Gradient')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plot the loss as a function of the batch / epoch number as well as a histogram of the final particles
    plt.hist(X0, density=True, bins=int(math.sqrt(N)), label='Initial Particles')
    plt.hist(particles, density=True, bins=int(math.sqrt(N)), label='Optimized Particles')
    plt.plot(x_array, dist, linestyle='--', label='Analytic Steady State')
    plt.xlabel('x')
    plt.ylabel(r'$\mu(x)$')
    plt.grid(True)
    plt.legend()
    
    plt.show()

def testSinkhornSGDSteadyState():
    # Physical functions defining the problem
    S = lambda x: pt.tanh(x)
    dS = lambda x: 1.0 / pt.cosh(x)**2
    chi = lambda s: 1 + 0.5 * s**2
    D = 0.1
    dist = lambda x: pt.exp( (S(x) + S(x)**3 / 6.0) / D)
    dt = 1.e-3
    T_psi = 1.0
    stepper = lambda X: timestepper(X, S, dS, chi, D, dt, T_psi, device=pt.device("cpu"), dtype=pt.float32)

    # Sample N particles from the invariant distribution
    N = 10_000
    samples_list = sampleInvariantMCMC(dist, N)

    # Calcuate the Sinkhorn-loss
    samples_tensor = pt.Tensor(samples_list).reshape((N,1))
    eps = sopt.choose_eps_blur(samples_tensor, stepper, N, multiplier=1.0)
    replicas = 10
    loss_fn = SamplesLoss(
        loss   = "sinkhorn",
        p      = 2,
        blur   = eps,
        debias = True,                       # Sinkhorn *divergence*
        scaling= 0.9,                        # ε-scaling warm start
        backend= "tensorized",               # fast for B ≲ 20 000
    )
    loss =  sopt.sinkhorn_loss(samples_tensor, stepper, loss_fn, replicas)
    print('Upper Bound for the Steady-State Sinkhorn Loss', loss.item())

    x_array = pt.linspace(-L, L, 1000)
    plt.hist(samples_list, density=True, bins=int(math.sqrt(N)), label='Particles')
    plt.plot(x_array, dist(x_array) / pt.trapz(dist(x_array), x_array), linestyle='--', label='Analytic Steady State')
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
        help="Specify the experiment to run (e.g., 'evolution', 'sinkhorn')."
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        required=False,
        dest='optimizer',
        default='Adam',
        help="Specify the optimizer to use for steady-state calculations. Options are SGD and Adam (default)."
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArguments()
    if args.experiment == 'evolution':
        timeEvolution()
    elif args.experiment == 'sinkhorn':
        steadyStateSinkhorn(args.optimizer)
    elif args.experiment == 'plot':
        plotSinkhornSteadyState(args.optimizer)
    else:
        print("This experiment is not supported. Choose either 'evolution', 'sinkhorn'.")