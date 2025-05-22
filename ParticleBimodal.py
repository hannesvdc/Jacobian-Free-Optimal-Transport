import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.optimize as opt
import matplotlib.pyplot as plt

from FastKDE import fast_sliding_kde

V = lambda x: 0.5* (x**2 - 1)**2
dV = lambda x: 2.0 * (x**2 - 1) * x
mu = lambda x: - dV(x)
sigma = lambda x: 1.0

# HMC Sampling routine to generate bimodal particles
def sampleBimodal(N : int):
    rng = rd.RandomState()

    dt = 1.0
    n_accepted = 0.0

    x = 0.0
    samples = np.zeros(N)
    for n in range(N):
        p = rng.normal(0.0, 1.0)
        current_H = V(x) + 0.5 * np.dot(p, p)

        # Half step for momentum
        p_new = p - 0.5 * dt * dV(x)

        # Full step for position
        x_new = x + dt * p_new

        # Another half step for momentum
        p_new = p_new - 0.5 * dt * dV(x_new)

        # Metropolis-Hastings criterion
        proposed_H = V(x_new) + 0.5 * np.dot(p_new, p_new)
        alpha = np.exp(current_H - proposed_H)

        if rng.uniform() <= alpha:
            n_accepted += 1
            x = x_new
        samples[n] = x

    print('Average Acceptance Rate', n_accepted / N)
    return samples

def EMOTStep(X, h, rng):
    return X + mu(X) * h +  np.sqrt(2.0 * h) * sigma(X) * rng.normal(0.0, 1.0, size=X.shape)

def EMOTTimestepper(X, h, T, rng, verbose=False):
    """
        Assumes X is sorted, but this is not a necessary precondition.
    """
    n_steps = int(T / h)
    for n in range(n_steps):
        if n % 1000 == 0 and verbose:
            print('t =', n*h)
        X = EMOTStep(X, h, rng)
    return np.sort(X)

def EMOTpsi(X0, h, Tpsi, rng):
    """
        Checks if X0 if sorted. If not, sorting is done in O(N log N) time.
    """
    if not np.all(X0[:-1] <= X0[1:]):
        X0 = np.sort(X0)
    X = EMOTTimestepper(X0, h, Tpsi, rng)
    return X0 - X

def Wasserstein(X0, X):
    """ 
    Assumes X0 and X are sorted and of the same length.
    """
    assert len(X0) == len(X)
    return np.sqrt(np.sum((X0 - X)**2) / len(X0))

def plotKDE():
    N = 1_000_000
    samples = sampleBimodal(N)
    samples = np.sort(samples)

    # Do KDE
    print('KDE')
    bandwidth = 0.05
    kde_eval = fast_sliding_kde(samples, bandwidth, verbose=True)

    # Get the analytic distribution
    x_array = np.linspace(-5, 5, 1001)
    dist = lambda x: np.exp(-V(x))
    Z = np.trapz(dist(x_array), x_array)

    # Plot everything
    print('Plotting')
    plt.hist(samples, density=True, bins=int(np.sqrt(N)), label='Samples')
    plt.plot(samples, kde_eval, label='KDE')
    plt.plot(x_array, dist(x_array)/Z, label='Exact Distribution')
    plt.xlabel(r'$x$')
    plt.legend()
    plt.show()

def timeEvolution():
    N = 1000000
    mean = 2.0
    stdev = 2.0
    rng = rd.RandomState()
    X0 = np.sort(rng.normal(mean, stdev, N))
 
    h = 0.001
    T = 10.0
    X = EMOTTimestepper(X0, h, T, rng)
    print('Steady-State Reached')

    # Evaluate Psi in steady state
    Tpsi = 1.0
    Xnew = EMOTTimestepper(X, h, Tpsi, rng)
    print('Steady-state W', Wasserstein(X, Xnew))

    # Plot the simulation results
    x_array = np.linspace(-3, 3, 1001)
    V_array = V(x_array)
    dist = np.exp(-V_array)
    dist = dist / np.trapz(dist, x_array)
    plt.hist(X, bins=int(np.sqrt(N)), density=True, label='Euler Maruyama + OT')
    plt.plot(x_array, dist, label='Bimodal Distribution')
    plt.xlabel(r'$x$')
    plt.title('EMOT Time Evolution')
    plt.show()

def minimizeW2():
    N = 1_000_000
    mean = 0.0
    stdev = 1.0
    rng = rd.RandomState()
    X0 = np.sort(rng.normal(mean, stdev, size=N))

    # Start in steady-state for testing purposes
    #X0 = sampleBimodal(N)
    #X0 = np.sort(X0)

    # define drift and diffusion
    h = 0.001
    Tpsi = 0.1
    class FunctionWithCache:
        def __init__(self):
            self.last_x = None
            self.last_f = None

        def F(self, X0):
            X0 = np.sort(X0)
            X = EMOTTimestepper(X0, h, Tpsi, rng)
            val = Wasserstein(X0, X)
            print('W_2', val)
            return val

        def __call__(self, x):
            self.last_x = np.copy(x)
            self.last_f = self.F(x)
            return self.last_f

        def get_last_result(self):
            return self.last_x, self.last_f
        
        def cb(self, xk):
            _, fx = self.get_last_result()
            print("Callback:", fx)

    # Do Newton minimization with FD for the gradient and BFGS for the Hessian
    f_cached = FunctionWithCache()
    ss_W2_values = []
    eps_values = [1.0, 1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6]
    for eps in eps_values:
        res = opt.minimize(f_cached, X0, method='BFGS', jac=False, options={'eps': eps, 'disp': True, 'maxiter': 10}, callback=f_cached.cb)
        ss_W2_values.append(res.fun)

    # plot the samples
    x_array = np.linspace(-3, 3, 1001)
    V_array = V(x_array)
    dist = np.exp(-V_array)
    dist = dist / np.trapz(dist, x_array)
    plt.hist(res.x, bins=int(np.sqrt(N)), density=True) # type: ignore
    plt.plot(x_array, dist, label='Bimodal Distribution')
    plt.xlabel(r'$x$')
    plt.title('Steady-state Distribution Newton-Krylov')
    plt.show()

def relaxedIteration():
    

def nkSteadyState():
    """
        Not working yet.
    """
    N = 1_000_000
    mean = 1.0
    stdev = 1.0
    rng = rd.RandomState()
    X0 = np.sort(rng.normal(mean, stdev, size=N))

    # define drift and diffusion
    h = 0.001
    Tpsi = 0.1
    rdiff = 1.e-4

    # Define Newton-Krylov parameters
    print('Starting Newton-Krylov...')
    def callback(x, f):
        print(f"Iteration: x = {x}, f = ", lg.norm(f)/N)
    f = lambda x: EMOTpsi(x, h, Tpsi,rng)
    try:
        X_ss = opt.newton_krylov(f, X0, verbose=True, rdiff=rdiff, maxiter=10, line_search=None, method='gmres', callback=callback)
    except opt.NoConvergence as e:
        X_ss = e.args[0]
    except ValueError as e:
        return
        pass

    # Plot the steady-state histogran
    x_array = np.linspace(-3, 3, 1001)
    V_array = V(x_array)
    dist = np.exp(-V_array)
    dist = dist / np.trapz(dist, x_array)
    plt.hist(X_ss, bins=int(np.sqrt(N)), density=True)
    plt.plot(x_array, dist, label='Bimodal Distribution')
    plt.xlabel(r'$x$')
    plt.title('Steady-state Distribution Newton-Krylov')
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
    elif args.experiment == 'minimizeW2':
        minimizeW2()
    else:
        print("This experiment is not supported. Choose either 'evolution', 'minimizeW2'.")