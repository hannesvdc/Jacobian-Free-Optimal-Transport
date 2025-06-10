import numpy as np
from scipy.interpolate import CubicSpline

def median_tabulated(x, mu):
    """Approximate the 50th percentile from (x_i, μ_i) samples."""
    area = 0.5 * (mu[:-1] + mu[1:]) * np.diff(x)   # trapezoids
    cdf  = np.hstack((0.0, np.cumsum(area)))       # un-normalised CDF
    half = 0.5 * cdf[-1]                           # target mass
    j    = np.searchsorted(cdf, half)              # first bin that exceeds ½
    t    = (half - cdf[j-1]) / (cdf[j] - cdf[j-1]) # linear frac inside bin
    return x[j-1] + t * (x[j] - x[j-1])

def reflected_hmc_from_tabulated_density(
        x_knots: np.ndarray,    # 1-D numpy array, shape (M,)
        mu_knots: np.ndarray,   # 1-D numpy array, same shape – *unnormalised* density values
        N : int,                # number of samples to return
        step_size : float,      # leap-frog step
        rng=None
    ):
    """
    Draw `N` samples from a 1-D density known only at `x_knots`.
    Neumann (zero-flux) boundary conditions are enforced via reflection.
    Returns a 1-D numpy array of length `N`.
    """
    if rng is None:
        rng = np.random.RandomState()

    # 1.  spline in log-space with clamped (∂x log μ = 0) BC
    spline = CubicSpline(x_knots, mu_knots, bc_type='clamped')   # Neumann at both ends
    U   = lambda x: -np.log(spline(x) )     # potential  U  = −log π
    dU  = lambda x: -spline.derivative()(x) / spline(x)  # dU/dx = - dmu(x)/dx / mu(x)

    # 2. Boundary points and starting point (50th percentile)
    a, b = x_knots[0], x_knots[-1]
    L = b - a
    boundary = lambda y: a + np.abs((y - a) % (2*L) - L)
    x = median_tabulated(x_knots, mu_knots)

    # 3.  Sampling Loop
    particles = np.zeros(N)
    n_accepted = 0
    for n in range(N):
        # Random momentum sampling
        p = rng.normal()
        E = U(x) + 0.5 * p * p

        # Do a leapfrog step
        p = p - 0.5 * step_size * dU(x)  # half-kick
        xp = x + step_size * p
        if xp < a or xp > b:
            xp = boundary(xp)
            p = -p
        p = p - 0.5 * step_size * dU(xp) # half-kick

        # Accept / Reject
        Ep = U(xp) + 0.5 * p * p
        ln_acc = -(Ep - E)
        if np.log(rng.random()) < ln_acc:
            x = xp
            n_accepted += 1
        particles[n] = x

    print(f"Acceptance rate: {n_accepted / N:.3f}")
    return particles