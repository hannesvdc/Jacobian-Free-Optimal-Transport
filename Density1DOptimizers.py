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

# Lifting
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

# Restriction through = KDE. 
def kde_1d_fft_neumann(particles: np.ndarray,
                       x_knots:   np.ndarray,
                       bw:        float) -> np.ndarray:
    """
    One-dimensional Gaussian kernel density estimation using FFT. 
    This routine also implements reflective boundary conditions

    Parameters
    ----------
    particles : (N,) array
        Sample points x₁, …, x_N.
    x_knots   : (M,) array
        Locations where the KDE is evaluated.
    bw        : float
        Bandwidth (standard deviation) of the Gaussian kernel.

    Returns
    -------
    mu_knots  : (M,) array
        Estimated (unnormalised) density at each x_knots[j].
    """

    # FFT KDE on the interior
    mu = _kde_fft_core(particles, x_knots, bw)   # O(B log B)

    # Impose discrete Neumann at both ends
    mu[0] = mu[1]  # left wall: μ₀ = μ₁  →   (μ₁-μ₀)/dx = 0
    mu[-1] = mu[-2]  # right wall
    mu /= (mu.sum() * (x_knots[1]-x_knots[0]))   # renormalise, ∫μ=1

    return mu

def _kde_fft_core(particles, x_knots, bw):
    """Internal: equidistant-grid FFT KDE (no boundary tweak)."""
    dx = np.diff(x_knots)
    dx = dx[0]
    B  = x_knots.size

    # 1. histogram
    edges = np.concatenate(([x_knots[0] - 0.5*dx],
                            0.5*(x_knots[:-1] + x_knots[1:]),
                            [x_knots[-1] + 0.5*dx]))
    counts, _ = np.histogram(particles, bins=edges)
    counts = counts.astype(float)

    # 2. FFT padding
    M = int(2 ** np.ceil(np.log2(2*B)))
    f_counts = np.fft.rfft(np.pad(counts, (0, M-B)))

    # 3. FFT of Gaussian kernel on same grid
    grid = np.arange(-B//2, B//2) * dx
    kernel = np.exp(-0.5*(grid/bw)**2)
    kernel /= kernel.sum()
    f_kernel = np.fft.rfft(np.pad(np.roll(kernel, B//2), (0, M-B)))

    # 4. convolution & crop
    density = np.fft.irfft(f_counts * f_kernel, n=M)[:B]
    return density / (particles.size * dx)