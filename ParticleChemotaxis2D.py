import gc
import torch as pt
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt

from Wasserstein2DOptimizers import wasserstein_adam
from typing import Optional

# --------------------------------------------
# Half-moon potential: grid and particle forms
# --------------------------------------------
def half_moon_potential_meshgrid(
    x: pt.Tensor, y: pt.Tensor,
    A: float, R: float, B: float, alpha: float, y_shift: float
) -> pt.Tensor:
    """
    Potential on a meshgrid: x,y have the same shape.
    Returns a tensor with that shape.
    """

    r = pt.sqrt(x**2 + y**2)
    U_radial = A * (r - R)**2
    U_wall   = B * pt.exp(-alpha * (y - y_shift))
    return U_radial + U_wall

def half_moon_potential(
    z: pt.Tensor,  # (N,2)
    A: float, R: float, B: float, alpha: float, y_shift: float
) -> pt.Tensor:
    """
    Potential evaluated at particles z=(x,y). Returns shape (N,).
    """
    r = pt.sqrt(z[:, 0]**2 + z[:, 1]**2)
    U_radial = A * (r - R)**2
    U_wall   = B * pt.exp(-alpha * (z[:, 1] - y_shift))
    return U_radial + U_wall

def gradient_half_moon_potential(
    z: pt.Tensor,  # (N,2)
    A: float, R: float, B: float, alpha: float, y_shift: float,
    eps: float = 1e-12
) -> pt.Tensor:
    """
    Analytic gradient ∇U(z) with shape (N,2).
    Uses a safe division for r near 0.
    """
    x = z[:, 0]
    y = z[:, 1]
    r = pt.sqrt(x**2 + y**2)

    # radial_prefactor = 2A (r - R) / r, with r=0 -> 0
    denom = pt.clamp(r, min=eps)
    radial_prefactor = 2 * A * (r - R) / denom
    radial_prefactor = pt.where(r > 0, radial_prefactor, pt.zeros_like(radial_prefactor))

    grad_radial_x = radial_prefactor * x
    grad_radial_y = radial_prefactor * y

    # wall gradient (x-component is zero)
    grad_wall_y = -alpha * B * pt.exp(-alpha * (y - y_shift))

    grad_x = grad_radial_x
    grad_y = grad_radial_y + grad_wall_y
    return pt.stack((grad_x, grad_y), dim=1)  # (N,2)

# ----------------
# Reflect operator
# ----------------
def reflect(coords: pt.Tensor, L: float) -> pt.Tensor:
    """
    Vectorized reflection of an (N,2) tensor into [-L, L]^2.
    Mirrors once like your NumPy version.
    """
    over  = coords >  L
    under = coords < -L
    # apply reflections without in-place ops that would break autograd graphs
    reflected = pt.where(over,  2*L - coords, coords)
    reflected = pt.where(under, -2*L - reflected, reflected)
    return reflected

# -----------
# One timestep
# -----------
def step(
    z: pt.Tensor, dt: float, gen: Optional[pt.Generator],
    A: float, R: float, B: float, alpha: float, y_shift: float, L: float
) -> pt.Tensor:
    """
    z : (N,2) tensor of positions
    returns updated (N,2) tensor
    """
    dt_t = pt.as_tensor(dt, device=z.device, dtype=z.dtype)
    z = reflect(z, L)
    drift = gradient_half_moon_potential(z, A, R, B, alpha, y_shift)
    noise = pt.randn(z.shape, generator=gen, device=z.device, dtype=z.dtype)
    z_new = z - drift * dt_t + pt.sqrt(2.0 * dt_t) * noise
    z_new = reflect(z_new, L)
    return z_new

# ---------------
# Many timesteps
# ---------------
def timestepper(
    z: pt.Tensor, dt: float, T: float, gen: Optional[pt.Generator],
    A: float, R: float, B: float, alpha: float, y_shift: float, L: float = 4.0
) -> pt.Tensor:
    """
    Advances the particle cloud to time T and returns final positions.
    """
    n_steps = int(pt.ceil(pt.as_tensor(T / dt)).item())
    for _ in range(n_steps):
        z = step(z, dt, gen, A=A, R=R, B=B, alpha=alpha, y_shift=y_shift, L=L)
    return z

def calculateSteadyStateWasserstein():
    device = pt.device('mps')
    gen = pt.Generator(device=device)
    R = 2.0
    A = 2.0
    B = 0.5
    alpha = 1.5
    y_shift = -0.5

    # Start with a standard normal Gaussian and sample particles
    N = 100_000
    L = 4.0
    particles = MultivariateNormal(pt.Tensor([0.0, 0.0]), pt.eye(2)).rsample((N,)).to(device=device)
    particles = reflect(particles, L)

    # Make the necessary objects for the W2 optimizer
    dt = 1e-3
    T_psi = 1.0
    particle_timestepper = lambda z: timestepper(z, dt, T_psi, gen, A, R, B, alpha, y_shift, L)
    
    # Do Wasserstein optimization
    batch_size = 1_000
    lr = 1.e-2
    lr_decrease_factor = 0.1
    lr_decrease_step = 100
    n_lrs = 3
    epochs = n_lrs * lr_decrease_step
    optimal_particles, losses, grad_norms = wasserstein_adam(particles, particle_timestepper, epochs, batch_size, lr, lr_decrease_factor, lr_decrease_step, device)
    optimal_particles = optimal_particles.cpu().numpy()

    # Clean up memory
    gc.collect()

    # Store the losses, gradnorms, and optimal particles
    loss_data = np.vstack((np.asarray(losses), np.asarray(grad_norms)))
    np.save('./Results/2DWasserstein_losses.npy', loss_data)
    np.save('./Results/2DWasserstein_particles.npy', optimal_particles)

    # Plot the loss as a function of the batch / epoch number as well as a histogram of the final particles
    batch_counter = pt.linspace(0.0, epochs, len(losses))
    plt.semilogy(batch_counter, losses, label='Wasserstein Loss')
    plt.semilogy(batch_counter, grad_norms, label='Wasserstein Loss Gradient')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    H, x_edges, y_edges = np.histogram2d(optimal_particles[:,0], optimal_particles[:,1], density=True, range=[[-L, L], [-L, L]], bins=[40,40])
    fig2d, ax2d = plt.subplots(figsize=(6, 5))
    im = ax2d.imshow(
        H.T,
        origin="lower",
        cmap="viridis",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], #type: ignore
        aspect="auto",
    )
    fig2d.colorbar(im, ax=ax2d, label="density")
    ax2d.set_xlabel("x"); ax2d.set_ylabel("y")
    ax2d.set_title(f"Histogram heat map")
    plt.tight_layout()

    plt.legend()
    plt.show()

if __name__ == '__main__':
    calculateSteadyStateWasserstein()