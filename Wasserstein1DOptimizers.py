import os
import torch as pt
import numpy as np
import scipy.optimize as opt

# ----------------------------------------------------------------
#  ½ W₂² loss 
# ----------------------------------------------------------------
def w2_loss_1d(
    x: pt.Tensor,  # (B,1)   requires_grad = False
    timestepper,   # (B,1) -> (B,1)
    burn_in: bool = False,      # perform burn-in simulation?
) -> pt.Tensor:
    """
    Returns the scalar ½ W₂²(X , φ_T(X)) for a single mini-batch.
    * x_batch is left on whatever device/dtype the caller uses.
    * timestepper must accept an input of shape (B,1) and return the same shape.
    """
    # 1) perform a burnin if required
    x_cur = timestepper(x) if burn_in else x

    # 2) push the batch through φ_T once
    y = timestepper(x_cur).detach()   # (B, 1)

    # 2) sort both clouds along the particle axis
    idx_x = x_cur[:, 0].argsort()             # (B,)
    idx_y = y[:, 0].argsort()
    diff  = x_cur[idx_x, 0] - y[idx_y, 0]     # (B,)

    # 3) ½ ‖ · ‖² averaged over particles
    loss  = 0.5 * diff.pow(2).mean()      # scalar
    return loss

# ------------------------------------------------------------------
#  ADAM-DRIVEN Wasserstein DESCENT
# ------------------------------------------------------------------
def wasserstein_adam(
    X0: pt.Tensor,  # (N,d)  initial cloud on *device*
    timestepper,
    n_epochs: int,
    batch_size: int,
    lr: float,
    lr_decrease_factor: float,
    lr_decrease_step: int,
    device=pt.device("mps"),
    store_directory: str | None = None):
    """
    Wasserstein-descent steady-state solver driven by Adam.

    Returns
    -------
    X_final   : (N,d) tensor on CPU
    losses    : list of per-batch ½ W_2^2 values
    grad_norms: list of per-batch loss gradients
    """
    X_param = pt.nn.Parameter(X0.clone())
    N, d    = X_param.shape

    print(f"[INFO] Initial Loss", w2_loss_1d(X_param, timestepper).item())

    lr_now = lr
    opt = pt.optim.Adam([X_param], lr=lr, betas=(0.9, 0.999))
    sched = pt.optim.lr_scheduler.StepLR(opt, step_size=lr_decrease_step, gamma=lr_decrease_factor)
    def save_particles(epoch):
        if store_directory is None: return
        pt.save(X_param.detach().cpu(), os.path.join(store_directory, f"particles_wasserstein_adam.pt"))

    losses = []
    grad_norms = []
    for epoch in range(1, n_epochs+1):
        print(f"Epoch {epoch:3d}")
        if store_directory is not None and epoch % 200 == 0:
            save_particles(epoch)

        perm = pt.randperm(N, device=device)

        for start in range(0, N, batch_size):
            idx   = perm[start:start+batch_size]
            x_sub = X_param[idx]
            
            opt.zero_grad()
            loss = w2_loss_1d(x_sub, timestepper, burn_in=False)
            loss.backward()
            grad_norm = X_param.grad.norm().item()

            opt.step()
            losses.append(loss.item())
            grad_norms.append(grad_norm)
        
        # Update the learning rate after each epoch
        sched.step()
        lr_now = opt.param_groups[0]['lr']

        # Display convergence information
        with pt.no_grad():
            probe_idx = pt.randperm(N, device=device)[:min(batch_size, N)]
            disp = (timestepper(X_param[probe_idx]) - X_param[probe_idx]).norm(dim=1).mean().item()
            print(f"last ½S_ε = {loss.item():.4e} | "
                f"‖grad‖₂={grad_norm:.3e} | "
                f"lr={lr_now:.2e} | "
                f"⟨|x-φ(x)|⟩ = {disp:.4e}")

    # Store the loss history
    if store_directory is not None:
        pt.save(pt.stack((pt.tensor(losses), pt.tensor(grad_norms))), os.path.join(store_directory or ".", "wasserstein_adam_losses.pt"))

    return X_param.detach().cpu(), losses, grad_norms

def wasserstein_newton_krylov(
    x0: np.ndarray,
    timestepper, # Must be pt.Tensor to pt.Tensor
    maxiter: int,
    rdiff : float,
    burn_in = False,
    device=pt.device("cpu"),
    dtype=pt.float64,
    store_directory: str | None = None
) -> np.ndarray:
    N = x0.size
    particle_filename = os.path.join(store_directory or ".", f"wasserstein_newton_krylov_particles_eps={rdiff}_N={N}.npy")

    # Create the objective function as a wrapper around w2_loss_1d
    def F(x: np.ndarray) -> np.ndarray:
        # Create a torch tensor from x (make sure to copy)
        X = pt.tensor(x, device=device, dtype=dtype, requires_grad=True).reshape((N,1))

        # Compute the loss
        loss = w2_loss_1d(X, timestepper, burn_in)

        # Compute gradient w.r.t. Xn
        grad, = pt.autograd.grad(loss, X)
        #print('loss', loss.item(), pt.norm(grad).item())

        # Return only the gradient in numpy format. Copy just to be sure
        return grad.detach().cpu().numpy().copy()[:,0]
    
    # Create a callback to store intermediate losses and particles
    losses = []
    grad_norms = []
    def callback(xk, fk):
        X = pt.tensor(xk, device=device, dtype=dtype, requires_grad=True).reshape((N, 1))

        # Compute loss (we need to recompute it here, since fk is just the gradient)
        loss = w2_loss_1d(X, timestepper).item()
        grad_norm = np.linalg.norm(fk)

        # Store values
        losses.append(loss)
        grad_norms.append(grad_norm)

        # Print for monitoring
        print(f"(N = {N}, rdiff = {rdiff}) Epoch {len(losses)}: loss = {loss}, ‖grad‖ = {grad_norm}")

        # Store the particles
        if store_directory is not None:
            np.save(particle_filename, xk)


    # Solve F(x) = 0 using scipy.newton_krylov. The parameter rdiff is key!
    line_search = 'wolfe'
    tol = 1.e-14
    try:
        x_inf = opt.newton_krylov(F, x0, f_tol=tol, maxiter=maxiter, rdiff=rdiff, line_search=line_search, callback=callback, verbose=False)
    except opt.NoConvergence as e:
        x_inf = e.args[0]

    # Store the loss and grad_norm history
    if store_directory is not None:
        filename = os.path.join(store_directory or ".", f"wasserstein_newton_krylov_losses_eps={rdiff}_N={N}.npy")
        data = np.stack((np.array(losses), np.array(grad_norms)), axis=0)
        np.save(filename, data)
        np.save(particle_filename, x_inf)

    return x_inf

def _call_loss(X0: pt.Tensor, timestepper, batch_size: int, device: str | pt.device = "mps"):
    # Act as if we're doing minibatching
    N, d = X0.shape
    perm = pt.randperm(N, device=device)
    idx = perm[0:batch_size]
    x_sub = X0[idx]

    # Call the loss function
    loss = w2_loss_1d(x_sub, timestepper)
    return loss