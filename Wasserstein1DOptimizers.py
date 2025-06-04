import os
import torch as pt

# ---------------------------------------------------------------
#  1-D ½ W₂² loss   (MC-average over stochastic images)
# ---------------------------------------------------------------
# ----------------------------------------------------------------
#  ½ W₂² loss  (no autograd)      — averaging over replicas
# ----------------------------------------------------------------
def w2_loss_1d(
    x: pt.Tensor,        # (B,1)   requires_grad = False
    batched_timestepper, # (L,B,1) -> (L,B,1)
    replicas: int = 1,
) -> pt.Tensor:
    """
    Returns the scalar ½ W₂²(X , φ_T(X)) for a single mini-batch.
    * x_batch is left on whatever device/dtype the caller uses.
    * batched_timestepper must accept an input of shape (L,B,1)
      where L = replicas, and return the same shape.
    """
    # 1) push L replicas through φ_T and average
    x_rep = x.repeat(replicas, 1, 1)          # (L,B,1)
    y_avg = batched_timestepper(x_rep).mean(0).detach()  # (B,1)

    # 2) sort both clouds along the particle axis
    idx_x = x[:, 0].argsort()                 # (B,)
    idx_y = y_avg[:, 0].argsort()             # (B,)

    diff  = x[idx_x, 0] - y_avg[idx_y, 0]     # (B,)
    loss  = 0.5 * diff.pow(2).mean()          # scalar
    return loss

# ------------------------------------------------------------------
#  ADAM-DRIVEN SINKHORN DESCENT
# ------------------------------------------------------------------
def wasserstein_adam(
    X0: pt.Tensor,                        # (N,d)  initial cloud on *device*
    timestepper,
    n_epochs: int,
    batch_size: int,
    lr: float,           # Adam base learning-rate
    replicas: int,
    device=pt.device("mps"),
    store_directory: str | None = None):
    """
    Sinkhorn-descent steady-state solver driven by Adam.

    Returns
    -------
    X_final  : (N,d) tensor on CPU
    losses   : list of per-batch ½ S_ε values
    """
    batched_timestepper = pt.vmap(timestepper, randomness="different")
    X_param = pt.nn.Parameter(X0.clone())
    N, d    = X_param.shape

    print(f"[INFO] Initial Loss", w2_loss_1d(X_param, batched_timestepper, replicas).item())

    lr_now = lr
    lr_decrease_step = 2500
    lr_decrease_factor = 0.1
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

        perm = pt.randperm(N, device=device)          # shuffle indices for batching

        for start in range(0, N, batch_size):
            idx   = perm[start:start+batch_size]
            x_sub = X_param[idx]                      # view on parameter

            # ----- forward + backward ----------------------------
            opt.zero_grad()
            loss = w2_loss_1d(x_sub, batched_timestepper, replicas)

            # Only gradients for rows in idx are non-zero; adam sees them
            loss.backward() # autograd on full parameter
            grad_norm = X_param.grad.norm().item()

            opt.step()
            losses.append(loss.item())
            grad_norms.append(grad_norm)
        
        # ---- Update the learning rate after each epoch ----------
        sched.step()
        lr_now = opt.param_groups[0]['lr']

        # ---- inexpensive displacement diagnostic ----------------
        with pt.no_grad():
            probe_idx = pt.randperm(N, device=device)[:min(batch_size, N)]
            disp = (timestepper(X_param[probe_idx]) - X_param[probe_idx]).norm(dim=1).mean().item()
            print(f"last ½S_ε = {loss.item():.4e} | "
                f"‖grad‖₂={grad_norm:.3e} | "
                f"lr={lr_now:.2e} | "
                f"⟨|x-φ(x)|⟩ = {disp:.4e}")

    # Store the loss history
    pt.save(pt.stack((pt.tensor(losses), pt.tensor(grad_norms))), os.path.join(store_directory or ".", "wasserstein_adam_losses.pt"))

    return X_param.detach().cpu(), losses, grad_norms

def _call_loss(X0: pt.Tensor, timestepper, batch_size: int, replicas: int, device: str | pt.device = "mps"):
    batched_timestepper = pt.vmap(timestepper, randomness="different")

    # Act as if we're doing minibatching
    N, d = X0.shape
    perm = pt.randperm(N, device=device)          # shuffle indices
    idx = perm[0:batch_size]
    x_sub = X0[idx]

    # Call the loss function
    loss = w2_loss_1d(x_sub, batched_timestepper, replicas)

    return loss
