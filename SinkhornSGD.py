# sinkhorn_sgd.py  ----------------------------------------------------------
# Minimal Sinkhorn-descent steady-state solver (SGD version only)
#
# Dependencies:   pip3 install torch geomloss
#   • PyTorch        – array engine + autograd
#   • GeomLoss       – entropic Sinkhorn divergence + gradients
# --------------------------------------------------------------------------
import os
from typing import Sequence
import torch as pt
from geomloss import SamplesLoss

def choose_eps_blur(x: pt.Tensor,
                    timestepper,
                    sample_size=10_000,
                    multiplier=3.0) -> float:
    """
    ε = multiplier × median‖x - φ_T(x)‖²   (φ_T may be stochastic).
    """
    N = x.size(0)
    idx = pt.randperm(N, device=x.device)[:min(sample_size, N)]
    d2  = ((x[idx] - timestepper(x[idx]))**2).sum(1)
    print('d2', d2.median())
    return multiplier * d2.median().item()

# --------------------------------------------------------------------------
# 1 ─ SINKHORN LOSS (½ S_eps) AND GRADIENT ON A MINI-BATCH
# --------------------------------------------------------------------------
def sinkhorn_loss(
    x_batch: pt.Tensor,                  # (B, d) – requires_grad=False
    timestepper,
    loss_fn,
    replicas: int = 1,
) -> tuple[pt.Tensor, pt.Tensor]:
    """
    Returns the scalar ½ Sε on a mini-batch.
    Autograd on x_batch is left intact; the caller decides when to
    zero-grad and call backward().
    """

    # Monte-Carlo average over images of the batch
    y = pt.stack([timestepper(x_batch) for _ in range(replicas)]).mean(0).detach()

    loss = 0.5 * loss_fn(x_batch, y)
    return loss

def sinkhorn_loss_and_grad(
    x_batch: pt.Tensor,                  # (B, d) – requires_grad=False
    timestepper,
    loss_fn,
    replicas: int = 1,
) -> tuple[pt.Tensor, pt.Tensor]:
    """
    Returns (loss_scalar,  gradient dL/dx_batch)  for one mini-batch.
    • We *detach* Y so that φ_T's Jacobian is NOT propagated.
    • If φ_T is stochastic, average over `n_kernel_draws` images.
    """
    x = x_batch.detach().requires_grad_()   # turn on autograd

    # Monte-Carlo average over images of the batch
    y = pt.stack([timestepper(x) for _ in range(replicas)]).mean(0).detach()

    loss = 0.5 * loss_fn(x, y)              # ½ S_ε for our objective
    loss.backward()                         # x.grad = d(loss)/dx
    return loss.detach(), x.grad.detach()

# --------------------------------------------------------------------------
# 2 ─ MAIN SGD DRIVER
# --------------------------------------------------------------------------
def sinkhorn_sgd(
    X0: pt.Tensor,                       # (N, d) initial cloud on cpu
    timestepper,
    n_epochs: int = 30,
    batch_size: int = 10_000,
    step_size: float = 0.3,
    replicas: int = 1,
    device: str | pt.device = "mps",
    store_directory=None
) -> tuple[pt.Tensor, Sequence[float]]:
    """
    Stochastic-gradient Sinkhorn descent.
    Returns (final_cloud_on_cpu, list_of_loss_values).
    """
    # Move data to device
    X = X0.to(device).clone()
    N, d = X.shape

    # Choose ε if not provided
    eps = choose_eps_blur(X, timestepper, multiplier=1.0)
    loss_fn = SamplesLoss(
        loss   = "sinkhorn",
        p      = 2,
        blur   = eps,
        debias = True,                      # Sinkhorn *divergence*
        scaling= 0.9,                       # ε-scaling warm start
        backend="tensorized",               # fast for B ≲ 20 000
    )
    print(f"[INFO] blur ε = {eps:.4e}")
    print(f"[INFO] Initial Loss", sinkhorn_loss_and_grad(X, timestepper, loss_fn, replicas)[0].item())

    losses: list[float] = []
    for epoch in range(n_epochs):
        print('Epoch #', epoch)

        if store_directory is not None and epoch % 200 == 0:
            pt.save(X.cpu(), store_directory + "particles.pt")

        # Shuffle once per epoch
        perm = pt.randperm(N, device=device)

        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            loss, grad = sinkhorn_loss_and_grad(X[idx], timestepper, loss_fn, replicas)

            # SGD update
            with pt.no_grad():
                X[idx] -= step_size * grad / pt.norm(grad)
            losses.append(loss.item())

        # --- monitor average displacement (cheap diagnostic) -------------
        with pt.no_grad():
            probe_idx = pt.randperm(N, device=device)[:min(10_000, N)]
            disp = (timestepper(X[probe_idx]) - X[probe_idx]).norm(dim=1).mean().item()

        print(f"epoch {epoch:3d} | last minibatch ½S_ε={loss.item():.4e} "
              f"| ⟨|x−φ(x)|⟩={disp:.4e}")
        
    pt.save(pt.Tensor(losses), store_directory + "sinkhorn_losses.pt")
    return X.cpu(), losses

# ------------------------------------------------------------------
#  ADAM-DRIVEN SINKHORN DESCENT
# ------------------------------------------------------------------
def sinkhorn_adam(
    X0: pt.Tensor,                        # (N,d)  initial cloud on *CPU*
    timestepper,
    n_epochs: int       = 30,
    batch_size: int     = 10_000,
    lr: float           = 0.1,           # Adam base learning-rate
    replicas: int       = 1,
    device: str | pt.device = "mps",
    store_directory: str | None = None
):
    """
    Sinkhorn-descent steady-state solver driven by Adam.

    Returns
    -------
    X_final  : (N,d) tensor on CPU
    losses   : list of per-batch ½ S_ε values
    """
    X_param = pt.nn.Parameter(X0.to(device).clone())
    N, d    = X_param.shape

    eps = choose_eps_blur(X_param.data, timestepper, multiplier=1.0)
    loss_fn = SamplesLoss("sinkhorn", p=2, blur=eps,
                          debias=True, backend="tensorized", scaling=0.9)
    print(f"[INFO] blur ε = {eps:.4e}")

    opt = pt.optim.Adam([X_param], lr=lr, betas=(0.9, 0.999))
    def save_cloud(epoch):
        if store_directory is None: return
        pt.save(X_param.detach().cpu(),
                os.path.join(store_directory, f"particles_e{epoch:04d}.pt"))

    losses = []
    for epoch in range(n_epochs):
        print(f"Epoch {epoch:3d}")
        save_cloud(epoch)

        perm = pt.randperm(N, device=device)          # shuffle indices

        for start in range(0, N, batch_size):
            idx   = perm[start:start+batch_size]
            x_sub = X_param[idx]                      # view on parameter

            # ----- forward + backward ----------------------------
            opt.zero_grad()
            loss = sinkhorn_loss(x_sub, timestepper, loss_fn, replicas)

            # Only gradients for rows in idx are non-zero; adam sees them
            loss.backward()           # autograd on full parameter
            opt.step()

            losses.append(loss.item())

        # ---- inexpensive displacement diagnostic ----------------
        with pt.no_grad():
            probe_idx = pt.randperm(N, device=device)[:min(10_000, N)]
            disp = (timestepper(X_param[probe_idx]) -
                    X_param[probe_idx]).norm(dim=1).mean().item()

        print(f"  last ½S_ε = {loss.item():.4e} | "
              f"⟨|x−φ(x)|⟩ = {disp:.4e}")

    save_cloud(n_epochs)
    pt.save(pt.tensor(losses),
            os.path.join(store_directory or ".", "sinkhorn_losses.pt"))

    return X_param.detach().cpu(), losses