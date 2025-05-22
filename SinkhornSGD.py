# sinkhorn_sgd.py  ----------------------------------------------------------
# Minimal Sinkhorn-descent steady-state solver (SGD version only)
#
# Dependencies:   pip install torch geomloss
#   • PyTorch        – array engine + autograd
#   • GeomLoss       – entropic Sinkhorn divergence + gradients
# --------------------------------------------------------------------------

from typing import Sequence
import torch as pt
from geomloss import SamplesLoss


# --------------------------------------------------------------------------
# 1 ─ SINKHORN LOSS (½ S_eps) AND GRADIENT ON A MINI-BATCH
# --------------------------------------------------------------------------
def sinkhorn_loss_and_grad(
    x_batch: pt.Tensor,                  # (B, d) – requires_grad=False
    timestepper,
    eps_blur: float,
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

    loss_fn = SamplesLoss(
        loss   ="sinkhorn",
        p      = 2,
        blur   = eps_blur,
        debias = True,                      # Sinkhorn *divergence*
        scaling= 0.9,                       # ε-scaling warm start
        backend="tensorized",               # fast for B ≲ 20 000
    )
    loss = 0.5 * loss_fn(x, y)              # ½ S_ε for our objective
    loss.backward()                         # d(loss)/dx ∈ x.grad
    return loss.detach(), x.grad.detach()


# --------------------------------------------------------------------------
# 2 ─ MAIN SGD DRIVER
# --------------------------------------------------------------------------
def sinkhorn_sgd(
    X0: pt.Tensor,                       # (N, d) initial cloud on cpu
    timestepper,
    n_epochs: int = 30,
    batch_size: int = 12_000,
    step_size: float = 0.3,
    eps_blur: float | None = None,
    replicas: int = 1,
    tol_disp: float = 1e-3,
    device: str | pt.device = "mps",
) -> tuple[pt.Tensor, Sequence[float]]:
    """
    Stochastic-gradient Sinkhorn descent.
    Returns (final_cloud_on_cpu, list_of_loss_values).
    """
    # Move data to device
    X = X0.to(device).clone()
    N, d = X.shape

    # Choose ε if not provided: 2 % of median squared distance of a sample - this is a well-motivated value
    if eps_blur is None:
        with pt.no_grad():
            sample = X[pt.randperm(N, device=device)[:min(10_000, N)]]
            med_sq = pt.cdist(sample, sample).median().item() ** 2
        eps_blur = 0.02 * med_sq

    print(f"[INFO] blur ε = {eps_blur:.4e}")

    losses: list[float] = []

    for epoch in range(n_epochs):
        # Shuffle once per epoch
        perm = pt.randperm(N, device=device)

        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            loss, grad = sinkhorn_loss_and_grad(
                X[idx], timestepper, eps_blur, replicas
            )

            # SGD update
            with pt.no_grad():
                X[idx] -= step_size * grad

            losses.append(loss.item())

        # --- monitor average displacement (cheap diagnostic) -------------
        with pt.no_grad():
            probe_idx = pt.randperm(N, device=device)[:min(10_000, N)]
            disp = (timestepper(X[probe_idx]) - X[probe_idx]).norm(dim=1).mean().item()

        print(f"epoch {epoch:3d} | last minibatch ½S_ε={loss.item():.4e} "
              f"| ⟨|x−φ(x)|⟩={disp:.4e}")

        if disp < tol_disp:
            print(f"[INFO] convergence: ⟨|x−φ(x)|⟩ < {tol_disp}")
            break

    return X.cpu(), losses