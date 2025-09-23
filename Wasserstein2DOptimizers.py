import torch as pt
import scipy.optimize as opt

from typing import Tuple, List

import torch as pt

def hungarian_permutation(cost: pt.Tensor) -> pt.Tensor:
    """
    Solve min_{perm} sum_i cost[i, perm[i]] using SciPy's Hungarian algorithm if available.
    Returns a LongTensor 'perm' with shape (N,) such that row i is matched to column perm[i].
    Falls back to a greedy approximation if SciPy is unavailable.
    """
    ri, cj = opt.linear_sum_assignment(cost.detach().cpu().numpy())
    perm = pt.as_tensor(cj, dtype=pt.long, device=cost.device)
    return perm

def w2_loss_2d(
    x: pt.Tensor,    # (B,2)   requires_grad = False
    timestepper,     # (B,2) -> (B,2)
) -> pt.Tensor:
    """
    Returns scalar ½ W₂²(X, φ_T(X)) for a single cloud in 2D.
    Uses the optimal assignment (permutation) to pair X with Y = φ_T(X).
    Gradient w.r.t. X is X - Y_perm (averaged over particles); no gradient flows through the matching.
    """
    assert x.ndim == 2 and x.shape[1] == 2, "x must have shape (B, 2)."

    # 1) Push particles through the timestepper once
    y = timestepper(x).detach() # (B, 2)

    # 2) Build the squared Euclidean cost matrix C_ij = ||x_i - y_j||^2
    #    Use torch.cdist for efficiency and numerical stability.
    C = pt.cdist(x, y, p=2).pow(2)  # (B, B)

    # 3) Solve the linear assignment to get the optimal permutation
    perm = hungarian_permutation(C)       # (B,)

    # 4) Reorder Y according to the optimal matching and form the loss
    y_matched = y.index_select(0, perm)           # (B, 2)
    diff = x - y_matched                          # (B, 2)
    loss = 0.5 * diff.pow(2).sum(dim=1).mean()    # scalar (½ * mean squared distance)

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
    device=pt.device("mps")) -> Tuple[pt.Tensor, List, List]:
    """
    Wasserstein-descent steady-state solver driven by Adam.

    Returns
    -------
    X_final   : (N,d) tensor on CPU
    losses    : list of per-batch ½ W_2^2 values
    grad_norms: list of per-batch loss gradients
    """
    X_param = pt.nn.Parameter(X0.clone()).to(device)
    N, d    = X_param.shape

    print(f"[INFO] Initial Loss", w2_loss_2d(X_param, timestepper).item())

    lr_now = lr
    opt = pt.optim.Adam([X_param], lr=lr, betas=(0.9, 0.999))
    sched = pt.optim.lr_scheduler.StepLR(opt, step_size=lr_decrease_step, gamma=lr_decrease_factor)
    
    losses = []
    grad_norms = []
    for epoch in range(1, n_epochs+1):
        print(f"Epoch {epoch:3d}")
        perm = pt.randperm(N, device=device)

        for start in range(0, N, batch_size):
            idx   = perm[start:start+batch_size]
            x_sub = X_param[idx]
            
            opt.zero_grad()
            loss = w2_loss_2d(x_sub, timestepper)
            loss.backward()
            if X_param.grad is not None:
                grad_norm = X_param.grad.norm().item()
            else:
                print('Wasserstein gradient not available.')
                grad_norm = float("nan")

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
            print(f"last ½W = {loss.item():.4e} | "
                f"‖grad‖₂={grad_norm:.3e} | "
                f"lr={lr_now:.2e} | "
                f"⟨|x-φ(x)|⟩ = {disp:.4e}")

    return X_param.detach().cpu(), losses, grad_norms