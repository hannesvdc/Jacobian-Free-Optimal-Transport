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

def call_loss(X0: pt.Tensor, timestepper, batch_size: int, replicas: int, device: str | pt.device = "mps"):
    batched_timestepper = pt.vmap(timestepper, randomness="different")

    # Act as if we're doing minibatching
    N, d = X0.shape
    perm = pt.randperm(N, device=device)          # shuffle indices
    idx = perm[0:batch_size]
    x_sub = X0[idx]

    # Call the loss function
    loss = w2_loss_1d(x_sub, batched_timestepper, replicas)

    return loss
