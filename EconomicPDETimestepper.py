import numpy as np

def step(rho, x_faces, dt, gamma, c, sigma):
    N = len(x_faces) - 1 # We have N midpoints
    assert N % 2 == 1 # Ensure x = 0 is a midpoint!
    h = 2.0 / N
    i0 = N // 2

    # Calculate the values and gradient of the local density at the cell boundaries
    mu = gamma * x_faces - c
    rho_upwind = np.where(mu[1:-1] > 0, rho[:-1],  rho[1:])
    rho_grad = (rho[1:] - rho[:-1]) / h

    dt_max = 0.4 * min(h**2 / sigma**2, h / np.max(np.abs(gamma * x_faces - c)))
    #print('dt_max', dt_max)

    # Compute the fluxes at the interior cell boundaries, and its gradient in the cell centers
    J_inner = -mu[1:-1] * rho_upwind - 0.5 * sigma**2 * rho_grad

    # left boundary  (face 0, x = -1)
    rho_up_L  = 0.0 if mu[0] > 0 else rho[0]
    J_left    = (-mu[0] * rho_up_L - 0.5 * sigma**2 * (rho[0] - 0.0) / h)

    # right boundary (face N, x = 1)
    rho_up_R  = 0.0 if mu[-1] < 0 else rho[-1]
    J_right   = (-mu[-1] * rho_up_R - 0.5 * sigma**2 * (0.0 - rho[-1]) / h)

    # Calculate R+, R- and S
    drho_dx_right = (0.0 - rho[-1]) / h
    Rplus = -0.5 * sigma**2 * drho_dx_right
    drho_dx_left = (rho[0] - 0.0) / h
    Rminus = 0.5 * sigma**2 * drho_dx_left
    S = Rplus + Rminus

    # Update rho everywhere
    J = np.concatenate(([J_left], J_inner, [J_right]))
    rho_new = rho - dt * (J[1:] - J[:-1]) / h
    rho_new[i0] += dt * S / h # Approximate the delta function by a block around x = 0
    #print(np.any(np.isnan(rho_new)))

    # Ensure the new density is normalized
    area = np.sum(rho_new * h)
    rho_new = rho_new / area

    return rho_new

def PDETimestepper(rho0, x_faces, dt, T, gamma, c, sigma):
    rho = np.copy(rho0)

    n_steps = int(T / dt)
    for n in range(n_steps):
        rho = step(rho, x_faces, dt, gamma, c, sigma)
        print('t =', (n+1)*dt)
    
    return rho