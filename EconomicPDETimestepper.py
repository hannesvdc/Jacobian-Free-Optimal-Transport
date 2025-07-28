import numpy as np
import numpy.linalg as lg

def step(rho, x_faces, dt, gamma, vplus, vminus, eplus, eminus, g):
    N = len(x_faces) - 1 # We have N midpoints
    assert N % 2 == 1 # Ensure x = 0 is a midpoint!
    h = 2.0 / N

    # Calculate the temporal model parameters
    drho_dx_right = (0.0 - rho[-1]) / h
    drho_dx_left = (rho[0] - 0.0) / h
    Q_matrix = np.array([[1.0 + 0.5 * gamma * eplus**2 * drho_dx_right, 0.5 * gamma * eminus**2 * drho_dx_right], \
                         [-0.5*gamma*eplus**2 * drho_dx_left, 1.0 - 0.5 * gamma * eminus**2 * drho_dx_left]])
    velocities = lg.solve(Q_matrix, np.array([vplus, vminus]))
    sigmasq = velocities[0] * eplus**2 + velocities[1] * eminus**2
    Rplus = -0.5 * sigmasq * drho_dx_right
    Rminus = 0.5 * sigmasq * drho_dx_left
    S = Rplus + Rminus

    # Calculate the values and gradient of the local density at the cell boundaries
    c = velocities[0] * eplus + velocities[1] * eminus
    mu = gamma * x_faces - c
    rho_upwind = np.where(mu[1:-1] > 0, rho[:-1],  rho[1:])
    rho_grad = (rho[1:] - rho[:-1]) / h

    # Compute the fluxes at the interior cell boundaries, and its gradient in the cell centers
    J_inner = -mu[1:-1] * rho_upwind - 0.5 * sigmasq * rho_grad

    # left boundary  (face 0, x = -1)
    rho_up_L  = 0.0 if mu[0] > 0 else rho[0]
    J_left    = (-mu[0] * rho_up_L - 0.5 * sigmasq * (rho[0] - 0.0) / h)

    # right boundary (face N, x = 1)
    rho_up_R  = 0.0 if mu[-1] < 0 else rho[-1]
    J_right   = (-mu[-1] * rho_up_R - 0.5 * sigmasq * (0.0 - rho[-1]) / h)

    # Update rho everywhere
    J = np.concatenate(([J_left], J_inner, [J_right]))
    rho_new = rho - dt * (J[1:] - J[:-1]) / h

    # Approximate the delta function by a Gaussian
    eps = 0.01
    x_centres = 0.5 * (x_faces[1:] + x_faces[:-1])
    gaussian = np.exp(-x_centres**2 / (2.0 * eps**2)) / np.sqrt(2.0 * np.pi * eps**2)
    rho_new += dt * S * gaussian
    #print(np.any(np.isnan(rho_new)))

    # Ensure the new density is normalized
    area = np.sum(rho_new * h)
    rho_new = rho_new / area

    return rho_new

def FV_step(rho, x_faces, dt, gamma, vpc, vmc, eplus, eminus, g):
    N = len(x_faces) - 1 # We have N midpoints
    assert N % 2 == 1 # Ensure x = 0 is a midpoint!
    h = 2.0 / N

    # Calculate the temporal model parameters
    drho_dx_right = (0.0 - rho[-1]) / h
    drho_dx_left = (rho[0] - 0.0) / h
    Q_matrix = np.array([[1.0 + 0.5 * gamma * eplus**2 * drho_dx_right, 0.5 * gamma * eminus**2 * drho_dx_right], \
                         [-0.5 * gamma * eplus**2 * drho_dx_left, 1.0 - 0.5 * gamma * eminus**2 * drho_dx_left]])
    velocities = lg.solve(Q_matrix, np.array([vpc, vmc]))
    sigmasq = velocities[0] * eplus**2 + velocities[1] * eminus**2
    Rplus = -0.5 * sigmasq * drho_dx_right
    Rminus = 0.5 * sigmasq * drho_dx_left
    S = Rplus + Rminus

    # Calculate the fluxes and resulting advection term in the cells
    c = velocities[0] * eplus + velocities[1] * eminus
    mu = gamma * x_faces - c
    F_inner = np.where(mu[1:-1] > 0, rho[:-1] * mu[1:-1],  rho[1:] * mu[1:-1])
    F_left = 0.0 if mu[0] > 0 else rho[0] * mu[0]
    F_right = 0.0 if mu[-1] > 0 else rho[-1] * mu[-1]
    F = np.concatenate(([F_left], F_inner, [F_right]))
    advection = (F[1:] - F[:-1]) / h

    # Calculate the diffusion term
    rho_ext = np.concatenate(([0.0], rho, [0.0]))
    diffusion = 0.5 * sigmasq * (np.roll(rho_ext, 1) - 2.0 * rho_ext + np.roll(rho_ext, -1)) / h**2
    diffusion = diffusion[1:-1]

    # Add all terms together
    i0 = N // 2
    rho_new = rho + dt * (advection + diffusion)
    rho_new[i0] += dt * S / h # Delta at 0
    rho_new = np.maximum(rho_new, 0.0)
    area = h * np.sum(rho_new)
    rho_new /= area

    # Return the new density
    return rho_new, Rminus, Rplus

def PDETimestepper(rho0, x_faces, dt, T, gamma, vplus, vminus, eplus, eminus, g):
    rho = np.copy(rho0)
    vmc = vminus
    vpc = vplus

    n_steps = int(T / dt)
    for n in range(n_steps):
        rho, Rminus, Rplus = FV_step(rho, x_faces, dt, gamma, vpc, vmc, eplus, eminus, g)
        vpc  = vplus + g * Rplus
        vmc = vminus + g * Rminus
        print('t =', (n+1)*dt)
    
    return rho