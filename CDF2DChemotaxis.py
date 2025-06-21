import numpy as np
import matplotlib.pyplot as plt

from CDF2DOptimizers import particles_from_joint_cdf_cubic, empirical_joint_cdf_on_grid, cdf_newton_krylov

# Define and Calculate the Potential Energy Function
def half_moon_potential_meshgrid(x, y, A, R, B, alpha, y_shift):
    """
    Calculates the half-moon potential energy at given (x, y) coordinates.
    """
    r = np.sqrt(x**2 + y**2)  # Radial distance from the origin
    U_radial = A * (r - R)**2 # Radial term: a parabolic well at radius R
    U_wall = B * np.exp(-alpha * (y - y_shift)) # Wall term: an exponential wall for y < y_shift
    
    return U_radial + U_wall

# Evaluated in particles Z = (X, Y)
def half_moon_potential(z, A, R, B, alpha, y_shift):
    r = np.sqrt(z[:,0]**2 + z[:,1]**2)
    U_radial = A * (r - R)**2
    U_wall   = B * np.exp(-alpha * (z[:,1] - y_shift))
    return U_radial + U_wall

# Evaluated in particles Z = (X, Y)
def gradient_half_moon_potential(z, A, R, B, alpha, y_shift):
    r = np.sqrt(z[:,0]**2 + z[:,1]**2)
    radial_prefactor = 2 * A * (r - R) / r
        
    grad_radial_x = radial_prefactor * z[:,0]
    grad_radial_y = radial_prefactor * z[:,1]
    
    # Note: The wall gradient is zero for the x-component.
    grad_wall_y = -alpha * B * np.exp(-alpha * (z[:,1] - y_shift))
    
    grad_x = grad_radial_x
    grad_y = grad_radial_y + grad_wall_y
    return np.column_stack((grad_x, grad_y))

def reflect(coords, L):
    """vectorised reflection of an (N,2) array into [-L,L]^2."""
    coords = coords.copy()
    over   = coords >  L
    under  = coords < -L
    coords[over]  =  2*L - coords[over]
    coords[under] = -2*L - coords[under]
    return coords


def step(z, dt, rng, A, R, B, alpha, y_shift, L):
    """
    z : (N,2) array of particle positions
    returns updated (N,2) array
    """
    z = reflect(z, L)
    z_new = z - gradient_half_moon_potential(z, A, R, B, alpha, y_shift) * dt + np.sqrt(2.0 * dt) * rng.normal(size=z.shape)
    z_new = reflect(z_new, L)
    return z_new

def timestepper(z, dt, T, rng, A, R, B, alpha, y_shift, L=4.0):
    """
    z0 : initial position (array-like length 2)
    dt : time step
    T  : final time
    rng: np.random.Generator
    returns path array shape (n_steps+1, 2)
    """
    n_steps = int(np.ceil(T / dt))

    for n in range(1, n_steps + 1):
        z = step(z, dt, rng, A=A, R=R, B=B, alpha=alpha, y_shift=y_shift, L=L)
    return z

def timeEvolution():
    rng = np.random.RandomState()
    R = 2.0
    A = 2.0
    B = 0.5
    alpha = 1.5
    y_shift = -0.5

    # Start with a standard normal Gaussian
    x_min, x_max = -4, 4
    y_min, y_max = -4, 4
    grid_points = 201
    x_grid = np.linspace(x_min, x_max, grid_points)
    y_grid = np.linspace(y_min, y_max, grid_points)
    X, Y = np.meshgrid(x_grid, y_grid)
    U = X**2 / 2.0 + Y**2 / 2.0
    prob_density = np.exp(-U)
    cdf0 = prob_density.cumsum(axis=0).cumsum(axis=1)
    cdf0 /= cdf0[-1,-1]
    print('cdf', cdf0)

    # Build the density-to-density timestepper
    N = 100172
    dt = 1.e-3
    T_psi = 1.0
    def cdf_timestepper(cdf):
        particles = particles_from_joint_cdf_cubic(x_grid, y_grid, cdf, N, 1.e-10)
        new_particles = timestepper(particles, dt, T_psi, rng, A, R, B, alpha, y_shift, L=4.0)
        return empirical_joint_cdf_on_grid(new_particles, x_grid, y_grid)
    
    # Do timestepping up to 500 seconds
    T = 500.0
    n_steps = int(T / T_psi)
    cdf = np.copy(cdf0)
    for n in range(n_steps):
        print('t =', n*T_psi)
        cdf = cdf_timestepper(cdf)
    print('t =', T)

    # Generate samples from the final CDF for plotting the density
    particles = particles_from_joint_cdf_cubic(x_grid, y_grid, cdf, N)
    H, x_edges, y_edges = np.histogram2d(particles[:,0], particles[:,1], density=True, range=[[x_min, x_max], [y_min, y_max]], bins=[100,100])
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

    # Plot a histogram of the sampled particles
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xc, Yc = np.meshgrid(x_centres, y_centres, indexing="ij")
    xpos, ypos = Xc.ravel(), Yc.ravel()
    zpos       = np.zeros_like(xpos)
    dx = (x_edges[1] - x_edges[0]) * np.ones_like(xpos)#type: ignore
    dy = (y_edges[1] - y_edges[0]) * np.ones_like(ypos)#type: ignore
    dz = H.ravel()
    norm   = plt.Normalize(dz.min(), dz.max())#type: ignore
    colours = cm.viridis(norm(dz))#type: ignore
    fig3d = plt.figure(figsize=(7, 6))
    ax3d  = fig3d.add_subplot(111, projection="3d", proj_type="ortho")
    ax3d.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colours, shade=True)#type: ignore
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("density")#type: ignore
    ax3d.set_title(f"3-D Histogram", pad=12)
    ax3d.view_init(elev=40, azim=-55)#type: ignore
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    timeEvolution()