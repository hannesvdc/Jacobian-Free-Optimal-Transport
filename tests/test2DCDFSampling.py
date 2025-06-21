import sys
sys.path.append('../')

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

from CDF2DOptimizers import particles_from_joint_cdf_cubic

EPS = 1.e-10

def testGaussianBimodal():
    # Setup the potential energy for this distribution
    gaussian_V = lambda x: (x - 1.0)**2 / (2.0 * 0.5**2) 
    bimodal_V = lambda y: 0.5 * (y**2 - 1)**2
    unnormalize_density = lambda x, y: np.exp(-gaussian_V(x) - bimodal_V(y))

    # Build the cumulative density
    n_points = 101
    x_min = -1
    x_max = 3
    y_min = -3
    y_max = 3
    grid_x = np.linspace(x_min, x_max, n_points)
    grid_y = np.linspace(y_min, y_max, n_points)
    dx = 2.0 * (x_max - x_min) / (n_points - 1)
    dy = 2.0 * (y_max - y_min) / (n_points - 1)
    X, Y = np.meshgrid(grid_x, grid_y)
    density_grid = unnormalize_density(X, Y).transpose() * dx * dy
    cdf_grid = density_grid.cumsum(axis=0).cumsum(axis=1)
    cdf_grid = cdf_grid / cdf_grid[-1, -1] # Rescale to ensure final value is 1
    cdf_grid = np.where(cdf_grid < EPS, 0.0, cdf_grid)

    # Lifting: generate particles
    N = 10**5
    particles = particles_from_joint_cdf_cubic(grid_x, grid_y, cdf_grid, N, eps=EPS)
    particles_x = particles[:,0]
    particles_y = particles[:,1]

    # Marginalize and plot in each dimension separately (there is no 'correlation' between X and Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, cdf_grid, cmap='viridis') # type: ignore
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    plt.figure()
    x_plot_array = np.linspace(x_min, x_max, 1001)
    x_density = np.exp(-gaussian_V(x_plot_array))
    x_density /= np.trapz(x_density, x_plot_array)
    plt.hist(particles_x, bins=40, density=True, label=r'CDF Sampling $X$')
    plt.plot(x_plot_array, x_density, label=r'Marginal Distribution in $X$')
    plt.xlabel(r'$x$')
    plt.legend()

    plt.figure()
    y_plot_array = np.linspace(y_min, y_max, 1001)
    y_density = np.exp(-bimodal_V(y_plot_array))
    y_density /= np.trapz(y_density, y_plot_array)
    plt.hist(particles_y, bins=40, density=True, label=r'CDF Sampling $Y$')
    plt.plot(y_plot_array, y_density, label=r'Marginal Distribution in $Y$')
    plt.xlabel(r'$y$')
    plt.legend()

    plt.show()

def testHalfMoonSampling():

    # Define Parameters for the Potential
    R = 2.0    # Radius of the half-moon
    A = 2.0    # Controls the width of the moon (larger A -> narrower moon)
    B = 0.5    # Controls the height of the wall (dissuading y < 0)
    alpha = 1.5  # Controls the steepness of the wall at y=0
    y_shift = -0.5

    # Make the domain large enough to see the potential rise at the edges
    x_min, x_max = -4, 4
    y_min, y_max = -4, 4
    grid_points = 200

    x = np.linspace(x_min, x_max, grid_points)
    y = np.linspace(y_min, y_max, grid_points)
    X, Y = np.meshgrid(x, y)

    # Define and Calculate the Potential Energy Function
    def half_moon_potential(x, y, A, R, B, alpha):
        """
        Calculates the half-moon potential energy at given (x, y) coordinates.
        """
        r = np.sqrt(x**2 + y**2)  # Radial distance from the origin
        U_radial = A * (r - R)**2 # Radial term: a parabolic well at radius R
        U_wall = B * np.exp(-alpha * (y - y_shift)) # Wall term: an exponential wall for y < y_shift
        
        return U_radial + U_wall

    # Calculate the potential energy and density
    U = half_moon_potential(X, Y, A, R, B, alpha)
    prob_density = np.exp(-U)
    cdf = prob_density.cumsum(axis=0).cumsum(axis=1)
    cdf /= cdf[-1,-1]
    print('cdf', cdf)

    # Compute the marginal of x for plotting purposes
    marginal_density_x = np.zeros_like(x)
    for index in range(grid_points):
        marginal_density_x[index] = np.trapz(prob_density[index,:], y)
    marginal_density_x /= np.trapz(marginal_density_x, x)

    # Sample the CDF
    N = 10**6
    particles = particles_from_joint_cdf_cubic(x, y, cdf, N, EPS)
    H, x_edges, y_edges = np.histogram2d(particles[:,0], particles[:,1], density=True, range=[[x_min, x_max], [y_min, y_max]], bins=[100,100])

    # Plot the Probability Density as a 3D Surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, prob_density.T, cmap=cm.viridis, linewidth=0, antialiased=False, rstride=2, cstride=2) #type: ignore
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Probability Density')#type: ignore
    ax.set_title('3D View of Half-Moon Probability Density', pad=20)
    ax.view_init(elev=45, azim=-65)#type: ignore
    fig.colorbar(surf, shrink=0.6, aspect=10, label='Probability Density $\propto e^{-U(x,y)}$')

    # --- 2-D heat-map ---------------------------------------------------
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

    # Plot the marginal of x and its samples
    plt.figure()
    plt.plot(x, marginal_density_x, label='Marginal Density X')
    plt.hist(particles[:,0], bins=100, density=True, label='Particles')
    plt.xlabel(r'$x$')
    plt.title('Marginal Density X')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    testHalfMoonSampling()