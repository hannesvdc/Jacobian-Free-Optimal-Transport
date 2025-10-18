import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

alpha = 0.5

def add_theta_arc(ax, theta, radius=0.6, color='blue',
                  label=r'$\theta$', z=0.0):
    """
    Draw a dashed circular arc in the xy-plane from angle 0 to `theta`,
    add a small arrowhead, and place a label.
    """
    # Slight lift to avoid z-fighting with the plane (if any)
    z = z + 1e-3

    # Arc
    t = np.linspace(0, theta, 100)
    x = radius*np.cos(t)
    y = radius*np.sin(t)
    ax.plot(x, y, np.full_like(t, z), linestyle='--', linewidth=1.2, color='k', alpha=alpha)

    # Arrowhead via a tiny tangent quiver at the arc end
    xe, ye = radius*np.cos(theta), radius*np.sin(theta)
    dx, dy = -radius*np.sin(theta), radius*np.cos(theta)   # tangent vector
    v = np.hypot(dx, dy)
    scale = 0.12*radius / v
    #ax.quiver(xe, ye, z, dx*scale, dy*scale, 0,
    #          arrow_length_ratio=0.4, color=color, linewidth=1)

    # Label a bit outside the middle of the arc
    tm = 0.55*theta
    xl = (radius+0.08)*np.cos(tm)
    yl = (radius+0.08)*np.sin(tm)
    ax.text(xl, yl, z, label, color='k', fontsize=11, ha='center', va='center', alpha=alpha)

# Define two rays at different angles
theta1, theta2 = np.deg2rad(25), np.deg2rad(70)
r = np.linspace(0, 1, 100)

# Define coordinates for the rays
x1, y1 = r * np.cos(theta1), r * np.sin(theta1)
x2, y2 = r * np.cos(theta2), r * np.sin(theta2)

# Define artificial 1D CDFs (values along z)
f = 5.0
z1 = 1.0 * np.sqrt(r) * (1.0 + 0.1*np.sin(f*r))
z2 = 1.0 * np.sqrt(r) * (1.0 + 0.1*np.sin(1.1*f*r))

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')

# Draw 2D grid (xy-plane)
xg, yg = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
#ax.plot_surface(xg, yg, np.zeros_like(xg), alpha=0.15, color='gray')

# Plot the rays
ax.plot(x1, y1, np.zeros_like(x1), 'b', lw=1.5, alpha=alpha)
ax.plot(x2, y2, np.zeros_like(x2), 'r', lw=1.5, alpha=alpha)

# Plot the 1D CDFs along the rays
ax.plot(x1, y1, z1, 'b', lw=2, alpha=alpha)
ax.plot(x2, y2, z2, 'r', lw=2, alpha=alpha)

# Grid points on rays
ax.scatter(x1[::10], y1[::10], np.zeros_like(x1)[::10], color='blue', s=25, alpha=alpha)
ax.scatter(x2[::10], y2[::10], np.zeros_like(x2)[::10], color='red', s=25, alpha=alpha)

# Label the angles
#ax.text(0.5*np.cos(theta1), 0.5*np.sin(theta1), 0, r"$\\theta_1$", color='b', fontsize=12)
#ax.text(0.5*np.cos(theta2), 0.5*np.sin(theta2), 0, r"$\\theta_2$", color='r', fontsize=12)

# for ray 1 (blue)
counter = 0
for xi, yi, zi in zip(x1, y1, z1):
    if counter % 10 == 0:
        ax.plot([xi, xi], [yi, yi], [0, zi], linestyle='--', linewidth=1, color='blue', alpha=alpha)
    counter += 1

# for ray 2 (red)
counter = 0
for xi, yi, zi in zip(x2, y2, z2):
    if counter % 10 == 0:
        ax.plot([xi, xi], [yi, yi], [0, zi], linestyle='--', linewidth=1, color='red', alpha=alpha)
    counter += 1
add_theta_arc(ax, theta1, radius=0.5, color='blue', label=r'$\theta_1$')
add_theta_arc(ax, theta2, radius=0.65, color='red',  label=r'$\theta_2$')

# Axes labels
ax.set_xlabel('x')
ax.set_ylabel('y')

# remove background panels (panes)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.fill = False            # no face
    axis.pane.set_edgecolor((1,1,1,0))# no edges

# remove grid + ticks (optional)
ax.grid(False)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

ax.view_init(elev=45, azim=-50)
plt.tight_layout()

plt.savefig("SW2.png", dpi=300, transparent=True, bbox_inches='tight')

plt.show()