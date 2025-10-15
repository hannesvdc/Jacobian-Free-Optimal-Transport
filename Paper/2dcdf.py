import matplotlib.pyplot as plt
import numpy as np

dist = lambda X, Y : (X**2 * Y**2)**(0.3) * np.exp(-(X**2 + Y**2))
x = np.linspace(-3.0, 3.0, 1001)
X, Y = np.meshgrid(x, x)
dist_values = dist(X, Y)

cdf_values = np.cumsum(np.cumsum(dist_values, axis=0), axis=1)
cdf_values /= cdf_values[-1,-1]

# Plot the 2D surface
# Create 3D figure and axes
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
# colors (light face, darker grid)
face_blue = '#88A8FF'   # light
grid_blue = '#1F4FA3'   # dark
surf = ax.plot_surface(X, Y, cdf_values, edgecolor='none', color=face_blue, alpha=0.7)
ax.plot_wireframe(X, Y, cdf_values,
                  rstride=100, cstride=100,
                  color=grid_blue, linewidth=0.6, alpha=0.9)

# remove background panels (panes)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.fill = False            # no face
    axis.pane.set_edgecolor((1,1,1,0))# no edges

# remove grid + ticks (optional)
ax.grid(False)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

# remove axis labels
ax.set_xlabel(r'$X$'); ax.set_ylabel(r'$Y$');

plt.savefig("1DCDF2.png", dpi=300, transparent=True, bbox_inches='tight')

plt.show()

plt.show()

