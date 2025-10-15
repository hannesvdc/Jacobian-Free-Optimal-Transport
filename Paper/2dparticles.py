import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState()
particles_1 = np.vstack((rng.uniform(-1.0, 1.0, 70), rng.uniform(-1.0, 1.0, 70)))
particles_2 = np.vstack((rng.uniform(-1.0, 1.0, 70), rng.uniform(-1.0, 1.0, 70)))

fig, ax = plt.subplots()
ax.scatter(
    particles_1[0], particles_1[1],
    s=36,
    facecolors='orange',
    edgecolors='blue',
    linewidths=1.5,
    marker='o'
)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
plt.savefig("particles1.png", dpi=300, transparent=True, bbox_inches='tight')

fig, ax = plt.subplots()
ax.scatter(
    particles_2[0], particles_2[1],
    s=36,
    facecolors='orange',
    edgecolors='blue',
    linewidths=1.5,
    marker='o'
)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
plt.savefig("particles2.png", dpi=300, transparent=True, bbox_inches='tight')
plt.show()