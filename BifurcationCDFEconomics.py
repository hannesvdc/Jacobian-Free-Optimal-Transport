import numpy as np
import matplotlib.pyplot as plt
import pycont

import EconomicAgentTimestepper as agents
import CDF1DOptimizers as cdfopt

# Model parameters
N = 100_000
eplus = 0.075
eminus = -0.072
vplus = 20
vminus = 20
vpc = vplus
vmc = vminus
gamma = 1

n_grid_points = 101
grid = np.linspace(-1.0, 1.0, n_grid_points)

# Time stepping parameters
Tpsi = 1.0
dt = 0.25
n_steps = int(Tpsi / dt)
def agent_timestepper(X: np.ndarray, g: float) -> np.ndarray:
    x = agents.evolveAgentsNumpy(X, n_steps, dt, gamma, vplus, vminus, vpc, vmc, eplus, eminus, g, len(X), verbose=False)
    return x
def cdf_timestepper(cdf : np.ndarray, g : float) -> np.ndarray:
    particles = cdfopt.particles_from_cdf(grid, cdf, N)
    new_particles = agent_timestepper(particles, g)
    cdf_new = cdfopt.empirical_cdf_on_grid(new_particles, grid)
    return cdf_new
def psi(cdf : np.ndarray, g : float) -> np.ndarray:
    return cdf - cdf_timestepper(cdf, g)

# Solve for the initial CDF on the path
print('\nComputing first point on the branch...')
g0 = 38.0
rdiff = 1e-1
maxiter = 5
sigma0 = 0.1
X0 = np.random.normal(0.0, sigma0, N)
cdf0 = np.array([np.mean(X0 <= grid[i]) for i in range(len(grid))])
cdf_inf, losses = cdfopt.cdf_newton_krylov(cdf0, grid, lambda cdf: agent_timestepper(cdf, g0), maxiter, rdiff, N)

# Do pseudo-arclength continuation
ds_min = 1e-6
ds_max = 0.1
ds = ds_max
n_steps = 100
tolerance = 1e-2
solver_parameters = {'rdiff': rdiff, 'nk_maxiter': maxiter, "tolerance": tolerance}
continuation_result = pycont.pseudoArclengthContinuation(psi, cdf0, g0, ds_min, ds_max, ds, n_steps, solver_parameters=solver_parameters)

for branch in continuation_result.branches:
    plt.plot(branch['p'], branch['u'], color='tab:blue')
plt.xlabel(r"$g$")
plt.ylabel(r"$\bar{X}$")
plt.show()