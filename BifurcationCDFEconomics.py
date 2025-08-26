import numpy as np
import matplotlib.pyplot as plt
import pycont

import EconomicAgentTimestepper as agents
import CDF1DOptimizers as cdfopt

import pickle

def computeMeanAgentPositions(branch, grid):
    mean_agent_positions = np.zeros(branch.shape[0])
    for index in range(branch.shape[0]):
        cdf = branch[index, :]
        mean_agent_position = grid[-1]*cdf[-1] - grid[0]*cdf[0] - np.trapezoid(cdf, grid)
        mean_agent_positions[index] = mean_agent_position

    return mean_agent_positions

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
T = 100.0
Tpsi = 1.0
dt = 0.25
n_steps = int(Tpsi / dt)
def agent_timestepper(X: np.ndarray, g: float) -> np.ndarray:
    x = agents.evolveAgentsNumpy(X, n_steps, dt, gamma, vplus, vminus, vpc, vmc, eplus, eminus, g, len(X), verbose=False)
    return x
def cdf_timestepper(cdf : np.ndarray, g : float) -> np.ndarray:
    cdf = cdf / cdf[-1]
    particles = cdfopt.particles_from_cdf(grid, cdf, N)
    new_particles = agent_timestepper(particles, g)
    cdf_new = cdfopt.empirical_cdf_on_grid(new_particles, grid)
    return cdf_new / cdf_new[-1]
def psi(cdf : np.ndarray, g : float) -> np.ndarray:
    return cdf - cdf_timestepper(cdf, g)

# Solve for the initial CDF on the path
print('\nComputing first point on the branch...')
g0 = 38.0
rdiff = 1e-1
maxiter = 10
sigma0 = 0.1
X0 = np.random.normal(0.0, sigma0, N)
k = int(T / dt)
X_inf = agents.evolveAgentsNumpy(X0, k, dt, gamma, vplus, vminus, vpc, vmc, eplus, eminus, g0, N)
cdf0 = np.array([np.mean(X_inf <= grid[i]) for i in range(len(grid))])
cdf0 /= cdf0[-1]
print('Done')

# Do pseudo-arclength continuation
ds_min = 1e-6
ds_max = 0.1
ds = ds_min
n_steps = 1
tolerance = 2e-2
solver_parameters = {'rdiff': rdiff, 'nk_maxiter': maxiter, "tolerance": tolerance}
continuation_result = pycont.pseudoArclengthContinuation(psi, cdf0, g0, ds_min, ds_max, ds, n_steps, solver_parameters=solver_parameters)

# Store the complete continuation result
with open("./Results/economic_cdf_bifurcation_result.pkl", "wb") as f:
    pickle.dump(continuation_result, f)

# Plot the branches
for branch in continuation_result.branches:
    mean_agent_positions = computeMeanAgentPositions(branch['u'], grid)
    plt.plot(branch['p'], mean_agent_positions, color='tab:blue')
plt.xlabel(r"$g$")
plt.ylabel(r"$\bar{X}$")
plt.show()