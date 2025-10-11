import numpy as np
import matplotlib.pyplot as plt

samples = np.random.normal(0.0, 1.0, 10000)
samples[samples > 0.0] *= 1.1
plt.hist(samples, bins=20, density=True, color='blue', alpha=0.7)
plt.xticks([])
plt.yticks([])
plt.show()