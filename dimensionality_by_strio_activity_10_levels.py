import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

def p_success(x):
    return 1 / (1 + np.exp(x))

# params
max_dim = 10
x0 = np.linspace(-2, 2, 400)
p = p_success(x0)

# compute pmfs
probs = [binom.pmf(k, max_dim, p) for k in range(max_dim+1)]
probs = [np.maximum(arr, 0) for arr in probs]
total = np.sum(probs, axis=0)
probs = [arr/total for arr in probs]

# colors
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, max_dim+1))

# plot
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))

# left: DMS
x_plot = x0 + 2
ax0.stackplot(x_plot, *probs, colors=colors, alpha=0.85)
ax0.set_xlim(0, 4)
ax0.set_ylim(0, 1)
ax0.set_xlabel('DMS striosome activity')
ax0.set_ylabel('Probability')

#DLS (mirrored)
x_plot_m = -x0 + 2
ax1.stackplot(x_plot_m, *probs, colors=colors, alpha=0.85)
ax1.set_xlim(0, 4)
ax1.set_ylim(0, 1)
ax1.set_xlabel('DLS striosome activity')

# colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_dim))
cb = plt.colorbar(sm, ax=[ax0, ax1])
cb.set_label('decision-space dimensionality')

plt.tight_layout()
plt.savefig('decision_space.pdf')
plt.show()