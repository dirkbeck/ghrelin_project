import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.rcParams['pdf.fonttype'] = 42

x = np.linspace(0, 1, 500)
D = np.linspace(0, 1, 200)
X, Y = np.meshgrid(x, D)

# performance function
sigma = 0.08
mu_D = 0.8 - 0.6 * Y
P_succ = np.exp(-(X - mu_D)**2 / (2 * sigma**2))

# 1D curves
def perf(x, mu):
    return np.exp(-(x - mu)**2 / (2 * 0.08**2))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#curves
ax1.plot(x, perf(x, 0.8), 'C0', lw=2, label='Simple')
ax1.plot(x, perf(x, 0.5), 'C1', lw=2, label='Mid')
ax1.plot(x, perf(x, 0.2), 'C2', lw=2, label='Complex')
ax1.set_xlabel('Activity')
ax1.set_ylabel('Performance')
ax1.set_title('A) Inverted-U')
ax1.legend()
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1.05)

# heatmap
im = ax2.imshow(P_succ, origin='lower', extent=(0,1,0,1),
                cmap='viridis', norm=TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1))
ax2.set_xlabel('Activity')
ax2.set_ylabel('Complexity')
ax2.set_title('B) Success heatmap')
plt.colorbar(im, ax=ax2, label='P(success)')

plt.tight_layout()
plt.savefig('perf_complexity.pdf', bbox_inches='tight', dpi=300)
plt.show()