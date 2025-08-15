import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42

def p_success(x):
    return 1 / (1 + np.exp(x))

n_perf = 2
c = 0.1

x_orig = np.linspace(-2, 2, 400)
x = x_orig + 2   # now runs from 0 to 4

p = p_success(x_orig)

B0 = np.ones_like(p)
B1 = 1 - binom.pmf(0, n_perf, p)
B2 = binom.pmf(n_perf, n_perf, p)
C  = c * x_orig**2

# Raw utilities and floor at zero
U0 = np.maximum(B0 - C, 0)
U1 = np.maximum(B1 - C, 0)
U2 = np.maximum(B2 - C, 0)

# find optima indices and values
i0 = np.argmax(U0)
i1 = np.argmax(U1)
i2 = np.argmax(U2)
x0_opt, U0_opt = x[i0], U0[i0]
x1_opt, U1_opt = x[i1], U1[i1]
x2_opt, U2_opt = x[i2], U2[i2]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for ax, title, invert in [
    (ax1, 'DMS Striosome Activity', False),
    (ax2, 'DLS Striosome Activity', True)
]:
    # Plot utilities
    ax.plot(x, U0, label='0D utility', color='gray', lw=2)
    ax.plot(x, U1, label='1D utility', color='C0', lw=2)
    ax.plot(x, U2, label='2D utility', color='C1', lw=2)

    # optimal points
    ax.plot(x0_opt, U0_opt, 'o', color='gray', label='0D optimum')
    ax.plot(x1_opt, U1_opt, 'o', color='C0', label='1D optimum')
    ax.plot(x2_opt, U2_opt, 'o', color='C1', label='2D optimum')

    ax.set_title(title)
    ax.set_xlabel('Striosome activity (arb. u.)')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlim(0, 4)
    if invert:
        ax.invert_xaxis()
    ax.legend(loc='upper right')

ax1.set_ylabel('Utility')

plt.tight_layout()

# Save as vector‐editable PDF
plt.savefig('ability_to_perform_tasks.pdf', format='pdf')

plt.show()