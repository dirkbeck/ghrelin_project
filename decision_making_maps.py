import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

plt.rcParams['pdf.fonttype'] = 42
np.random.seed(0)

# params
gamma, w_r, w_p, alpha, beta, k_sig = 2.0, 1.0, 1.0, 5.0, 5.0, 10.0
thr_low, thr_high, D_mid, k_dim = 1.0, 10.0, 5.0, 1.0

# grid
I_r = np.linspace(0, 1, 200)
I_p = np.linspace(0, 1, 200)
Ir, Ip = np.meshgrid(I_r, I_p)


def sigmoid(x, k=1.0):
    return 1.0 / (1.0 + np.exp(-k * x))


def w_dims(D):
    return 1.0 - sigmoid(D - D_mid, k=k_dim)


def compute_P_avoid(D, use_dims=True):
    Q_avoid = w_p * Ip
    Q_approach = w_r * Ir
    if use_dims:
        g3, g4 = sigmoid(D - thr_low, k=k_sig), sigmoid(D - thr_high, k=k_sig)
        fade = w_dims(D)
        Q_avoid += fade * alpha * g3 * (Ir * Ip)
        Q_approach += fade * beta * g4 * (Ir * (1 - Ip))
    num = np.exp(gamma * Q_avoid)
    return num / (num + np.exp(gamma * Q_approach))


# surfaces
P_base = compute_P_avoid(0.0, True)
P_low_dims, P_low_nodims = compute_P_avoid(1.0, True), compute_P_avoid(1.0, False)
P_high_dims, P_high_nodims = compute_P_avoid(10.0, True), compute_P_avoid(10.0, False)

dP_low, dP_low_nd = P_low_dims - P_base, P_low_nodims - P_base
dP_high, dP_high_nd = P_high_dims - P_base, P_high_nodims - P_base

# colormap
cmap_rw = LinearSegmentedColormap.from_list("rwg",
                                            [(0.0, "#660000"), (0.4, "white"), (0.6, "white"), (1.0, "#003300")])

norm_P = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
norm_dP = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)


def plot_panels(axs, P_base, P_D, dP, label, use_dims, row):
    mats = [(P_base, f'P(avoid) base\n(dims={"on" if use_dims else "off"})', norm_P),
            (P_D, f'P(avoid) {label}\n(dims={"on" if use_dims else "off"})', norm_P),
            (dP, f'ΔP(avoid) {label}\n(dims={"on" if use_dims else "off"})', norm_dP)]

    for col, (mat, title, norm) in enumerate(mats):
        axs[row, col].imshow(mat, origin='lower', extent=(0, 1, 0, 1), cmap=cmap_rw, norm=norm)
        axs[row, col].set_title(title)
        axs[row, col].set_xlabel('Reward')
        axs[row, col].set_ylabel('Cost')
        axs[row, col].grid(alpha=0.3)


# low ghrelin
fig1, axs1 = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
plot_panels(axs1, P_base, P_low_dims, dP_low, 'low-ghrelin', True, 0)
plot_panels(axs1, P_base, P_low_nodims, dP_low_nd, 'low-ghrelin', False, 1)
fig1.colorbar(plt.cm.ScalarMappable(norm=norm_P, cmap=cmap_rw),
              ax=axs1[:, 0:2], label='P(avoid)')
fig1.colorbar(plt.cm.ScalarMappable(norm=norm_dP, cmap=cmap_rw),
              ax=axs1[:, 2], label='ΔP(avoid)')
plt.savefig('low_ghrelin_maps.pdf', dpi=300)
plt.show()

# high ghrelin
fig2, axs2 = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
plot_panels(axs2, P_base, P_high_dims, dP_high, 'high-ghrelin', True, 0)
plot_panels(axs2, P_base, P_high_nodims, dP_high_nd, 'high-ghrelin', False, 1)
fig2.colorbar(plt.cm.ScalarMappable(norm=norm_P, cmap=cmap_rw),
              ax=axs2[:, 0:2], label='P(avoid)')
fig2.colorbar(plt.cm.ScalarMappable(norm=norm_dP, cmap=cmap_rw),
              ax=axs2[:, 2], label='ΔP(avoid)')
plt.suptitle('High‐Ghrelin Decision Maps\n(red=higher avoidance, green=higher approach)')
plt.savefig('high_ghrelin_maps.pdf', dpi=300)
plt.show()