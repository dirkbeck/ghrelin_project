import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['pdf.fonttype'] = 42

# params
satiety_decay, inverse_temp, reward_weight, cost_weight = 3.0, 5.0, 2.0, 1.0
fade_midpoint, fade_steepness = 5.0, 1.0
gate_center_conflict = 2.0

def sigmoid(x, k=1.0):
    return 1.0 / (1.0 + np.exp(-k * x))

def rule_fade(dose):
    return 1.0 - sigmoid(dose - fade_midpoint, k=fade_steepness)

# grids
contexts = np.linspace(0, 1, 200)
doses = np.linspace(0, 10, 200)
ctx_grid, dose_grid = np.meshgrid(contexts, doses, indexing='xy')

# context gates
reward_impact = np.where(ctx_grid <= 0.5, 2 * ctx_grid, 1.0)
cost_impact = np.where(ctx_grid <= 0.5, 1.0, 2 - 2 * ctx_grid)

# q values
satiety = np.exp(-dose_grid / satiety_decay)
deficit = 1.0 - satiety
Q_base = reward_weight * deficit - cost_weight * cost_impact
Q_complex = reward_weight * deficit * reward_impact - cost_weight * cost_impact

# mix rules
fade_matrix = np.tile(rule_fade(doses), (len(contexts), 1))
Q_mixed = fade_matrix * Q_complex + (1 - fade_matrix) * Q_base

# probabilities
P_complex_rule = 1.0 / (1.0 + np.exp(-inverse_temp * Q_mixed))
P_base_rule = 1.0 / (1.0 + np.exp(-inverse_temp * Q_base))

# colormap
cmap = LinearSegmentedColormap.from_list("rwg",
    [(0.0, "#660000"), (0.4, "white"), (0.6, "white"), (1.0, "#003300")])

# phase plot
rgba = cmap(P_complex_rule)
rgba[..., :3] = fade_matrix[...,None] * rgba[..., :3] + (1 - fade_matrix[...,None]) * 1.0

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.imshow(rgba, origin='lower', aspect='auto',
           extent=[doses.min(), doses.max(), contexts.min(), contexts.max()])
ax1.contour(doses, contexts, P_complex_rule, levels=[0.5], colors='black', linewidths=2)
ax1.set_yticks([0.0, 0.5, 1.0])
ax1.set_yticklabels(['Cost Only', 'Conflict', 'Reward Only'])
ax1.set_ylabel('Task Context')
ax1.set_xlabel('Ghrelin Dose')
ax1.set_title('Phase Diagram with Rule Fade')
plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1), cmap=cmap),
             ax=ax1, label='P(Approach)')
plt.savefig('phase_plot.pdf')
plt.show()

# line plots
mean_complex = P_complex_rule.mean(axis=0)
mean_base = P_base_rule.mean(axis=0)
mean_diff = np.abs(P_complex_rule - P_base_rule).mean(axis=0)
conflict_idx = np.abs(contexts - 0.5).argmin()
P_complex_conflict = P_complex_rule[conflict_idx, :]
P_base_conflict = P_base_rule[conflict_idx, :]

fig2, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

axs[0,0].plot(doses, mean_complex, label='Complex', color='C0')
axs[0,0].plot(doses, mean_base, label='Base', color='C1')
axs[0,0].set_title('Average P(Approach)')
axs[0,0].set_xlabel('Dose')
axs[0,0].legend()

axs[0,1].plot(doses, mean_diff, color='C2')
axs[0,1].axvline(gate_center_conflict, ls='--', color='gray')
axs[0,1].set_title('Mean |Complex−Base|')
axs[0,1].set_xlabel('Dose')

axs[1,0].plot(doses, P_complex_conflict, label='Complex', color='C3')
axs[1,0].plot(doses, P_base_conflict, label='Base', color='C4', ls='--')
axs[1,0].set_title('P(Approach) at Conflict')
axs[1,0].set_xlabel('Dose')
axs[1,0].legend()

axs[1,1].plot(doses, sigmoid(doses - gate_center_conflict, 10), label='Gate', color='C5')
axs[1,1].plot(doses, rule_fade(doses), label='Fade', color='C6')
axs[1,1].set_title('Gate & Fade')
axs[1,1].set_xlabel('Dose')
axs[1,1].legend()

plt.savefig('line_analysis.pdf')
plt.show()