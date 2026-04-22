import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['pdf.fonttype'] = 42

inverse_temp = 10.0

def sigmoid(x, k=1.0):
    return 1.0 / (1.0 + np.exp(-k * x))

def striosome_activity(dose, peak_dose=3.0, width=1.2):
    return np.exp(-((dose - peak_dose) ** 2) / (2 * width ** 2))

n = 200
contexts = np.linspace(0, 1, n)
doses    = np.linspace(0, 10, n)

ctx_col  = contexts.reshape(n, 1)
dose_row = doses.reshape(1, n)

Q_base_col = 6.0 * (ctx_col - 0.5)

strio_row = striosome_activity(dose_row)

conflict_sigma = 0.12
conflict_weight_col = np.exp(
    -((ctx_col - 0.5) ** 2) / (2 * conflict_sigma ** 2)
)

ghrelin_strength = 5.0
ghrelin_effect = -ghrelin_strength * conflict_weight_col * strio_row

Q_total    = Q_base_col + ghrelin_effect
P_approach = sigmoid(Q_total, k=inverse_temp)

cmap = LinearSegmentedColormap.from_list("rwg",
    [(0.0, "#660000"), (0.4, "white"), (0.6, "white"), (1.0, "#003300")])

fig1, ax1 = plt.subplots(figsize=(8, 5))
im = ax1.imshow(P_approach, origin='lower', aspect='auto',
                extent=[doses.min(), doses.max(),
                        contexts.min(), contexts.max()],
                cmap=cmap, vmin=0, vmax=1)
ax1.contour(doses, contexts, P_approach,
            levels=[0.5], colors='black', linewidths=2)
ax1.set_yticks([0.0, 0.5, 1.0])
ax1.set_yticklabels(['Cost Only', 'Conflict', 'Reward Only'])
ax1.set_ylabel('Task Context')
ax1.set_xlabel('Ghrelin Dose (arb. u.)')
ax1.set_title('Phase Diagram: IBU modifies behavior based on context and dose')
plt.colorbar(im, ax=ax1, label='P(Approach)')
plt.tight_layout()
plt.savefig('phase_plot.pdf')
plt.show()
