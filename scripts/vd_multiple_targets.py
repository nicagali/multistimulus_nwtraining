

import sys
sys.path.append("src/")
import networks
import plotting
import training
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use("src/plotting_style.mplstyle")


target_values = [1, 3, 4]
training_steps = 15
training_type = 'allostery'
weight_type = 'pressure'
delta_weight = 1e-3
learning_rate = 1e2

G_target = networks.voltage_divider(save_data=True) 
for target_index in range(len(target_values)):

    G_target.nodes[1]['desired'] = target_values[target_index]

    training.train(G_target, training_type=training_type, training_steps=training_steps, weight_type=weight_type, delta_weight = delta_weight, learning_rate=learning_rate, multi_train=True, write_weights=True)


fig, ax = plt.subplots(2, 1, figsize = (8,7))
plotting.plot_multi_train(ax[0], ax[1], G_target, training_steps, weight_type, target_values)
ax[0].legend()
fig.tight_layout()
fig.savefig(f"plots/allosteryvd/multiple_targets.png", dpi=100)

