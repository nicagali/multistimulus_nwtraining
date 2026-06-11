import sys
sys.path.append("src/")
import networks
import training
import plotting
import parameters as par
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use("src/plotting_style.mplstyle")

G = networks.voltage_divider(save_data=True) 
training_type = 'allostery'
training_steps = 50

# --------- TRAIN NETWORK WITH DIFFERENT TARGETS ---------

target_values = [1, 3, 4]
training_steps = 15

fig, ax = plt.subplots(2, 1, figsize = (7,7))

G_target = networks.voltage_divider(save_data=True) 
for target_index in range(len(target_values)):

    G_target.nodes[1]['desired'] = target_values[target_index]

    training.train(G_target, training_type=training_type, training_steps=training_steps, weight_type='pressure', delta_weight = 1e-3, learning_rate=100, write_weights=True)

    plotting.plot_weights(ax[1], G_target, training_steps=training_steps, training_type=training_type, weight_type='pressure', show_xlabel=True, starting_step=(target_index*training_steps))

plotting.plot_final_potential_vd(ax[0], target_values)
ax[0].legend()
fig.tight_layout()
fig.savefig(f"plots/paper/fig3.pdf")