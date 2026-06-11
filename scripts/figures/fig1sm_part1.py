import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
sys.path.append("src/")
import plotting
import networks
plt.style.use("src/plotting_style.mplstyle")

G = networks.voltage_divider(save_data=True) 
training_type = 'allostery'
training_steps = 50

fig, ax = plt.subplots(1,2, figsize=(12,5.5))

h1, l1 = plotting.plot_weights(ax[0], G, training_steps, training_type, weight_type='length_radius_base')

h2, l2 = plotting.plot_weights(ax[1], G, training_steps, training_type, weight_type='length_pressure')

handles = h1 + h2
labels = l1 + l2
by_label = dict(zip(labels, handles))

fig.legend(by_label.values(), by_label.keys(), loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=7)
fig.savefig(f"plots/paper/fig1sm_part1.png", dpi=300)
