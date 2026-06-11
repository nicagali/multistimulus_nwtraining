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

fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1])

ax1 = fig.add_subplot(gs[0, 1:3]) 
plotting.plot_mse(ax1, fig, G.graph['name'], training_type, f'length', show_xlabel=False)
plotting.plot_mse(ax1, fig, G.graph['name'], training_type, f'radius_base', show_xlabel=False)
plotting.plot_mse(ax1, fig, G.graph['name'], training_type, f'rho', show_xlabel=False)
plotting.plot_mse(ax1, fig, G.graph['name'], training_type, f'pressure', show_xlabel=False)
plotting.plot_mse(ax1, fig, G.graph['name'], training_type, f'length_pressure', show_xlabel=False)
plotting.plot_mse(ax1, fig, G.graph['name'], training_type, f'length_radius_base', show_xlabel=False)
plotting.plot_mse(ax1, fig, G.graph['name'], training_type, f'best_choice', show_xlabel=False)
ax1.legend()

ax2 = fig.add_subplot(gs[0, 3:4])
plotting.plot_weights(ax2, G, training_steps=training_steps, training_type=training_type, weight_type=f'length', show_xlabel=False)

ax3 = fig.add_subplot(gs[1, 0])
plotting.plot_weights(ax3, G, training_steps=training_steps, training_type=training_type, weight_type=f'radius_base', show_xlabel=False)

ax4 = fig.add_subplot(gs[1, 1])
plotting.plot_weights(ax4, G, training_steps=training_steps, training_type=training_type, weight_type=f'rho', show_xlabel=False)

ax5 = fig.add_subplot(gs[1, 2])
plotting.plot_weights(ax5, G, training_steps=training_steps, training_type=training_type, weight_type=f'pressure')


handles, legend = ax1.get_legend_handles_labels()
ax1.get_legend().remove()
fig_leg = fig.legend(handles, legend, bbox_to_anchor = (1,1.05), ncol = 7, fontsize=22)
fig.tight_layout()
fig.savefig(f"plots/paper/fig1.png", dpi=300)
