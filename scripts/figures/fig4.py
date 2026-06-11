import sys
sys.path.append("src/")
import networks
import training
import plotting
import parameters as par
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use("src/plotting_style.mplstyle")

graph_id = 'G00010001'
training_type = 'allostery'

fig, ax = plt.subplots(figsize=(8,7))
plotting.plot_mse(ax, fig, graph_id, training_type, f'length')
plotting.plot_mse(ax, fig, graph_id, training_type, f'radius_base')
# plotting.plot_mse(ax, fig, graph_id, training_type, f'length_radius_base')
plotting.plot_mse(ax, fig, graph_id, training_type, f'rho')
plotting.plot_mse(ax, fig, graph_id, training_type, f'pressure')
plotting.plot_mse(ax, fig, graph_id, training_type, f'best_choice')
# plotting.plot_mse(ax, fig, graph_id, training_type, f'length_pressure')
ax.legend()
fig.tight_layout()
fig.savefig(f"plots/paper/fig4.pdf", transparent=True)

fig, ax = plt.subplots(1, 3, figsize = (20,7))

plotting.plot_mse(ax[0], fig, graph_id, training_type, f'length')
plotting.plot_mse(ax[0], fig, graph_id, training_type, f'radius_base')
# plotting.plot_mse(ax[0], fig, graph_id, training_type, f'length_radius_base')
plotting.plot_mse(ax[0], fig, graph_id, training_type, f'rho')
plotting.plot_mse(ax[0], fig, graph_id, training_type, f'pressure')
plotting.plot_mse(ax[0], fig, graph_id, training_type, f'best_choice')
# plotting.plot_mse(ax[0], fig, graph_id, training_type, f'length_pressure')
# ax.legend()
fig.tight_layout()
fig.savefig(f"plots/paper/fig4_horizontal.pdf", transparent=True)