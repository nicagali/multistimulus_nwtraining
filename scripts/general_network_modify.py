"""
Revert the direction of memeristors of an existing network, then train and plot for inoput-output voltage mapping task.

Outputs:
- plots/networks/<graph_id>.png
- plots/<training_type>_<network_name>/mse.png -> normalized mean squared error of the trained network at each training step
- plots/<training_type>_<network_name>/weights.png -> evolution of the weights at each training step (not implemented for 'best_choice' weight type)
"""
from src import networks, training, plotting
from pathlib import Path
plt.style.use("src/plotting_style.mplstyle")
import matplotlib.pyplot as plt
import networkx as nx

graph_id_original = 'G00020001'
graph_id = 'G00020002'

# --------- INITIALIZE NETWORK ---------
G = nx.read_graphml(f'data/networks/{graph_id_original}.graphml')

# Shuffle randomly directions of memristors
G = networks.to_directed_graph(G, shuffle=True)

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph(G)
fig.tight_layout()
fig.savefig(f"plots/networks/{graph_id}.png", dpi=100)

# --------- TRAIN NETWORK---------

training_steps = 50
training_type = 'allostery' 
weight_type = 'radius_base'
# choose from: ['length', 'radius_base', 'rho', 'pressure', 'length_radius_base', 'length_pressure', 'best_choice']
delta_weight = 1
# previously used: [1e-3, 1, 5e-5, 1e-3, [1e-3, 1], [1e-3, 1e-3], [1e-3, 1, 1e-4, 1e-3]] 
learning_rate = 8e-7
# previously used: [1e-6, 8e-7, 1e-4, 20, [1e-6, 8e-7], [1e-6, 20], [1e-6, 8e-7, 1e-4, 20]]

G_train = G.copy(as_view=False)
training.train(G_train, training_type, training_steps, weight_type, delta_weight, learning_rate, save_final_graph=True, write_weights=True)

PLOT_PATH = Path(f"plots/{training_type}{G.graph['name']}/")
PLOT_PATH.mkdir(parents=True, exist_ok=True)
# --------- PLOT ERROR AND WEIGHTS ---------

fig, ax = plt.subplots()
plotting.plot_mse(ax, fig, G.name, training_type, weight_type)
ax.legend()
fig.tight_layout()
fig.savefig(f"{PLOT_PATH}/mse.png", dpi=100)

# Plot weights if not best choice, feature not implemented
if weight_type != 'best_choice':
    fig, ax = plt.subplots(figsize = (5.5,4))
    plotting.plot_weights(ax, G, training_steps, training_type, weight_type, show_xlabel=False)
    fig.tight_layout()
    fig.savefig(f"{PLOT_PATH}/weights_{weight_type}.png", dpi=100)
