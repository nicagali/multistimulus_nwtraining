"""
Train and plot a general shape network for inoput-output voltage (here called allostery) mapping task.

Outputs:
- plots/networks/<graph_id>.png
- plots/<training_type>_<network_name>/mse.png -> normalized mean squared error of the trained network at each training step
- plots/<training_type>_<network_name>/weights.png -> evolution of the weights at each training step (not implemented for 'best_choice', 'length_radius_base' and 'length_pressure' weight types)
"""
import sys
sys.path.append("src/")
import networks
import plotting
import training
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use("src/plotting_style.mplstyle")

graph_id = 'G00010001'

# --------- INITIALIZE NETWORK ---------

# -> DEFINE graph from networks module, random graph with specified number of nodes, edges, sources and targets. 
import networkx as nx
G = networks.random_graph(number_nodes=9, number_edges=12, number_sources=3, number_targets=1, graph_id=graph_id, save_data=True) 
# Otherwise, read an already generated one
G = nx.read_graphml(f'data/networks/{graph_id}.graphml')

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph(G)
fig.tight_layout()
fig.savefig(f"plots/networks/{graph_id}.png", dpi=100)

# --------- TRAIN NETWORK---------

training_steps = 400
training_type = 'allostery' 
weight_type = 'length'
# choose from: ['length', 'radius_base', 'rho', 'pressure', 'length_radius_base', 'length_pressure', 'best_choice']
delta_weight = 1e-3
# previously used: [1e-3, 1, 5e-5, 1e-3, [1e-3, 1], [1e-3, 1e-3], [1e-3, 1, 1e-4, 1e-3]] 
learning_rate = 1e-6
# previously used: [1e-6, 8e-7, 1e-4, 20, [1e-6, 8e-7], [1e-6, 20], [1e-6, 8e-7, 1e-4, 20]]
relative_noise = 0.1
# previously used = no noise [0, 0, 0, 0], 10 percent [0.1, 0.1, 0.01, 0.01], 5 percent [0.05, 0.05, 0.005, 0.005], 1 percent [0.01, 0.01, 0.001, 0.001]
n_cores = 10

G_train = G.copy(as_view=False)
training.train(G_train, training_type, training_steps, weight_type, delta_weight, learning_rate, relative_noise, n_cores=n_cores, save_final_graph=True, write_weights=True)

PLOT_PATH = Path(f"plots/{training_type}{G.graph['name']}/")
PLOT_PATH.mkdir(parents=True, exist_ok=True)

# --------- PLOT ERROR AND WEIGHTS ---------

fig, ax = plt.subplots()
plotting.plot_mse(ax, fig, G.name, training_type, weight_type, relative_noise=relative_noise)
ax.legend()
fig.tight_layout()
fig.savefig(f"{PLOT_PATH}/mse.png", dpi=200)

# Plot weights if not best choice, l_rb and l_p feature not implemented
if weight_type != 'best_choice' and weight_type != 'length_radius_base' and weight_type != 'length_pressure':
    fig, ax = plt.subplots()
    plotting.plot_weights(ax, G, training_steps, training_type, weight_type)
    fig.tight_layout()
    fig.savefig(f"{PLOT_PATH}/weights_{weight_type}.png", dpi=200)
