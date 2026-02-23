import sys
sys.path.append("src/")
import networks
import training
import plotting
import parameters as par
import matplotlib.pyplot as plt
from pathlib import Path

graph_id = 'G00010001'
# --------- INITIALIZE NETWORK ---------

# -> DEFINE graph from networks module
G = networks.random_graph(number_edges=20, number_nodes=10, save_data=True, graph_id=graph_id) 

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph(G)
fig.tight_layout()
fig.savefig(f"plots/networks/{graph_id}.png", dpi=100)

# --------- TRAIN NETWORK ---------
training_steps = 50   # choose
training_type = 'regression'    # choose
weight_type = 'length'
# choose from: ['length', 'radius_base', 'rho', 'pressure', 'length_radius_base', 'length_pressure']
delta_weight = 1e-3
# previously used: [1e-3, 1e-3, 1e-4, 1e-3, 1e-3]
learning_rate = 5e-7
# previously used: [5e-7, 1e-6, 1e-2, 2e4, 1e5]
constant_source = 11
# previously used: [11, 4, 4, 11, 4]
constant_source_node = '3'

weight_type_index = 0   # choose

G_train = G.copy(as_view=False)

training.train(G_train, training_type, training_steps, weight_type, delta_weight, learning_rate, constant_source=constant_source, save_final_graph=True, write_weights=True, )

PLOT_PATH = Path(f"plots/{training_type}{G.graph['name']}/")
PLOT_PATH.mkdir(parents=True, exist_ok=True)
# --------- PLOT ERROR AND WEIGHTS ---------

fig, ax = plt.subplots()
plotting.plot_mse(ax, fig, G.name, training_type, weight_type)
ax.legend(fontsize = par.legend_size)
fig.tight_layout()
fig.savefig(f"{PLOT_PATH}/mse.png", dpi=100)

fig, ax = plt.subplots(figsize = (5.5,4))
plotting.plot_weights(ax, G, training_steps, training_type, weight_type, show_xlabel=False)
fig.tight_layout()
fig.savefig(f"{PLOT_PATH}/weights_{weight_type}.png", dpi=100)

# --------- TEST REGRESSION AND PLOT RESULT ---------
training.test_regression(G, step=0, weight_type=weight_type)
training.test_regression(G, step=int(training_steps/2), weight_type=weight_type)
training.test_regression(G, step=training_steps, weight_type=weight_type)

fig, ax = plt.subplots(1, 3, figsize=(15,5))
plotting.plot_regression(ax[0], graph_id, weight_type, step=0)
plotting.plot_regression(ax[1], graph_id, weight_type, step=int(training_steps/2))
plotting.plot_regression(ax[2], graph_id, weight_type, step=training_steps)
fig.tight_layout()
fig.savefig(f"{PLOT_PATH}/snapshots_{weight_type}.png", dpi=100)