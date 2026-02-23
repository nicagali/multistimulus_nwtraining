import sys
sys.path.append("src/")
import networks
import training
import plotting
import parameters as par
from pathlib import Path
import matplotlib.pyplot as plt

# --------- INITIALIZE NETWORK ---------

# -> DEFINE graph from networks module
G = networks.voltage_divider(save_data=True) 

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph(G)
fig.tight_layout()
fig.savefig(f"plots/networks/vd.png", dpi=100)

# --------- TRAIN NETWORK ---------

training_steps = 50
training_type = 'allostery'
weight_type = 'length' # choose between: ['length', 'resistance', 'radius_base', 'rho', 'pressure', 'length_radius_base', 'length_pressure', 'best_choice']
delta_weight = 1e-3 # previously used: [1e-3, 1e-3, 1, 1e-4, 1e-3, [1e-3, 1e-2], [1e-3, 1e-3], [1e-3, 1, 1e-4, 1e-3]]
learning_rate = 3e-6 # previously used: [3e-6, 1e-3, 3.5e-6, 5e-4, 1e2, [1e-6, 2*1e-6], [5e-6, 1], [2e-6, 3.5e-6, 5e-4, 1e2]]

G_ml = G.copy(as_view=False)  
training.train(G_ml, training_type, training_steps, weight_type, delta_weight, learning_rate, write_weights=True)

PLOT_PATH = Path(f"plots/{training_type}{G.graph['name']}/")
PLOT_PATH.mkdir(parents=True, exist_ok=True)
# --------- PLOT EVOLUTION OF TRAINED NW ---------
 
fig, ax = plt.subplots(figsize = par.figsize_1)
plotting.plot_potential_each_node(ax, G_ml)
fig.tight_layout()
fig.savefig(f"{PLOT_PATH}/evolution_finalnw.png", dpi=100)

# --------- PLOT ERROR AND WEIGHTS ---------

fig, ax = plt.subplots(figsize = par.figsize_1)
plotting.plot_mse(ax, fig, G.name, training_type, weight_type)
fig.tight_layout()
fig.savefig(f"{PLOT_PATH}/mse.png", dpi=100)

fig, ax = plt.subplots(figsize = par.figsize_1)
plotting.plot_weights(ax, G, training_steps, training_type, weight_type)
fig.tight_layout()
fig.savefig(f"{PLOT_PATH}/weights.png", dpi=100)

# --------- PLOT RESISTANCES OF MEMRISTORS TRAINED GRAPH ---------

fig, ax = plt.subplots(figsize = par.figsize_1)
plotting.plot_memristor_resistances(ax, G)
fig.tight_layout()
fig.savefig(f"{PLOT_PATH}/memristors_resisatnces.png", dpi=100)
