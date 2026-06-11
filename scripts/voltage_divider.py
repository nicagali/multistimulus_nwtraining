"""
Train and plot a voltage-divider network.

Outputs:
- plots/networks/vd.png
- plots/<training_type>_<network_name>/mse.png -> normalized mean squared error of the trained network at each training step
- plots/<training_type>_<network_name>/evolution_finalnw.png -> evolution to the staedy state of the trained network, showing the potential at each node at the final training step
- plots/<training_type>_<network_name>/weights.png -> evolution of the weights at each training step (not implemented for 'best_choice' weight type)
"""
from src import networks, training, plotting
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use("src/plotting_style.mplstyle")

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
training_type = 'length_radius_base'
weight_type = 'pressure' # choose between: ['length', 'resistance', 'radius_base', 'rho', 'pressure', 'length_radius_base', 'length_pressure', 'best_choice']
delta_weight = 1e-3 # previously used: [1e-3, 1e-3, 1, 1e-4, 1e-3, [1e-3, 1e-2], [1e-3, 1e-3], [1e-3, 1, 1e-4, 1e-3]]
learning_rate =  1e2 # previously used: [3e-6, 1e-3, 3.5e-6, 5e-4, 1e2, [1e-6, 2*1e-6], [5e-6, 1], [2e-6, 3.5e-6, 5e-4, 1e2]]

G_ml = G.copy(as_view=False)  
training.train(G_ml, training_type, training_steps, weight_type, delta_weight, learning_rate, write_weights=True)

PLOT_PATH = Path(f"plots/{training_type}{G.graph['name']}/")
PLOT_PATH.mkdir(parents=True, exist_ok=True)

# --------- PLOT EVOLUTION OF TRAINED NW ---------
 
fig, ax = plt.subplots(figsize=(5,5))
plotting.plot_potential_each_node(ax, G_ml)
fig.tight_layout()
fig.savefig(f"{PLOT_PATH}/evolution_finalnw.png", dpi=300)

# --------- PLOT ERROR AND WEIGHTS ---------

fig, ax = plt.subplots()
plotting.plot_mse(ax, fig, G.name, training_type, weight_type)
ax.legend()
fig.tight_layout()
fig.savefig(f"{PLOT_PATH}/mse.png", dpi=300)

# Plot weights if not best choice, feature not implemented
if weight_type != 'best_choice':
    fig, ax = plt.subplots()
    plotting.plot_weights(ax, G, training_steps, training_type, weight_type)
    fig.tight_layout()
    fig.savefig(f"{PLOT_PATH}/weights.png", dpi=200)

