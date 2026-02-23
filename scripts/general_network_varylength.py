import sys
sys.path.append("src/")
import networks
import training
import plotting
import parameters as par
import matplotlib.pyplot as plt
from pathlib import Path

graph_id = 'G00020001'

# --------- INITIALIZE NETWORK ---------

# -> DEFINE graph from networks module
G = networks.random_graph(number_nodes=8, number_edges=12, number_sources=2, number_targets=1, save_data=True, graph_id=graph_id) 

# -> PLOT graph in /plots 
fig, ax = plt.subplots()
pos = plotting.plot_graph(G)
fig.tight_layout()
fig.savefig(f"plots/networks/{graph_id}.png", dpi=100)
# --------- TRAIN NETWORK WITH DIFFERENT WEIGHTS ---------

training_steps = 300
training_type = 'allostery'
weight_type = 'rho'
delta_weight = 1e-5
learning_rate = 2e-3

# G_train = G.copy(as_view=False)
# training.train(G_train, training_type=training_type, training_steps=training_steps, weight_type=weight_type, delta_weight = delta_weight, learning_rate=learning_rate, save_final_graph=True, write_weights=True, varying_len=False)

# G_train = G.copy(as_view=False)
# training.train(G_train, training_type=training_type, training_steps=training_steps, weight_type=weight_type, delta_weight = delta_weight, learning_rate=learning_rate, save_final_graph=True, write_weights=True, varying_len=True)

PLOT_PATH = Path(f"plots/{training_type}{G.graph['name']}/")
PLOT_PATH.mkdir(parents=True, exist_ok=True)
# --------- PLOT ERROR AND WEIGHTS ---------

fig, ax = plt.subplots()
plotting.plot_mse(ax, fig, graph_id, training_type, f'{weight_type}')
plotting.plot_mse(ax, fig, graph_id, training_type, f'{weight_type}', var_l=True)
fig.tight_layout()
fig.savefig(f"{PLOT_PATH}/mse.png", dpi=100)

import numpy as np
import ahkab
def plot_length_function(circuit):

    potential = np.linspace(-5, 5)
    mysistor = ahkab.Circuit.get_elem_by_name(circuit, part_id = 'M1')


    function = ahkab.transient.length_sigmoid(mysistor, potential) 
    function = function/(function[0])
    # print(function)

    fig, ax = plt.subplots()
    ax.plot(potential, function, c = 'green', lw=4, label = rf'$\sigma(\Delta V)$')
    ax.set_ylabel(r'$L[\mu m]/L_0$', fontsize = par.axis_fontsize)
    ax.set_xlabel(r'$\Delta V$[$\Omega$]', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)
    ax.legend(fontsize = par.legend_size)
    fig.tight_layout()
    fig.savefig(f"{PLOT_PATH}/length_function.png", dpi=100)

circuit = ahkab.circuit.Circuit('Single Memristor')
circuit.add_mysistor('M1', 'n1', circuit.gnd, value=1/4, rho_b=0.1, length_channel=10e-6, radius_base=200e-9, pressure=0, delta_rho=0)
plot_length_function(circuit)

