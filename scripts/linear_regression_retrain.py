"""
Retrain a network for a differnet linear regression task, after it was trained with length.

Outputs:
- plots/<training_type>_<network_name>/retrained_<weight_type>.png -> test and visual evaluation of trained regression task for choosen retraining weight
"""

import sys
sys.path.append("src/")
import training
import parameters as par
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
import numpy as np

graph_id = 'G00020001'

# -> READ graph 
G = nx.read_graphml(f'data/networks/{graph_id}.graphml')

# -> IMPOSE learned lenghts from time-step 391 (a training protocol must have been run before)
data = np.loadtxt(f"data/regression{G.graph['name']}/weights/length/length391.txt", unpack=True)
weight_vec = data[1]
for index, edge in enumerate(G.edges):
        G.edges[edge]['length'] = weight_vec[index]


# --------- TRAIN NETWORK ---------
training_steps = 50  # choose
training_type = 'regression'    # choose


weight_type = 'pressure'
delta_weight = 1e-2
learning_rate = 1e5
constant_source = 11
constant_source_node = '3'

# choose the function to retrain the regression with the new weight type, for example:                                                      
a = 0.3
b = 0.3

training.train(G, training_type, training_steps, weight_type, delta_weight, learning_rate, constant_source=constant_source, constant_source_node=constant_source_node, a = a, b = b, save_final_graph=True, write_weights=True)
training.test_regression(G, constant_source=constant_source, constant_source_node=constant_source_node, step=training_steps, weight_type=weight_type, retrain=True, a = a, b = b)
        
PLOT_PATH = Path(f"plots/{training_type}{G.graph['name']}/")
PLOT_PATH.mkdir(parents=True, exist_ok=True)

# --------- PLOT RESULT ---------

data_length = np.loadtxt(f"data/regression{graph_id}/relations_regression/relations_regressionlength400.txt", unpack=True)
data_newweight = np.loadtxt(f"data/regression{graph_id}/relations_regression/relations_regression{weight_type}{training_steps}_retrain.txt", unpack=True)
fig, ax = plt.subplots()

x = data_length[0]
x_interval = np.linspace(np.min(x),np.max(x))
y1_desired = 0.2*x_interval + 0.3
ax.plot(x_interval, y1_desired, label = r'$V^{D} = 0.2V_{7}+0.3$', c = 'lightpink', lw=2, zorder=0)
y1 = data_length[1]
ax.scatter(x, y1, marker = 'o', s=30, c = par.cmap.colors[0], lw=0.6, zorder=1, label=r'$L\ (step \ 400)$')

y2_desired = a*x_interval + b
ax.plot(x_interval, y2_desired, label = rf'$V = {a}V_{7}+{b}$', c = 'mediumseagreen', lw=2, zorder=0)
y2 = data_newweight[1]
ax.scatter(x, y2, marker = 'o', s=30, c = par.cmap.colors[3], lw=0.6, zorder=1, label=rf'${weight_type} \ (step \ 400+50)$')

fig.tight_layout()
fig.savefig(f"{PLOT_PATH}/retrained_{weight_type}.png", dpi=200)