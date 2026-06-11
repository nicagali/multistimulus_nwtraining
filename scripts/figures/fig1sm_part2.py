import sys
sys.path.append("src/")
import networks
import training
import plotting
import parameters as par
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use("src/plotting_style.mplstyle")
import numpy as np
import networkx as nx

graph_id = 'G00010001'
training_type = 'allostery'
DATA_PATH = f'data/{training_type}{graph_id}/'
training_steps = 400
G = nx.read_graphml(f'data/networks/{graph_id}.graphml')


# fig, ax = plt.subplots(2, 2, figsize = (14,12))
# plotting.plot_weights(ax[0,0], G, training_steps=training_steps, training_type=training_type, weight_type=f'length', show_xlabel=False)
# plotting.plot_weights(ax[0,1], G, training_steps=training_steps, training_type=training_type, weight_type=f'radius_base', show_xlabel=False)
# plotting.plot_weights(ax[1,0], G, training_steps=training_steps, training_type=training_type, weight_type=f'rho', show_xlabel=False)
# plotting.plot_weights(ax[1,1], G, training_steps=training_steps, training_type=training_type, weight_type=f'pressure', show_xlabel=False)
# fig.tight_layout()
# fig.savefig(f"plots/paper/fig1sm_part2all.png", dpi=300)

graph_id = 'G00020001'
training_type = 'regression'
DATA_PATH = f'data/{training_type}{graph_id}/'
training_steps = 400
G = nx.read_graphml(f'data/networks/{graph_id}.graphml')


fig, ax = plt.subplots(2, 2, figsize = (14,12))
plotting.plot_weights(ax[0,0], G, training_steps=training_steps, training_type=training_type, weight_type=f'length', show_xlabel=False)
plotting.plot_weights(ax[0,1], G, training_steps=training_steps, training_type=training_type, weight_type=f'radius_base', show_xlabel=False)
plotting.plot_weights(ax[1,0], G, training_steps=training_steps, training_type=training_type, weight_type=f'rho', show_xlabel=False)
plotting.plot_weights(ax[1,1], G, training_steps=training_steps, training_type=training_type, weight_type=f'pressure', show_xlabel=False)
fig.tight_layout()
fig.savefig(f"plots/paper/fig1sm_part2regr.png", dpi=300)
