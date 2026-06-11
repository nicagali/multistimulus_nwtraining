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

graph_id = 'G00020001'
training_type = 'regression'

fig, ax = plt.subplots(1, 3, figsize = (20,7))

# mse
plotting.plot_mse(ax[0], fig, graph_id, training_type, f'length')
plotting.plot_mse(ax[0], fig, graph_id, training_type, f'radius_base')
# plotting.plot_mse(ax, fig, graph_id, training_type, f'length_radius_base')
plotting.plot_mse(ax[0], fig, graph_id, training_type, f'rho')
plotting.plot_mse(ax[0], fig, graph_id, training_type, f'pressure')
plotting.plot_mse(ax[0], fig, graph_id, training_type, f'best_choice')
ax[0].legend()

# regression test
h1, l1 = plotting.plot_regression(ax[1], graph_id, 'length', step=0, marker='^')
h2, l2 = plotting.plot_regression(ax[1], graph_id, 'length', step=100, marker='P')
h3, l3 = plotting.plot_regression(ax[1], graph_id, 'length', step=400, marker='o')

# reconfiguration

data_length = np.loadtxt(f"data/regression{graph_id}/relations_regression/relations_regressionlength400.txt", unpack=True)
x = data_length[0]
x_interval = np.linspace(np.min(x),np.max(x))
y1_desired = 0.2*x_interval + 0.3
h4 = ax[2].plot(x_interval, y1_desired, label = r'$V^{D} = 0.2V_{7}+0.3$', c = 'lightpink', lw=2, zorder=0)
y1 = data_length[1]
h5 = ax[2].scatter(x, y1, marker = 'o', s=plt.rcParams["lines.markersize"]**2, c = par.cmap.colors[0], lw=0.6, zorder=1, label=r'$L\ (step \ 400)$')

a = 0.35
b = 0.4
data_rho = np.loadtxt(f"data/regression{graph_id}/relations_regression/relations_regressionrho50_retrain.txt", unpack=True)
y2_desired = a*x_interval + b
h6 = ax[2].plot(x_interval, y2_desired, label = rf'$V = {a}V_{7}+{b}$', c = 'moccasin', lw=2, zorder=0)
y2 = data_rho[1]
h7 = ax[2].scatter(x, y2, marker = 'o', s=plt.rcParams["lines.markersize"]**2, c = par.cmap.colors[1], lw=0.6, zorder=1, label=rf'$\rho \ (step \ 400+50)$')

a=0.3
b=0.3
data_pressure = np.loadtxt(f"data/regression{graph_id}/relations_regression/relations_regressionpressure50_retrain.txt", unpack=True)
y2_desired = a*x_interval + b
h8= ax[2].plot(x_interval, y2_desired, label = rf'$V = {a}V_{7}+{b}$', c = 'mediumseagreen', lw=2, zorder=0)
y2 = data_pressure[1]
h9 = ax[2].scatter(x, y2, marker = 'o', s=plt.rcParams["lines.markersize"]**2, c = par.cmap.colors[3], lw=0.6, zorder=1, label=rf'$P \ (step \ 400+50)$')
ax[2].set_ylabel(r'$V_{6}$')
ax[2].set_xlabel(r'$V_{7}$')


handles = [h1, h2, h3, h4[0], h5, h6[0], h7, h8[0], h9]
labels = [l1, l2, l3, h4[0].get_label(), h5.get_label(), h6[0].get_label(), h7.get_label(), h8[0].get_label(), h9.get_label()]
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc="upper center", bbox_to_anchor=(0.665, 1.13), ncol=4)
fig.tight_layout()
fig.savefig(f"plots/paper/fig5.pdf")