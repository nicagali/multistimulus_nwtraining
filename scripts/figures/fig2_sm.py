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

graph_id = 'G00010001'
training_type = 'allostery'
DATA_PATH = f'data/{training_type}{graph_id}/'

fig, ax = plt.subplots(2, 2, figsize = (8,7))

relative_noisev = [0.1, 0.1, 0.01, 0.01]
for w, weight_type in enumerate(['length', 'radius_base', 'rho', 'pressure']):
    loaded = np.load(f"{DATA_PATH}mse/robustness_mse_{weight_type}.npy", allow_pickle=True).item()
    baseline = loaded["baseline_cost"]
    perturbed_costs = loaded["perturbed_costs"]
    relative_noise = relative_noisev[w]
    
    if w==2 or w==3:
        xlabel = r'$C(\boldsymbol{w})/C(\boldsymbol{w}^0)$'
    else:
        xlabel = None
        
    if w == 0 or w == 2:
        ylabel = 'Counts'
    else:
        ylabel = None

    plotting.plot_cost_histogram(ax.flat[w], baseline, perturbed_costs, weight_type=weight_type, relative_noise=relative_noise, xlabel=xlabel, ylabel=ylabel)
    
fig.tight_layout()
fig.savefig(f"plots/paper/fig2_sm_part1.png", dpi=300)

graph_id = 'G00020001'
training_type = 'regression'
DATA_PATH = f'data/{training_type}{graph_id}/'

fig, ax = plt.subplots(2, 2, figsize = (8,7))

relative_noisev = [0.1, 0.1, 0.01, 0.01]
for w, weight_type in enumerate(['length', 'radius_base', 'rho', 'pressure']):
    loaded = np.load(f"{DATA_PATH}mse/robustness_mse_{weight_type}.npy", allow_pickle=True).item()
    baseline = loaded["baseline_cost"]
    perturbed_costs = loaded["perturbed_costs"]
    relative_noise = relative_noisev[w]
    
    if w==2 or w==3:
        xlabel = r'$C(\boldsymbol{w})/C(\boldsymbol{w}^0)$'
    else:
        xlabel = None
        
    if w == 0 or w == 2:
        ylabel = 'Counts'
    else:
        ylabel = None

    plotting.plot_cost_histogram(ax.flat[w], baseline, perturbed_costs, weight_type=weight_type, relative_noise=relative_noise, xlabel=xlabel, ylabel=ylabel)
    
fig.tight_layout()
fig.savefig(f"plots/paper/fig2_sm_part2.png", dpi=300)