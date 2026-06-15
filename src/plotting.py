from pathlib import Path

import parameters as par
import networkx as nx
import numpy as np
import ahkab  
import networks
import training
import matplotlib.pyplot as plt


from matplotlib.colors import to_rgb, to_hex
def lighten_color(base_color, factor=0.5):
    """
    Lightens a given color by blending it with a lighter shade of itself.
    
    Parameters:
    - base_color: A HEX color or an (R, G, B) tuple in normalized [0, 1] range.
    - factor: Float, how much to lighten the color (0 = no change, 1 = fully white).
    
    Returns:
    - Lightened color as a HEX string.
    """
    base = to_rgb(base_color)  # Convert to RGB if it's a HEX string
    lightened = [(1 - factor) * c + factor for c in base]  # Blend toward lighter version of itself
    return to_hex(lightened)

def plot_graph(G, weight_type = None):

    # GET graph data
    # G = nx.read_graphml(f'data/{name_graph}.graphml')
    
    # GET color of node depending on which type of node it is
    color_attributes = [G.nodes[node]['color'] for node in G.nodes()]
    
    # LABEL nodes with potential, pressure and densty indeces
    labels = {node: fr'$V_{{{int(node)}}}$' for node in G.nodes()}
    
    # LABEL edges M_{i,j} : memristor that connect node i to node j (i = BASE, j = TIP)
    # edge_labels = {(u, v): fr'$M_{{ {int(u)}, {int(v)} }}$' for u, v in G.edges()}  
    edge_labels = {(u, v): fr'$M_{{{index+1}}}$' for index, (u, v) in enumerate(G.edges())}
    
    # CREATE a box on the edge that represents the memristor
    bbox = {'facecolor': 'lightblue', 'edgecolor': par.color_edges, 'alpha': 1, 'boxstyle': 'round,pad=0.5'}
    
    # GENERATE determined positions - Check out better with general nw
    # pos = nx.kamada_kawai_layout(G, scale=2)
    pos = nx.spring_layout(G)

    edge_weights = [G[u][v].get(f'{weight_type}', 1) for u, v in G.edges()]  # Default weight = 1 if missing
    max_weight = max(edge_weights) if edge_weights else 1
    widths = [2 + 10 * (w / max_weight) for w in edge_weights]  # Normalize and scale widths

    # DRAW the network
    nx.draw(G, with_labels=True, node_color=color_attributes,  
            pos = pos,
            node_size=par.nodes_size,
            labels = labels,
            width = widths,
            font_size=par.font_size_nw,
            edge_color=par.color_edges)
    # adding the memristors on the edges
    nx.draw_networkx_edge_labels(G, bbox=bbox, pos=pos, edge_labels=edge_labels, font_size=par.font_size_nw, font_color=par.color_font_edges)

    return  

def plot_mse(ax, fig, graph_id, training_type, weight_type, show_xlabel=True, label = True, relative_noise=0):
    
    if relative_noise>0:
        data = np.loadtxt(f"data/{training_type}{graph_id}/mse/mse_{weight_type}_noise{relative_noise}.txt", unpack=True)
    else:
        data = np.loadtxt(f"data/{training_type}{graph_id}/mse/mse_{weight_type}.txt", unpack=True)

    x = data[0]
    y = np.array(data[1])
    y = y

    if weight_type == 'best_choice':
        choosen_weight = np.loadtxt(f"data/{training_type}{graph_id}/weights/best_choice/choosen_weights.txt", unpack=True)
        choosen_weight = choosen_weight[1]
        color_vec = ['#F2CB05FF']
        possible_weights = ['length', 'radius_base', 'rho', 'pressure']
        
        for point in choosen_weight:
            point = int(point)
            # weight_type_par = weight_type[choosen_weight]
            style = par.weight_styles[f'{possible_weights[point]}']
            color_vec.append(style['c'])
            
        ax.scatter(x, y, color = color_vec, s=plt.rcParams["lines.markersize"]**2, marker = '^', lw = style['lw'], zorder=2)
        ax.plot(x, y, color = 'black', ls = ':', label = 'best choice', zorder=1)
        
    else:

        style = par.weight_styles[f'{weight_type}']
        color = style['c']
        
        if label:
            label = style['label']
        else:            
            label = None
            
        if relative_noise == 0.1:
            color = '#e7b46a'
        elif relative_noise == 0.05:
            color = '#A3AA5A'
        elif relative_noise == 0.01:
            color = '#f1a17e'
        elif relative_noise == 0.005:
            color = '#A3AA5A'
        elif relative_noise == 0.001:
            color = '#f1a17e'

        if relative_noise == 0.01 and weight_type in ('rho', 'pressure'):
                color = '#e7b46a'
        
        ax.plot(x, y, color = color, marker = style['marker'], lw = style['lw'], label = label, zorder=1)
        
    ax.set_yscale('log') 
    
    if relative_noise>0:
        avg_cost = np.mean(y[-100:-50])
        ax.axhline( y=avg_cost, color=color, linestyle='--', linewidth=2, label=fr'${relative_noise*100}\%$' )
      
    if training_type=='regression':
        ax.set_ylabel(r'$C_{test}(\boldsymbol{w})/C_{test}(\boldsymbol{w^0})$')
    else:
        ax.set_ylabel(r'$C(\boldsymbol{w})/C(\boldsymbol{w^0})$')

    if show_xlabel:
        ax.set_xlabel(r'Training steps')
    ax.tick_params(axis='both')


# Functions for weight plotting
def get_weight_count(G, weight_type):
    if weight_type in {"pressure", "rho"}:
        return G.number_of_nodes()
    return G.number_of_edges()


def get_weight_path(G, training_type, weight_type, step):
    return (Path('data') / f"{training_type}{G.graph['name']}" / "weights" / weight_type / f"{weight_type}{step}.txt" )


def load_weight_history(G, training_steps, training_type, weight_type, column, index, target_value = None):
    values = []

    for step in range(training_steps + 1):
        if target_value is None:
            filename = f"{weight_type}{step}.txt"
            path = Path(f"data/{training_type}{G.graph['name']}/weights/{weight_type}") / filename
        else:
            filename = f"{weight_type}{step}_target{target_value}.txt"
            path = Path(f"data/{training_type}{G.graph['name']}/multi_target") / filename

        data = np.loadtxt(path)

        values.append(data[index][column])

    return values

def make_palette(weight_type, n, color_factor=None):
    style = par.weight_styles[weight_type]

    if color_factor is None:
        if n == 2:
            color_factor = 0.6
        elif n == 3:
            color_factor = 0.4
        else:
            color_factor = 0.9 / (n - 1)

    return [ lighten_color(style["c"], factor=i * color_factor) for i in range(n) ]

def plot_weights(ax, G, training_steps, training_type, weight_type, target_value = None, show_xlabel=True, starting_step=0):
    x = list(range(starting_step, starting_step + training_steps + 1))

    if weight_type == "length_radius_base": 
        handles, labels = plot_length_radius_base( ax, G, training_steps, training_type, weight_type, x, show_xlabel )
        return handles, labels

    if weight_type == "length_pressure":
        handles, labels = plot_weights_len_press( ax, G, training_steps, training_type, weight_type, x, show_xlabel )
        return handles, labels

    # regular single-weight plotting

    n_weights = get_weight_count(G, weight_type)
    style = par.weight_styles[weight_type]
    palette = make_palette(weight_type, n_weights)

    for i in range(n_weights):
        
        weight = load_weight_history(G, training_steps, training_type, weight_type, column=1, index=i, target_value=target_value)

        label_name = style["label"][1:-1]

        ax.plot(x, weight, color=palette[i], marker=style["marker"], lw=style["lw"], label=rf"${{{label_name}}}_{{{i + 1}}}$")

    if show_xlabel:
        ax.set_xlabel("Training steps")

    ax.tick_params(axis="both")
    ax.set_ylabel(style["ylabel_weights"])

def plot_length_radius_base(ax, G, training_steps, training_type, weight_type, x, show_xlabel=True):
    ax2 = ax.twinx()

    n_weights = G.number_of_edges()

    length_style = par.weight_styles["length"]
    radius_style = par.weight_styles["radius_base"]

    palette_len = make_palette("length", n_weights, color_factor=0.6)
    palette_rad = make_palette("radius_base", n_weights, color_factor=0.6)

    lines = []

    for i in range(n_weights):
        length = load_weight_history(G, training_steps, training_type, weight_type, column=1, index=i )
        radius = load_weight_history(G, training_steps, training_type, weight_type, column=2, index=i )

        lines += ax.plot(x, length, color=palette_len[i], marker=length_style["marker"], lw=length_style["lw"], label=rf"$L_{{{i+1}}}$")

        lines += ax2.plot(x, radius, color=palette_rad[i], marker="^", lw=radius_style["lw"], label=rf"$R_{{b{i+1}}}$")

    if show_xlabel:
        ax.set_xlabel(r"Training steps")

    ax.tick_params(axis="both")
    ax2.tick_params(axis="both")

    # labels = [line.get_label() for line in lines]
    # ax.legend(lines, labels, loc=0)

    ax.set_ylabel(r"L[$\mu$m]")
    ax2.set_ylabel(r"$R_b$[$n$m]")
    ax2.set_ylim(70, 310)

    return lines, [line.get_label() for line in lines]


def plot_weights_len_press(ax, G, training_steps, training_type, weight_type, x, show_xlabel=True):

    n_len = G.number_of_edges()
    n_pres = G.number_of_nodes()

    ax2 = ax.twinx()
    lines = []

    length_style = par.weight_styles["length"]
    pressure_style = par.weight_styles["pressure"]

    palette_len = make_palette("length", n_len, color_factor=0.4)
    palette_pres = make_palette("pressure", n_pres, color_factor=0.4)

    # Length weights
    for i in range(n_len):
        weight_len = load_weight_history( G, training_steps, training_type, weight_type, column=1, index=i, )

        lines += ax.plot( x, weight_len, color=palette_len[i], marker=length_style["marker"], lw=length_style["lw"], label=rf"$L_{{{i + 1}}}$", )

    # Pressure weights
    for i in range(n_pres):
        weight_pres = load_weight_history( G, training_steps, training_type, weight_type, column=1, index=i + n_len, )

        lines += ax2.plot( x, weight_pres, color=palette_pres[i], marker="^", lw=pressure_style["lw"], label=rf"$P_{{{i + 1}}}$", zorder=1, )

    if show_xlabel:
        ax.set_xlabel(r"Training steps")

    ax.tick_params(axis="both")
    ax2.tick_params(axis="both")

    ax.set_ylim(4, 13 * 1.1)
    ax2.set_ylim(0.999, 1.001)

    # labels = [line.get_label() for line in lines]
    # ax2.legend(lines, labels)

    ax.set_ylabel(r"L[$\mu$m]")
    ax2.set_ylabel(r"$P$[bar]")

    return lines, [line.get_label() for line in lines]

def plot_memristor_resistances(ax, G):

    circuit = networks.circuit_from_graph(G, type='memristors') 
    analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None)

    # DEFINE a transient analysis (analysis of the circuit over time)
    result = ahkab.run(circuit, an_list=analysis) #returns two arrays: resistance over time of the memristors, voltages over time in the nodes
    resistances = result[1]
    
    x = np.array(range(len(resistances)))

    for edge_index in range(len(G.edges())):
        y = [1/(resistances[time_index][edge_index]) for time_index in range(len(resistances))]
        ax.plot(x, y, **par.memr_resistances_style, label = rf'$M_{{{edge_index+1}}}$') 
    ax.legend()
    ax.set_ylabel(r'$c[pS]$')
    ax.set_xlabel(r't[s]')
    ax.tick_params(axis='both')

def plot_potential_each_node(ax, G, factor_time=1):

    circuit = networks.circuit_from_graph(G, type='memristors')

    tran_analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None)
    result = ahkab.run(circuit, an_list=tran_analysis)  
    resistances = result[1][-1]

    result = result[0]

    time = result['tran']['T']  

    index_input = 0
    index_target = 0
    index_hidden = 0
    for node in G.nodes():

        potential = result['tran'][f'VN{node}']

        if G.nodes[node]['type'] == 'source':
            ax.plot(time, potential, lw=3, color = par.color_dots[0], label = fr"$V_{int(node)+1}$")
            index_input += 1
            
        if G.nodes[node]['type'] == 'target':
            desired_value = G.nodes[node]['desired']
            ax.plot(time, potential, lw=1, marker = 'o', color = par.color_dots[1], label = fr"$V_{int(node)+1}$")
            index_target += 1
            print(f'Final potential node {int(node)+1} = ', potential[-1])
            
        if G.nodes[node]['type'] == 'hidden':
            ax.plot(time, potential, lw=3, color = par.color_dots[2], label = fr"$V_{int(node)+1}$")
            index_hidden += 1


    ax.plot((time[0], time[-1]), (desired_value, desired_value), color='mediumpurple', lw=1.5, ls='--', label=r'$V_D$')

    ax.margins(x=0)
    ax.legend()
    ax.set_ylabel(r'$V$[V]')
    ax.set_xlabel(r'$t$[s]')
    ax.tick_params(axis='both')

def plot_multi_train(ax1, ax2, G, training_steps, weight_type, target_values):
    
    for target_index, target_value in enumerate(target_values):

        y = np.loadtxt(f"data/allosteryvd/multi_target/potential_targets{target_value}.txt", unpack=True)
        x = np.array(range(len(y))) + target_index*(len(y)-1)

        # PLOT 

        # Target
        ax1.plot(x, y, lw=1, marker = 'o', color = par.color_dots[1], label="$V_2$" if target_index == 0 else "_nolegend_", zorder = 1)        
        # Inputs
        ax1.plot(x, [5 for _ in range(len(x))], lw=3, color = par.color_dots[0], label="$V_1$" if target_index == 0 else "_nolegend_")
        ax1.plot(x, [0 for _ in range(len(x))], lw=3, color = par.color_dots[0], label = "$V_3$" if target_index == 0 else "_nolegend_")
        # Desired
        ax1.plot(x, [target_value for _ in range(len(x))],lw=1.5, ls='--', color = 'mediumpurple', label = "$V_D$" if target_index == 0 else "_nolegend_", zorder = 0)
        
        plot_weights(ax2, G, training_steps=training_steps, training_type='allostery', weight_type=weight_type, target_value=target_value, show_xlabel=True, starting_step=target_index * training_steps)

    ax1.set_ylabel(r'$V[V]$')
    ax1.tick_params(axis='both')

def plot_regression(ax, graph_id, weight_type, step, a = 0.2, b = 0.3, marker=None):
    
    data = np.loadtxt(f"data/regression{graph_id}/relations_regression/relations_regression{weight_type}{step}.txt", unpack=True)
    x = data[0]
    y1 = data[1]
    
    x_interval = np.linspace(np.min(x),np.max(x))
    y1_desired = training.regression_function(x_interval, a, b)

    style_des = par.regression_styles[f'{weight_type}_des']
    style = par.regression_styles[f'{weight_type}']
    style['marker'] = marker if marker is not None else style.get('marker', 'o')
    style['s'] = plt.rcParams["lines.markersize"]**2
    ax.plot(x_interval, y1_desired, **style_des, zorder=0)
    
    handle = ax.scatter(x, y1, label = rf'$L\ (step \ {step})$', **style)
    
    ax.set_ylabel(r'$V_{6}$')
    ax.set_xlabel(r'$V_{7}$')
    
    return handle, handle.get_label()

from pypalettes import load_cmap
def plot_cost_histogram(ax, baseline_cost, perturbed_costs, weight_type, relative_noise, title=None, bins=100, xlabel = None, ylabel = None):
    
    cmap = load_cmap("Callanthias_australis")
    colors = {weight_type: cmap(i) for i, weight_type in enumerate(["length", "rho", "radius_base", "pressure"])}
    colors_lines = {'length': 'darkviolet', 'radius_base': 'orange', 'pressure': 'green', 'rho': 'red'}


    if weight_type=='length':
        label = r'$L$'
    if weight_type=='radius_base':
        label = r'$R_b$'
    if weight_type=='rho':
        label = r'$\rho$'
    if weight_type=='pressure':
        label = r'$P$'

    ax.hist(perturbed_costs, bins=bins, alpha=0.75, color=colors[weight_type], label=fr'{label}  (${relative_noise*100:.0f}\%$)')

    ax.axvline(baseline_cost,linestyle='--',linewidth=2, color = colors_lines[weight_type], label='Unperturbed')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend()