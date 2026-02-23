import parameters as par
import networkx as nx
import numpy as np
import ahkab  
import networks
import training

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

def plot_mse(ax, fig, graph_id, training_type, weight_type, show_xlabel=True, var_l=False):
    
    if var_l:
        data = np.loadtxt(f"data/{training_type}{graph_id}/mse/mse_{weight_type}_var.txt", unpack=True)
    else:
        data = np.loadtxt(f"data/{training_type}{graph_id}/mse/mse_{weight_type}.txt", unpack=True)

    x = data[0]
    y = np.array(data[1])
    y = y

    style = par.weight_styles[f'{weight_type}']
    if var_l:
        color = 'red'
    else:
        color = style['c']

    ax.plot(x, y, color = color, marker = style['marker'], lw = style['lw'], label = style['label'], zorder=1)
    ax.set_yscale('log')   
    
    if training_type=='regression':
        ax.set_ylabel(r'$C_{test}(w)$', fontsize = par.axis_fontsize)
    else:
        ax.set_ylabel(r'$C(w)$', fontsize = par.axis_fontsize)

    if show_xlabel:
        ax.set_xlabel(r'Training steps', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)

def plot_weights(ax, G, training_steps, training_type, weight_type, show_xlabel=True, starting_step = 0):

    x =list(range(starting_step, starting_step + training_steps+1))
    
    if weight_type == 'pressure' or weight_type == 'rho':
        number_weights = G.number_of_nodes()
    else:
        number_weights = G.number_of_edges()
        
    # CREATE color palettes with lighter shades of base color (the color of the mse)
    if number_weights == 2:
        color_factor = 0.6
    elif number_weights == 3:
        color_factor = 0.4
    else:
        color_factor = 0.9 / ((number_weights) - 1)
    
    style = par.weight_styles[f'{weight_type}']
    base_color = style['c']
    palette = [lighten_color(base_color, factor = i * color_factor) for i in range(number_weights)]
    # palette = plt.get_cmap('tab20')
      
    # GET data: data/training_job/weight_type contains files weight_type{step} with the list of weights per step
    if weight_type == 'length_radius_base':
        ax2 = ax.twinx()  
            
        color_factor = 0.6
        
        style = par.weight_styles[f'length']
        base_color = style['c']
        palette = [lighten_color(base_color, factor = i * color_factor) for i in range(number_weights)]
        
        style = par.weight_styles[f'radius_base']
        base_color = style['c']
        palette_rad = [lighten_color(base_color, factor = i * color_factor) for i in range(number_weights)]

    for weight_indx in range(number_weights):

        weight = []
        weight2 = []
        for step in range(training_steps+1):
            data = np.loadtxt(f"data/{training_type}{G.graph['name']}/weights/{weight_type}/{weight_type}{step}.txt", unpack=True)
            y = data[1]
            weight.append(y[weight_indx])
            if weight_type=='length_radius_base':
                y2 = data[2]
                weight2.append(y2[weight_indx])

        if weight_type=='length_radius_base':
            plt1 = ax.plot(x, weight, color=palette[weight_indx], marker = style['marker'], lw = style['lw'], label = rf'$L_{{{weight_indx}}}$')
            plt2 = ax2.plot(x, weight2, color=palette_rad[weight_indx], marker = '^', lw = style['lw'], label = rf'$R_{{b{weight_indx}}}$')
            if weight_indx==0:
                plts = plt1 + plt2
            else:
                plts += plt1 + plt2
        else:
            
            label_without_weightindex = style['label'][1:-1]
            ax.plot(x, weight, color=palette[weight_indx], marker = style['marker'], lw = style['lw'], label = rf'${{{label_without_weightindex}}}_{{{weight_indx+1}}}$')

    if show_xlabel:
        ax.set_xlabel(r'Training steps', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)

    if weight_type=='length_radius_base':
        # plts = plt1 + plt2
        labs = [l.get_label() for l in plts]
        ax.legend(plts, labs, loc=0)
        ax.set_ylabel(rf'L[$\mu$m]', fontsize = par.axis_fontsize)
        ax2.set_ylabel(rf'$R_b$[$n$m]', fontsize = par.axis_fontsize)
        ax2.set_ylim(70,310)
        ax2.tick_params(axis='both', labelsize=par.size_ticks)
    else:
        label = style['ylabel_weights']
        # ax.legend(fontsize = par.legend_size)
        ax.set_ylabel(f'{label}', fontsize = par.axis_fontsize)

def plot_weights_len_press(ax, G, training_steps, training_type, weight_type, show_xlabel=True, starting_step = 0):

    x =list(range(starting_step, starting_step + training_steps+1))

    number_weights_pres = G.number_of_nodes()

    number_weights_len = G.number_of_edges()
        

    color_factor = 0.4
    
    style = par.weight_styles[f'length']
    base_color = style['c']
    palette_len = [lighten_color(base_color, factor = i * color_factor) for i in range(number_weights_len)]
    
    style = par.weight_styles[f'pressure']
    base_color = style['c']
    palette_pres = [lighten_color(base_color, factor = i * color_factor) for i in range(number_weights_pres)]
    
    ax2 = ax.twinx()  

    plts = []  # list of all plot handles

    # Length weights
    for weight_indx in range(number_weights_len):
        weight_len = []
        for step in range(training_steps+1):
            data = np.loadtxt(
                f"data/{training_type}{G.graph['name']}/weights/{weight_type}/{weight_type}{step}.txt",
                unpack=True
            )
            y = data[1]
            weight_len.append(y[weight_indx])

        line = ax.plot(
            x, weight_len,
            color=palette_len[weight_indx],
            marker=style['marker'],
            lw=style['lw'],
            label=rf'$L_{{{weight_indx+1}}}$'
        )
        plts += line   # extend list with line handle(s)

    # Pressure weights
    for weight_indx in range(number_weights_pres):
        weight_pres = []
        for step in range(training_steps+1):
            data = np.loadtxt(
                f"data/{training_type}{G.graph['name']}/weights/{weight_type}/{weight_type}{step}.txt",
                unpack=True
            )
            y = data[1]
            weight_pres.append(y[weight_indx + number_weights_len])

        line = ax2.plot(
            x, weight_pres,
            color=palette_pres[weight_indx],
            marker='^',
            lw=style['lw'],
            label=rf'$P_{{{weight_indx+1}}}$',
            zorder=1
        )
        plts += line

    if show_xlabel:
        ax.set_xlabel(r'Training steps', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)

    ax.set_ylim(4,13 * 1.1)
    ax2.set_ylim(0.999,1.001)
    # ax2.set_ylim(np.min(weight_pres), np.max(weight_pres) * 1.1)

    # plts = plt1 + plt2
    labs = [l.get_label() for l in plts]
    ax2.legend(plts, labs)
    ax.set_ylabel(rf'L[$\mu$m]', fontsize = par.axis_fontsize)
    ax2.set_ylabel(rf'$P$[bar]', fontsize = par.axis_fontsize)
    ax2.tick_params(axis='both', labelsize=par.size_ticks)

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
    ax.set_ylabel(r'$c[pS]$', fontsize = par.axis_fontsize)
    ax.set_xlabel(r't[s]', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)

def plot_potential_each_node(ax, G, factor_time=1):

    circuit = networks.circuit_from_graph(G, type='memristors')

    tran_analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None)
    result = ahkab.run(circuit, an_list=tran_analysis)  
    resistances = result[1][-1]
    # print(1/resistances[0], 1/resistances[1])

    result = result[0]
    # print(result['tran'].keys())

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
    ax.set_ylabel(r'$V[V]$', fontsize = par.axis_fontsize)
    ax.set_xlabel(r't[s]', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)

def plot_final_potential_vd(ax, target_values):
    
    for target_index, target_value in enumerate(target_values):

        y = np.loadtxt(f"data/potential_targets/potential_targets{target_value}.txt", unpack=True)
        x = np.array(range(len(y))) + target_index*(len(y)-1)

        # PLOT 

        # Target
        ax.plot(x, y, lw=1, marker = 'o', color = par.color_dots[1], label="$V_2$" if target_index == 0 else "_nolegend_", zorder = 1)        
        # Inputs
        ax.plot(x, [5 for _ in range(len(x))], lw=3, color = par.color_dots[0], label="$V_1$" if target_index == 0 else "_nolegend_")
        ax.plot(x, [0 for _ in range(len(x))], lw=3, color = par.color_dots[0], label = "$V_3$" if target_index == 0 else "_nolegend_")
        # Desired
        ax.plot(x, [target_value for _ in range(len(x))],lw=1.5, ls='--', color = 'mediumpurple', label = "$V_D$" if target_index == 0 else "_nolegend_", zorder = 0)

    ax.set_ylabel(r'$V[V]$', fontsize = par.axis_fontsize)
    # ax.set_xlabel(r'Time steps', fontsize = par.axis_fontsize)
    ax.tick_params(axis='both', labelsize=par.size_ticks)

def plot_regression(ax, graph_id, weight_type, step):
    
    data = np.loadtxt(f"data/regression{graph_id}/relations_regression/relations_regression{weight_type}{step}.txt", unpack=True)
    x = data[0]
    y1 = data[1]
    
    x_interval = np.linspace(np.min(x),np.max(x))
    y1_desired = training.regression_function(x_interval)

    style_des = par.regression_styles[f'{weight_type}_des']
    style = par.regression_styles[f'{weight_type}']

    ax.plot(x_interval, y1_desired, **style_des, zorder=0)
    
    ax.scatter(x, y1, **style)
    
    ax.set_ylabel(r'$V_{out}$', fontsize = (par.axis_fontsize))
    ax.set_xlabel(r'$V_{in}$', fontsize = (par.axis_fontsize))
    
    ax.grid(ls=":")
    ax.tick_params(axis='both', labelsize=par.size_ticks)
    
    return


