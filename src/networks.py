import sys
sys.path.append("src/")
import networkx as nx
import parameters as par
import numpy as np
import ahkab    
import random

# This module contains functions to:
# 1 - Initialize graph parameters
# 2 - Define different graphs
# 3 - Transform a graph in a circuit

# 1 --------- INITIALIZE GRAPH PARAMETERS ---------

def initialize_nodes(G, sources, targets, voltage_input=None, voltage_desired=None):

    # Assign attributes nodes
    constantsource_index=0
    for node in list(G.nodes()):
        # print(node)
        if node in sources:
            G.nodes[node]['type'] = 'source'
            G.nodes[node]['color'] = par.color_dots[0]
            if constantsource_index < (len(sources)-1):
                G.nodes[node]['constant_source'] = True
                constantsource_index += 1
            else:
                G.nodes[node]['constant_source'] = False

        elif node in targets:
            G.nodes[node]['type'] = 'target'
            G.nodes[node]['color'] = par.color_dots[1]
        else:
            G.nodes[node]['type'] = 'hidden'
            G.nodes[node]['color'] = par.color_dots[2]

    # Initialize every node with the same values of pressure = initial_pressure and rho = initial_rho
    initial_rho = 0.2
    initial_pressure = 1

    index_sources=0
    index_desired=0
    for node in G.nodes():

        G.nodes[node]['rho'] = initial_rho 
        G.nodes[node]['pressure'] = initial_pressure

    # Initialize source and desired voltages
        if G.nodes[node]['type'] == 'source':
            G.nodes[node]['voltage'] = voltage_input[index_sources]            
            index_sources+=1
        if G.nodes[node]['type'] == 'target':
            # print(G.nodes[node]['desired'])
            G.nodes[node]['desired'] = voltage_desired[index_desired]
            index_desired+=1


def initialize_edges(G, mix_base_tip = False):

    initial_length = 10 # [mu m]
    initial_radius_base = 200 # [nm]
    initial_value_resistance = 50 #do I still need this??
    initial_value_conductance = 1/initial_value_resistance

    edge_list = list(G.edges())

    # Reverse base and tip diection if asked
    for edge in edge_list:

        if mix_base_tip:
            dice = random.random()  

            if dice > 0.5:

                G.remove_edge(edge[0], edge[1])  # Remove the old edge
                G.add_edge(edge[1], edge[0])  # Add the reversed edge

    # Initialize network edges
    for edge in G.edges():

        G.edges[edge]['resistance'] = initial_value_resistance
        G.edges[edge]['conductance'] = initial_value_conductance
        G.edges[edge]['length'] = initial_length
        G.edges[edge]['radius_base'] = initial_radius_base
        G.edges[edge]['pressure'] = 0
        G.edges[edge]['delta_rho'] = 0

# 2 --------- DEFINE DIFFERENT GRAPHS ---------


# SINGLE MEMRISTOR: 0 --- 1 
def single_memristor(save_data=False):

    G = nx.DiGraph()    # I am using directed graphs to keep trak of sign when def circuit
    G.name = 'single_memristor'

    # ADD nodes
    attributes = {"type" : "source", 'color' : par.color_dots[0]}
    G.add_node(0, **attributes)
    
    attributes = {"type" : "source", 'color' : par.color_dots[0]}
    G.add_node(1, **attributes)

    # ADD edges
    G.add_edge(0,1)
    
    # INITIALIZE nodes and edges
    voltage_input = [5, 0] # node initialized here because different for differnent nw

    initialize_nodes(G, sources=[0,1], targets=None, voltage_input=voltage_input)
    initialize_edges(G, mix_base_tip=False)

    # SAVE to data folder
    if save_data:
        nx.write_graphml(G, f"{par.DATA_PATH}single_memristor.graphml")

    return G

# VOLTAGE DIVIDER: 0 --- 1 --- 2
def voltage_divider(save_data=False, voltage_desired = [4]):

    G = nx.DiGraph()    # I am using directed graphs to keep trak of sign when def circuit
    G.name = 'vd'

    # ADD nodes
    attributes = {"type" : "source", 'color' : par.color_dots[0]}
    G.add_node(0, **attributes)
    G.nodes[0]['constant_source'] = False
    attributes = {"type" : "target", 'color' : par.color_dots[1]}
    G.add_node(1, **attributes)
    attributes = {"type" : "source", 'color' : par.color_dots[0]}
    G.add_node(2, **attributes)
    G.nodes[2]['constant_source'] = True

# learning_rate_vec = [1e-5, 2e-6, 5e-3, 5e2]

    # ADD edges
    G.add_edge(0,1)
    G.add_edge(1,2)

    
    # INITIALIZE nodes and edges
    voltage_input = [5, 0] # node initialized here because different for differnent nw
    voltage_desired = [3]

    initialize_nodes(G, sources=[0,2], targets=[1], voltage_input=voltage_input, voltage_desired=voltage_desired)
    initialize_edges(G, mix_base_tip=False)

    # SAVE to data folder
    if save_data:
        nx.write_graphml(G, f"data/networks/vd.graphml")

    return G

# RANDOM NETWORK
 
def random_graph(graph_id, number_nodes=9, number_edges=12, number_sources=5, number_targets = 3, save_data=False, res_change=False):

    # CREATE random graph with number_nodes conected by number_edges
    G = nx.dense_gnm_random_graph(number_nodes, number_edges)
    G.name = f'{graph_id}'

    # DEFINE number sources and targets, then randomly select sources and targets nodes between number_nodes : sources containg source index and targets contains target indeces
    sources = random.sample(list(G.nodes()), number_sources)
    target_sampling_list = [x for x in G.nodes() if x not in sources]
    targets = random.sample(target_sampling_list, number_targets)

    voltage_input = [0, 5, 3, 2, 1] # node initialized here because different for differnent nw
    voltage_desired = [2, 2, 3]

    # INITIALIZE nodes and edges
    initialize_nodes(G, sources, targets, voltage_input, voltage_desired)
    initialize_edges(G)

    if save_data:

        nx.write_graphml(G, f"data/networks/{G.name}.graphml")

    return G

# 3 --------- GRAPH -> CIRCUIT ---------
# Create a the class 'Circuit' used in the package ahkab from the desired graph.
def circuit_from_graph(G, type, imposed_pressure=None):

    circuit = ahkab.circuit.Circuit('Circuit')
    
    # ADD voltage sources 
    for node in G.nodes():

        # Adding the voltage sources from ground to input nodes
        if G.nodes[node]['type'] == 'source':
            # print(node, G.nodes[node]['type'], G.nodes[node]['voltage'])
            circuit.add_vsource(f"VN{node}", n1=f'n{node}', n2=circuit.gnd, dc_value=G.nodes[node]['voltage'])
            
    # ADD elements on links
    for index, edge in enumerate(G.edges()):
        
        # An edge = (u,v), the nodes are then called 'n u' and 'n v', u = edge[0], ...

        if type == 'memristors':
            if imposed_pressure==None:
                circuit.add_mysistor(f'M{index+1}', f'n{edge[0]}', f'n{edge[1]}', value = G.edges[edge]["resistance"], rho_b=G.nodes[edge[0]]['rho'], length_channel = G.edges[edge]['length']*1e-6, radius_base = G.edges[edge]['radius_base']*1e-9, pressure=(G.nodes[edge[0]]['pressure']-G.nodes[edge[1]]['pressure'])*1e5, delta_rho = (G.nodes[edge[0]]['rho']-G.nodes[edge[1]]['rho']))
                # print(G.edges[edge]['length'], (G.nodes[edge[0]]['rho']-G.nodes[edge[1]]['rho']))
            else:
                circuit.add_mysistor(f'M{index+1}', f'n{edge[0]}', f'n{edge[1]}', value = G.edges[edge]["resistance"], rho_b=G.nodes[edge[0]]['rho'], length_channel = G.edges[edge]['length']*1e-6, radius_base = G.edges[edge]['radius_base']*1e-9, pressure=(imposed_pressure)*1e5, delta_rho = (G.nodes[edge[0]]['rho']-G.nodes[edge[1]]['rho']))

        else:

            circuit.add_resistor(f'R{index}', f'n{edge[0]}', f'n{edge[1]}', value = G.edges[edge]['resistance'])
        
    return circuit


def to_directed_graph(G_structure, shuffle = False):

    G = nx.DiGraph()
    voltage_input = []
    voltage_desired = []
    
    nodes = [node for node in G_structure.nodes()]
    targets = [x for x in G_structure.nodes() if G_structure.nodes[x]['type']=='target']
    sources = [x for x in G_structure.nodes() if G_structure.nodes[x]['type']=='source']
    voltage_input = [0, 5, 3, 2, 1] # node initialized here because different for differnent nw
    voltage_desired = [2, 2, 3]

    G.add_nodes_from(nodes)

    initialize_nodes(G, sources, targets, voltage_input, voltage_desired)

    edges = [edge for edge in G_structure.edges()]
    if shuffle:
        for edge_index in range(len(edges)):
            direction = random.random()
            if direction>0.3:
                edges[edge_index] = (edges[edge_index][0], edges[edge_index][1])
            else:
                edges[edge_index] = (edges[edge_index][1], edges[edge_index][0])

    G.add_edges_from(edges)

    initialize_edges(G)
    
    G.graph['name'] = G_structure.graph['name']
    
    return G    


def reverse(self, copy=True):
    """Returns the reverse of the graph.

    The reverse is a graph with the same nodes and edges
    but with the directions of the edges reversed.

    Parameters
    ----------
    copy : bool optional (default=True)
        If True, return a new DiGraph holding the reversed edges.
        If False, the reverse graph is created using a view of
        the original graph.
    """
    if copy:
        H = self.__class__()
        H.graph.update(deepcopy(self.graph))
        H.add_nodes_from((n, deepcopy(d)) for n, d in self.nodes.items())
        H.add_edges_from((v, u, deepcopy(d)) for u, v, d in self.edges(data=True))
        return H
    return nx.reverse_view(self)