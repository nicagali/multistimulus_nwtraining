import parameters as par
import networks
import multiprocessing as mp
import numpy as np
import sys
import networkx as nx
from pathlib import Path
import ahkab 

# --------- WRITE RESULTS TO FILES ---------

def write_weights_to_file(G, DATA_PATH, step, weight_type):

    file = open(f"{DATA_PATH}/{weight_type}{step}.txt", "w")
    
    for index, edge in enumerate(G.edges()): 
        if weight_type=='length' or weight_type=='radius_base' or weight_type=='resistance':
            file.write(f"{index}\t{G.edges[edge][weight_type]}\n")
        if weight_type=='length_radius_base' or weight_type=='best_choice':
            file.write(f"{index}\t{G.edges[edge]['length']}\t{G.edges[edge]['radius_base']}\n")

    for node in G.nodes():
        if weight_type=='pressure' or weight_type=='rho':  
            file.write(f"{int(node)}\t{G.nodes[node][weight_type]}\n")
        if weight_type=='best_choice':
            file.write(f"{int(node)}\t{G.nodes[node]['pressure']}\t{G.nodes[node]['rho']}\n")

    if weight_type=='length_pressure':
        for index, edge in enumerate(G.edges()): 
            file.write(f"{index}\t{G.edges[edge]['length']}\n")
        for node in G.nodes():
            file.write(f"{int(node)}\t{G.nodes[node]['pressure']}\n")
            
    file.close()
    
    return

# --------- UTILITIES FUNCTIONS ---------

def regression_function(input):
    # return 0.2*input + 0.3
    return 0.3*input + 0.4
    # return 0.35*input + 0.4

# --------- ALGORITHM FUNCTIONS ---------

def cost_function(G, weight_type, write_potential_target_to_file=None, update_initial_res = False, varying_len=False):
    
    # TRANSFORM graph into circuit
    if weight_type == 'resistance':
        circuit = networks.circuit_from_graph(G, type='resistors') 
        analysis = ahkab.new_op()
    else:
        circuit = networks.circuit_from_graph(G, type='memristors') 
        analysis = ahkab.new_tran(tstart=0, tstop=0.05, tstep=1e-3, x0=None, varying_len=varying_len)

    # DEFINE a transient analysis (analysis of the circuit over time)
    result = ahkab.run(circuit, an_list=analysis) #returns two arrays: resistance over time of the memristors, voltages over time in the nodes

    resistance_vec = result[1]
    result = result[0]

    # UPDATE the value of the initial conductance to the last in the tran simulation to speed up (in a voltage divider 5 steps speeds of 0.01s giving same results)
    if update_initial_res:
        for index, edge in enumerate(G.edges()):
            resistance = resistance_vec[-1][index]
            G.edges[edge]['conductance'] = resistance

    # COMPUTE error     
    error=0
    for node in G.nodes():
        target_index=0
        if G.nodes[node]['type']=='target':  
            if weight_type =='resistance':
                error += (G.nodes[node]['desired'] - result['op'][f'VN{node}'][0][0])**2
            else: 
                error += (G.nodes[node]['desired'] - result['tran'][f'VN{node}'][-1])**2

            target_index+=1
            # print("error in cost function", node, G.nodes[node]['desired'], result['tran'][f'VN{node}'][-1])
            
    # WRITE last element potential drop each edge (useful in the voltage divider case, otherwise too many)
    if G.name == 'vd' and write_potential_target_to_file is not None:

        if weight_type == 'resistance':
            write_potential_target_to_file.write(f"{result['op'][f'VN{1}'][0][0]}\n")
        else:
            write_potential_target_to_file.write(f"{result['tran'][f'VN{1}'][-1]}\n")


    return error    

def update_input_output_volt(G, input_voltage, desired_voltage=None, datastep=None):

    input_index = 0
    output_index = 0
    
    for node in G.nodes():
        
        # if G.nodes[node]['type'] == 'source':
            # print(datastep, (int(node)+1), G.nodes[node]['constant_source'], G.nodes[node]['voltage'])
            
        
        if G.nodes[node]['type'] == 'source' and G.nodes[node]['constant_source']==False:

            # print(datastep, (int(node)+1), input_voltage[datastep][input_index])
            # print(input_voltage)

            G.nodes[node]['voltage'] = input_voltage[datastep][input_index]
            input_index += 1

        if desired_voltage is not None and G.nodes[node]['type'] == 'target':

            # print(datastep, (int(node)+1), desired_voltage[datastep][output_index])

            # if isinstance(desired_voltage[datastep], (list, tuple, np.array)):
            G.nodes[node]['desired'] = desired_voltage[datastep][output_index]
            # else:
                # G.nodes[node]['desired'] = desired_voltage[datastep]
            output_index += 1

def cost_function_regression(G, weight_type, dataset_input_voltage, dataset_output_voltage, datastep=None):

    if datastep != None:
        update_input_output_volt(G, dataset_input_voltage, dataset_output_voltage, datastep)
        error = cost_function(G, weight_type)

    else:

        error = 0
        for datastep in range(len(dataset_input_voltage)):
            update_input_output_volt(G, dataset_input_voltage, dataset_output_voltage, datastep)
            error += cost_function(G, weight_type)
            # print(error) 
        error = error/len(dataset_input_voltage)

    return error

def update_weights(G, training_type, base_error, weight_type, delta_weight, learning_rate, dataset_input_voltage, dataset_output_voltage, datastep, varying_len=False):

    gradients = []
    gradients2 = []
    
    if weight_type=='pressure' or weight_type == 'rho':

        for node in G.nodes():

            G_increment = G.copy(as_view=False)
            
            G_increment.nodes[node][f'{weight_type}'] += delta_weight

            if training_type == 'allostery':
                error = cost_function(G_increment, weight_type, varying_len=varying_len)  
            else:
                error = cost_function_regression(G_increment, weight_type, dataset_input_voltage, dataset_output_voltage, datastep)

            if weight_type == 'pressure':
                denominator = delta_weight*1e5
            else:
                denominator = delta_weight
                
            gradients.append((error - base_error)/denominator)


        for node in G.nodes():

            G.nodes[node][f'{weight_type}'] -= learning_rate*gradients[int(node)]

    elif weight_type == 'resistance':

        for index, edge in enumerate(G.edges()):

            G_increment = G.copy(as_view=False)
  
            G_increment.edges[edge][f'{weight_type}'] += delta_weight
            
            if training_type == 'allostery':
                error = cost_function(G_increment, weight_type, varying_len=varying_len)  
            else:
                error = cost_function_regression(G_increment, weight_type, dataset_input_voltage, dataset_output_voltage, datastep)

            gradients.append((error - base_error)/delta_weight)
            
        for index, edge in enumerate(G.edges()):    #Different loop cause you don't want to change edges yet
            
            G.edges[edge][f'{weight_type}'] -= learning_rate*gradients[index]
            if G.edges[edge][f'{weight_type}'] < 0:
                G.edges[edge][f'{weight_type}'] += learning_rate*gradients[index]
                print(f"Error: Negative weight detected for edge {edge} with weight type '{weight_type}'.")

    elif weight_type=='length_radius_base':

        for index, edge in enumerate(G.edges()):

            G_increment = G.copy(as_view=False)
  
            G_increment.edges[edge][f'length'] += delta_weight[0]
            
            if training_type == 'allostery':
                error = cost_function(G_increment, weight_type, varying_len=varying_len)  
            else:
                error = cost_function_regression(G_increment, weight_type, dataset_input_voltage, dataset_output_voltage, datastep)

            denominator = delta_weight[0]*1e-6
            gradients.append((error - base_error)/denominator)

            G_increment = G.copy(as_view=False)
            G_increment.edges[edge][f'radius_base'] += delta_weight[1]

            if training_type == 'allostery':
                error = cost_function(G_increment, weight_type, varying_len=varying_len)  
            else:
                error = cost_function_regression(G_increment, weight_type, dataset_input_voltage, dataset_output_voltage, datastep)

            denominator = delta_weight[1]*1e-9
            gradients2.append((error - base_error)/denominator)

        for index, edge in enumerate(G.edges()):    #Different loop cause you don't want to change edges yet
            G.edges[edge]['length'] -= learning_rate[0]*gradients[index]
            G.edges[edge]['radius_base'] -= learning_rate[1]*gradients2[index]

    elif weight_type=='length_pressure':

        for index, edge in enumerate(G.edges()):

            G_increment = G.copy(as_view=False)
  
            G_increment.edges[edge][f'length'] += delta_weight[0]
            
            if training_type == 'allostery':
                error = cost_function(G_increment, weight_type, varying_len=varying_len)  
            else:
                error = cost_function_regression(G_increment, weight_type, dataset_input_voltage, dataset_output_voltage, datastep)

            denominator = delta_weight[0]*1e-6
            gradients.append((error - base_error)/denominator)

        for node in G.nodes():

            G_increment = G.copy(as_view=False)
            
            G_increment.nodes[node][f'pressure'] += delta_weight[1]

            if training_type == 'allostery':
                error = cost_function(G_increment, weight_type, varying_len=varying_len)  
            else:
                error = cost_function_regression(G_increment, weight_type, dataset_input_voltage, dataset_output_voltage, datastep)

            denominator = delta_weight[1]*1e5
            gradients2.append((error - base_error)/denominator)

        for index, edge in enumerate(G.edges()):    #Different loop cause you don't want to change edges yet
            G.edges[edge]['length'] -= learning_rate[0]*gradients[index]
        for node in G.nodes():
            G.nodes[node][f'pressure'] -= learning_rate[1]*gradients2[int(node)]

    else:

        for index, edge in enumerate(G.edges()):

            G_increment = G.copy(as_view=False)
  
            G_increment.edges[edge][f'{weight_type}'] += delta_weight
            
            if training_type == 'allostery':
                error = cost_function(G_increment, weight_type, varying_len=varying_len)  
            elif training_type == 'regression':
                error = cost_function_regression(G_increment, weight_type, dataset_input_voltage, dataset_output_voltage, datastep)

            if weight_type == 'length':
                denominator = delta_weight*1e-6
            else:
                denominator = delta_weight*1e-9

            gradients.append((error - base_error)/denominator)
            
        for index, edge in enumerate(G.edges()):    #Different loop cause you don't want to change edges yet

            G.edges[edge][f'{weight_type}'] -= learning_rate*gradients[index]

            if G.edges[edge][f'{weight_type}'] < 0:
                G.edges[edge][f'{weight_type}'] += learning_rate*gradients[index]
                sys.exit(f"Error: Negative weight detected for edge {edge} with weight type '{weight_type}'.")
        
    return G

def update_output(G, voltages):

    index_do = 0
    eta = 0.1

    for node in G.nodes():

        if G.nodes[node]['type'] == 'target':

            # print('upadting clamped', node, voltages[int(node)], (G.nodes[node]['desired']), eta*(G.nodes[node]['desired']) + (1 - eta) * (voltages[int(node)]))

            G.nodes[node]['type'] = 'source'

            G.nodes[node]['voltage'] = eta*(G.nodes[node]['desired']) + (1 - eta) * (voltages[int(node)])
            # G.nodes[node]['voltage'] = 1


    return G

def update_resistances(G_free, training_type, dataset_input_voltage, dataset_output_voltage, step):

    eta = 0.1
    alpha = 1000
    gamma = alpha/(2*eta)

    if training_type == 'regression' or training_type == 'iris':
        update_input_output_volt(G_free, dataset_input_voltage, dataset_output_voltage, datastep=step)

    # solve graph free

    circuit = networks.circuit_from_graph(G_free, type='resistors') 
    analysis = ahkab.new_op()
    result = ahkab.run(circuit, an_list=analysis) #returns two arrays: resistance over time of the memristors, voltages over time in the nodes
    result = result[0]
    # print(result.keys())
    # print(f'VN0')
    voltages_free = np.array([result['op'][f'VN{node}'][0][0] for node in range(len(G_free.nodes()))])
    # print(result['op'])
    # print(voltages_free)

    G_clamped = G_free.copy(as_view=False)

    G_clamped = update_output(G_clamped, voltages_free)

    circuit = networks.circuit_from_graph(G_clamped, type='resistors') 
    analysis = ahkab.new_op()
    result = ahkab.run(circuit, an_list=analysis) #returns two arrays: resistance over time of the memristors, voltages over time in the nodes
    result = result[0]
    voltages_clamped = np.array([result['op'][f'VN{node}'][0][0] for node in G_clamped.nodes()])
    # print(voltages_clamped)


    for edge in G_free.edges():

        u, v = edge

        # print(u,v, voltages_free[u], voltages_free[v])

        diff_free = np.abs(voltages_free[int(u)] - voltages_free[int(v)])
        diff_clamped = np.abs(voltages_clamped[int(u)] - voltages_clamped[int(v)])

        delta_v = diff_clamped-diff_free
        standard_dev = 0.01
        sample = 2* np.random.rand() * standard_dev - standard_dev/2
        delta_v += sample
        # print(delta_v)
        delta_v_sq = sample*(diff_clamped**2 - diff_free**2)
        

        prefac = gamma * (1 / (G_free.edges[edge]['resistance'])**2)

        delta_R_cont = prefac * (delta_v_sq)

        G_free.edges[edge]['resistance'] += delta_R_cont

        # if delta_v>0:
        #     G_free.edges[edge]['resistance'] += 0.781
        # else:
        #     G_free.edges[edge]['resistance'] -= 0.781



        if G_free.edges[edge]['resistance'] < 1 :

            G_free.edges[edge]['resistance'] = 1

        if G_free.edges[edge]['resistance'] > 128 :

            G_free.edges[edge]['resistance'] = 128
        
    return G_free 

def update_length_cl(G_free):

    eta = 0.1
    alpha = 1000
    gamma = alpha/(2*eta)


    # solve graph free

    circuit = networks.circuit_from_graph(G_free, type='memristors') 
    analysis = ahkab.new_tran(tstart=0, tstop=0.05, tstep=1e-3, x0=None)
    result = ahkab.run(circuit, an_list=analysis) #returns two arrays: resistance over time of the memristors, voltages over time in the nodes
    result = result[0]
    voltages_free = np.array([result['tran'][f'VN{node}'][-1] for node in range(len(G_free.nodes()))])

    G_clamped = G_free.copy(as_view=False)

    G_clamped = update_output(G_clamped, voltages_free)

    circuit = networks.circuit_from_graph(G_clamped, type='memristors') 
    analysis = ahkab.new_tran(tstart=0, tstop=0.05, tstep=1e-3, x0=None)
    result = ahkab.run(circuit, an_list=analysis) #returns two arrays: resistance over time of the memristors, voltages over time in the nodes
    result = result[0]
    voltages_clamped = np.array([result['tran'][f'VN{node}'][-1] for node in G_clamped.nodes()])


    for edge in G_free.edges():

        u, v = edge

        # print(u,v, voltages_free[u], voltages_free[v])

        diff_free = np.abs(voltages_free[int(u)] - voltages_free[int(v)])
        diff_clamped = np.abs(voltages_clamped[int(u)] - voltages_clamped[int(v)])

        delta_v = diff_clamped-diff_free
        # standard_dev = 0.01
        # sample = 2* np.random.rand() * standard_dev - standard_dev/2
        # delta_v += sample
        # # print(delta_v)
        # delta_v_sq = sample*(diff_clamped**2 - diff_free**2)
        

        # prefac = gamma * (1 / (G_free.edges[edge]['resistance'])**2)
        prefac = 0.5

        delta_R_cont = prefac * (delta_v)

        G_free.edges[edge]['length'] += delta_R_cont

        # if delta_v>0:
        #     G_free.edges[edge]['resistance'] += 0.781
        # else:
        #     G_free.edges[edge]['resistance'] -= 0.781



        # if G_free.edges[edge]['resistance'] < 1 :

        #     G_free.edges[edge]['resistance'] = 1

        # if G_free.edges[edge]['resistance'] > 128 :

        #     G_free.edges[edge]['resistance'] = 128
        
    return G_free 

# Returns two arrays with length 15: input voltage and corresponding desired output following the linear relationship

def generate_dataset(training_steps, random=False):

    training_steps +=1

    if random:
        input_voltage = np.random.uniform(1,4, training_steps)
    else:
        input_voltage = np.linspace(1,4, training_steps)

    desired_output = regression_function(input_voltage)

    dataset_input = []
    dataset_output = []
    for step in range(training_steps):
        dataset_input.append([input_voltage[step]])
        dataset_output.append([desired_output[step]])

    return dataset_input, dataset_output

# FUNC----------- PARALLEL

def compute_single_gradient_parallel_helper(G, weight_index, training_type, base_error, weight_type, delta_weight, dataset_input_voltage, dataset_output_voltage, datastep, varying_len, stored_gradient, lock):
    
    # print(dataset_input_voltage[datastep], dataset_output_voltage[datastep])

    gradient = 0
    G_increment = G.copy(as_view=False)
    if weight_type=='pressure' or weight_type == 'rho':

        node = list(G.nodes)[weight_index]
        G_increment.nodes[node][f'{weight_type}'] += delta_weight

        if weight_type == 'pressure':
            denominator = delta_weight*1e5
        else:
            denominator = delta_weight

        if training_type == 'allostery':
            error = cost_function(G_increment, weight_type, varying_len=varying_len)  
        else:
            error = cost_function_regression(G_increment, weight_type, dataset_input_voltage, dataset_output_voltage, datastep)
        
        gradient = (error - base_error) / denominator

    else:
        
        edge = list(G.edges)[weight_index]
        G_increment.edges[edge][f'{weight_type}'] += delta_weight

        denominator = delta_weight
        if weight_type == 'length':
            denominator = delta_weight*1e-6
        if weight_type == 'radius_base':
            denominator = delta_weight*1e-9
        

        if training_type == 'allostery':
            error = cost_function(G_increment, weight_type, varying_len=varying_len)  
        else:
            error = cost_function_regression(G_increment, weight_type, dataset_input_voltage, dataset_output_voltage, datastep)


        gradient = (error - base_error) / denominator

    with lock:
        stored_gradient[weight_index] = gradient

    return stored_gradient
     
def to_shared_array(array):
    shared_array = mp.Array('d', array.size, lock=False)
    temp = np.frombuffer(shared_array, dtype=array.dtype)
    temp[:] = array.flatten(order = "C")
    return shared_array

def to_numpy_array(shared_array, shape):
    array = np.ctypeslib.as_array(shared_array)
    return array.reshape(shape)

def  update_weights_parallel(G, training_type, base_error, weight_type, delta_weight, learning_rate, dataset_input_voltage, dataset_output_voltage, varying_len, step):

    if weight_type=='pressure' or weight_type=='rho':
        batch_size = G.number_of_nodes()
        # batch_size = 1
        number_of_weights = G.number_of_nodes()
    else:
        # batch_size = int(G.number_of_edges()/2)
        batch_size = 7
        number_of_weights = G.number_of_edges()

    # Check if numb_nodes is a multiple of batch_size
    if number_of_weights % batch_size != 0:
        raise ValueError(f"number_of_weights ({number_of_weights}) must be a multiple of batch_size ({batch_size}).")

    # Initialize gradient vector
    init_gradient = np.zeros((number_of_weights), dtype = np.float64)  

    # Create multiprocessing array to which the different processes can access to. 
    # Thanks to temp, we can write a numpy array to a mp array and initialize it in this case.
    shared_gradient = to_shared_array(init_gradient)    #Returns initialized shared array

    # Create bridge to a nupy vector 
    stored_gradient = to_numpy_array(shared_gradient, init_gradient.shape)

    lock = mp.Lock()
    # execute in batches
    for i in range(0, number_of_weights, batch_size):
        # execute all tasks in a batch
        processes = [mp.Process(target = compute_single_gradient_parallel_helper, 
                                args=(G, p, training_type, base_error, weight_type, delta_weight, dataset_input_voltage, dataset_output_voltage, step, varying_len, stored_gradient, lock)) for p in range(i, i + batch_size)]

        # start all processes
        for process in processes:
            process.start()
        # wait for all processes to complete
        for process in processes:
            process.join()

    if weight_type == "pressure" or weight_type=='rho':

        for node in G.nodes():
            G.nodes[node][f'{weight_type}'] -= learning_rate*stored_gradient[int(node)]

    else:

        for index, edge in enumerate(G.edges()):  

            G.edges[edge][f'{weight_type}'] -= learning_rate*stored_gradient[index]
            # G.edges[edge][f'{weight_type}'] -= learning_rate*stored_gradient[index] + np.random.normal(0,0.1)
            
            # print(learning_rate*stored_gradient[index])
            if G.edges[edge][f'{weight_type}'] < 0:
                G.edges[edge][f'{weight_type}'] += learning_rate*stored_gradient[index]
                print(f"Error: Negative weight detected for edge {edge} with weight type '{weight_type}'.")

    return G

# --------- TRAINING FUNCTIONS ---------

def train(G, training_type, training_steps, weight_type, delta_weight, learning_rate, save_final_graph=False, write_weights = False, graph_id=None, constant_source=None, varying_len=False):

    # OPEN files, CREATE folders to write results
    DATA_PATH = Path(f"data/{training_type}{G.graph['name']}/")
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    DATA_PATH_MSE = DATA_PATH / "mse"
    DATA_PATH_MSE.mkdir(parents=True, exist_ok=True)

    DATA_PATH_WEIGHT = DATA_PATH / "weights" / weight_type
    DATA_PATH_WEIGHT.mkdir(parents=True, exist_ok=True)
    
    if varying_len:
        mse_file = open(f"{DATA_PATH_MSE}/mse_{weight_type}_var.txt", "w") #write error 
    else:
        mse_file = open(f"{DATA_PATH_MSE}/mse_{weight_type}.txt", "w") #write error 
    
    if G.name == 'vd': 
        potential_target_file = open(f"{DATA_PATH}/potential_targets{G.nodes[1]['desired']}.txt", "w") #write potential target during training (for voltage divider)  
    else:
        potential_target_file = None     

    if write_weights:
        # WRITE initial condition: intialized wieghts and intial error
        write_weights_to_file(G, DATA_PATH_WEIGHT, step=0, weight_type=weight_type)
    dataset_input_voltage = None
    dataset_output_voltage = None
    if training_type == 'regression':
        dataset_input_voltage, dataset_output_voltage = generate_dataset(training_steps, random=True)
        testset_input_voltage, testset_output_voltage = generate_dataset(5)

    # COMPUTE initial error
    if training_type == 'allostery':
        error = cost_function(G, weight_type, potential_target_file, update_initial_res=False, varying_len=varying_len) 
        print('Step:', 0, error)
        error_normalization = error #define it as normalization error
        mse_file.write(f"{0}\t{error/error_normalization}\n")

    elif training_type=='regression':

        error = cost_function_regression(G, weight_type, dataset_input_voltage, dataset_output_voltage, datastep=0)
        test_error = cost_function_regression(G, weight_type, testset_input_voltage, testset_output_voltage)
        print('Step:', 0, test_error)
        error_normalization = test_error #define it as normalization error
        mse_file.write(f"{0}\t{test_error/error_normalization}\n")

    # LOOP over training steps
    for step in range(training_steps): 

        # update_weights(G, training_type, error, weight_type, delta_weight, learning_rate, dataset_input_voltage, dataset_output_voltage, step, varying_len=varying_len)
        # update_weights_parallel(G, training_type, error, weight_type, delta_weight, learning_rate, dataset_input_voltage, dataset_output_voltage, varying_len=varying_len, step=step)
        # error = cost_function_regression(G, weight_type, dataset_input_voltage, dataset_output_voltage, step)
        # update_resistances(G, training_type, dataset_input_voltage, dataset_output_voltage, step=step)
        update_length_cl(G)

        if write_weights:
            write_weights_to_file(G, DATA_PATH_WEIGHT, step=step+1, weight_type=weight_type)
            
        # COMPUTE error
        if training_type == 'allostery':
            error = cost_function(G, weight_type, potential_target_file, update_initial_res=False, varying_len=varying_len)  
            print('Step:', step+1, error)
            mse_file.write(f"{step+1}\t{error/error_normalization}\n")
        elif training_type == 'regression':
            error = cost_function_regression(G, weight_type, dataset_input_voltage, dataset_output_voltage, step +1)
            test_error = cost_function_regression(G, weight_type, testset_input_voltage, testset_output_voltage)
            print('Step:', step+1, test_error)
            mse_file.write(f"{step+1}\t{test_error/error_normalization}\n")

        if save_final_graph==True and (step+1)==training_steps:
            nx.write_graphml(G, f"{DATA_PATH}/trained_graph_{weight_type}.graphml")
                            
    mse_file.close()
    
    if G.name == 'vd':
        potential_target_file.close()

def test_regression(G, step, weight_type, imposed_pressure=None, slope = None, potential=None):
        
    if slope is not None:        
        if potential is not None:
            reg_file = open(f"{par.DATA_PATH}regression{G.graph['name']}/relations_regression/relations_regression{weight_type}{step}_{slope}_{potential}.txt", "w") 
            data = np.loadtxt(f"{par.DATA_PATH}regression{G.graph['name']}/weights/{weight_type}/{weight_type}{step}_{slope}_{potential}.txt", unpack=True)
        else:
            reg_file = open(f"{par.DATA_PATH}regression{G.graph['name']}/relations_regression/relations_regression{weight_type}{step}_{slope}.txt", "w") 
            data = np.loadtxt(f"{par.DATA_PATH}regression{G.graph['name']}/weights/{weight_type}/{weight_type}{step}_{slope}.txt", unpack=True)
    else:
        reg_file = open(f"{par.DATA_PATH}regression{G.graph['name']}/relations_regression/relations_regression{weight_type}{step}.txt", "w") 
        data = np.loadtxt(f"{par.DATA_PATH}regression{G.graph['name']}/weights/{weight_type}/{weight_type}{step}.txt", unpack=True)
    weight_vec = data[1]

    if weight_type == 'pressure' or weight_type == 'rho':

        for node in G.nodes():

            G.nodes[node][f'{weight_type}'] = weight_vec[int(node)]
    else:

        for index, edge in enumerate(G.edges):
            G.edges[edge][f'{weight_type}'] = weight_vec[index]


    testset_input_voltage, testset_output_voltage = generate_dataset(15)

    error = 0
    for datastep in range(len(testset_input_voltage)):

        update_input_output_volt(G, testset_input_voltage, testset_output_voltage, datastep)

        if weight_type == 'resistance':
            circuit = networks.circuit_from_graph(G, type='resistors') 
            analysis = ahkab.new_op()
            # analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None)
        else:
            circuit = networks.circuit_from_graph(G, type='memristors', imposed_pressure=imposed_pressure) 
            analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None)

        # DEFINE a transient analysis (analysis of the circuit over time)
        result = ahkab.run(circuit, an_list=analysis) #returns two arrays: resistance over time of the memristors, voltages over time in the nodes

        result = result[0]
        
        output_voltage_target = []
        for node in G.nodes():
            
            if G.nodes[node]['type']=='target':  
                if weight_type =='resistance':
                    output_voltage_target.append(result['op'][f'VN{node}'][0][0])
                else: 
                    output_voltage_target.append(result['tran'][f'VN{node}'][-1])

        reg_file.write(f"{testset_input_voltage[datastep][0]}\t{output_voltage_target[0]}")
        reg_file.write("\n")

    test_error = cost_function_regression(G, weight_type, testset_input_voltage, testset_output_voltage)
    print(f"Error step {step}", test_error)
    reg_file.close()
    
    return
