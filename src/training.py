import parameters as par
import networks
import multiprocessing as mp
import numpy as np
import sys
import networkx as nx
from pathlib import Path
import ahkab 

# --------- WRITE RESULTS TO FILES ---------

def write_weights_to_file(G, DATA_PATH, step, weight_type, multi_train=False):

    if multi_train:
        file = open(f"{DATA_PATH}/{weight_type}{step}_target{G.nodes[1]['desired']}.txt", "w")
    else:
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
            
    if weight_type=='best_choice':
        file.write(f"{int(node)}\t{G.nodes[node]['pressure']}\t{G.nodes[node]['rho']}\n")
        
    file.close()
    
    return

# --------- UTILITIES FUNCTIONS ---------

def regression_function(input, a, b):
    return a*input + b

# --------- ALGORITHM FUNCTIONS ---------

def cost_function(G, weight_type, write_potential_target_to_file=None, update_initial_res = False):
    
    # TRANSFORM graph into circuit
    if weight_type == 'resistance':
        circuit = networks.circuit_from_graph(G, type='resistors') 
        analysis = ahkab.new_op()
    else:
        circuit = networks.circuit_from_graph(G, type='memristors') 
        analysis = ahkab.new_tran(tstart=0, tstop=0.05, tstep=1e-3, x0=None)
        

    # DEFINE a transient analysis (analysis of the circuit over time)
    result = ahkab.run(circuit, an_list=analysis) #returns two arrays: resistance over time of the memristors, voltages over time in the nodes

    resistance_vec = result[1]
    # print(1/resistance_vec[-1][0])
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

def cost_function_regression(G, weight_type, dataset_input_voltage, dataset_output_voltage, datastep=None, a=0.2, b=0.3):

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

def generate_dataset(training_steps, random=False, a=0.2, b=0.3):

    training_steps +=1

    if random:
        input_voltage = np.random.uniform(1,4, training_steps)
    else:
        input_voltage = np.linspace(1,4, training_steps)

    desired_output = regression_function(input_voltage, a=a, b=b)

    dataset_input = []
    dataset_output = []
    for step in range(training_steps):
        dataset_input.append([input_voltage[step]])
        dataset_output.append([desired_output[step]])

    return dataset_input, dataset_output

# FUNC----------- PARALLEL

def compute_single_gradient_parallel_helper(G, task_index, task, training_type, base_error, weight_type, dataset_input_voltage, dataset_output_voltage, datastep, stored_gradient, lock, a=0.2, b=0.3):
    
    obj_type, obj, attr, delta, denominator = task

    G_increment = G.copy(as_view=False)

    if obj_type == "node":
        G_increment.nodes[obj][attr] += delta
    elif obj_type == "edge":
        G_increment.edges[obj][attr] += delta

    if training_type == "allostery":
        error = cost_function(G_increment, weight_type)
    elif training_type == "regression":
        error = cost_function_regression(
            G_increment,
            weight_type,
            dataset_input_voltage,
            dataset_output_voltage,
            datastep,
            a,
            b
        )
    else:
        raise ValueError(f"Unknown training_type: {training_type}")

    gradient = (error - base_error) / denominator

    with lock:
        stored_gradient[task_index] = gradient
     
def to_shared_array(array):
    shared_array = mp.Array('d', array.size, lock=False)
    temp = np.frombuffer(shared_array, dtype=array.dtype)
    temp[:] = array.flatten(order = "C")
    return shared_array

def to_numpy_array(shared_array, shape):
    array = np.ctypeslib.as_array(shared_array)
    return array.reshape(shape)

def update_weights_parallel(G, training_type, base_error, weight_type, delta_weight, learning_rate, dataset_input_voltage, dataset_output_voltage, datastep, a=0.2, b=0.3, relative_noise=0, n_cores=1 ):

    tasks = []

    if weight_type == "pressure" or weight_type == "rho":

        for node in G.nodes():
            if weight_type == "pressure":
                denominator = delta_weight * 1e5
            else:
                denominator = delta_weight

            tasks.append(("node", node, weight_type, delta_weight, denominator))

    elif weight_type == "length_radius_base":

        for edge in G.edges():
            tasks.append(("edge", edge, "length", delta_weight[0], delta_weight[0] * 1e-6))
            tasks.append(("edge", edge, "radius_base", delta_weight[1], delta_weight[1] * 1e-9))

    elif weight_type == "length_pressure":

        for edge in G.edges():
            tasks.append(("edge", edge, "length", delta_weight[0], delta_weight[0] * 1e-6))

        for node in G.nodes():
            tasks.append(("node", node, "pressure", delta_weight[1], delta_weight[1] * 1e5))

    else:

        for edge in G.edges():
            if weight_type == "length":
                denominator = delta_weight * 1e-6
            else:
                denominator = delta_weight * 1e-9

            tasks.append(("edge", edge, weight_type, delta_weight, denominator))

    number_of_tasks = len(tasks)

    init_gradient = np.zeros(number_of_tasks, dtype=np.float64)
    shared_gradient = to_shared_array(init_gradient)
    stored_gradient = to_numpy_array(shared_gradient, init_gradient.shape)

    lock = mp.Lock()

    processes = []

    for task_index, task in enumerate(tasks):

        proc = mp.Process(
            target=compute_single_gradient_parallel_helper,
            args=(
                G, task_index, task, training_type, base_error, weight_type,
                dataset_input_voltage, dataset_output_voltage, datastep,
                stored_gradient, lock, a, b
            )
        )

        processes.append(proc)
        proc.start()

        if len(processes) >= n_cores:
            for proc in processes:
                proc.join()
            processes = []

    for proc in processes:
        proc.join()

    task_index = 0

    if weight_type == "pressure" or weight_type == "rho":

        for node in G.nodes():
            G.nodes[node][weight_type] -= learning_rate * stored_gradient[task_index]

            eta = np.random.uniform(-relative_noise, relative_noise)
            G.nodes[node][weight_type] *= (1 + eta)

            task_index += 1

    elif weight_type == "length_radius_base":

        for edge in G.edges():
            G.edges[edge]["length"] -= learning_rate[0] * stored_gradient[task_index]
            task_index += 1

            G.edges[edge]["radius_base"] -= learning_rate[1] * stored_gradient[task_index]
            task_index += 1

    elif weight_type == "length_pressure":

        for edge in G.edges():
            G.edges[edge]["length"] -= learning_rate[0] * stored_gradient[task_index]
            task_index += 1

        for node in G.nodes():
            G.nodes[node]["pressure"] -= learning_rate[1] * stored_gradient[task_index]
            task_index += 1

    else:

        for edge in G.edges():
            G.edges[edge][weight_type] -= learning_rate * stored_gradient[task_index]

            eta = np.random.uniform(-relative_noise, relative_noise)
            G.edges[edge][weight_type] *= (1 + eta)

            if G.edges[edge][weight_type] < 0:
                G.edges[edge][weight_type] += learning_rate * stored_gradient[task_index]
                raise ValueError(
                    f"Negative weight detected for edge {edge} "
                    f"with weight type '{weight_type}'."
                )

            task_index += 1

    return G

# --------- TRAINING FUNCTIONS ---------

def train(G, training_type, training_steps, weight_type, delta_weight, learning_rate, relative_noise = 0, a = 0.2, b = 0.3, save_final_graph=False, write_weights = False, constant_source=None, constant_source_node=None, multi_train=False, n_cores=1):

    # OPEN files, CREATE folders to write results
    DATA_PATH = Path(f"data/{training_type}{G.graph['name']}/")
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    DATA_PATH_MSE = DATA_PATH / "mse"
    DATA_PATH_MSE.mkdir(parents=True, exist_ok=True)

    DATA_PATH_WEIGHT = DATA_PATH / "weights" / weight_type
    DATA_PATH_WEIGHT.mkdir(parents=True, exist_ok=True)

    if relative_noise>0:
        mse_file = open(f"{DATA_PATH_MSE}/mse_{weight_type}_rn{relative_noise}.txt", "w") #write error 
    else:
        mse_file = open(f"{DATA_PATH}/mse/mse_{weight_type}.txt", "w") #write error 
    
    if multi_train: 
        DATA_PATH_MULTITARGET = DATA_PATH / "multi_target"
        DATA_PATH_MULTITARGET.mkdir(parents=True, exist_ok=True)
        potential_target_file = open(f"{DATA_PATH_MULTITARGET}/potential_targets{G.nodes[1]['desired']}.txt", "w") #write potential target during training (for voltage divider)  
        DATA_PATH_WEIGHT = DATA_PATH_MULTITARGET
    else:
        potential_target_file = None     
        
    if weight_type == 'best_choice':
        best_choice_file = open(f"{DATA_PATH_WEIGHT}/choosen_weights.txt", "w")


    if write_weights:
        # WRITE initial condition: intialized wieghts and intial error
        write_weights_to_file(G, DATA_PATH_WEIGHT, step=0, weight_type=weight_type, multi_train=multi_train)
         
    dataset_input_voltage = None
    dataset_output_voltage = None
    if training_type == 'regression':
        # impose contant voltage source for enabling biasing
        G.nodes[constant_source_node]['voltage'] = constant_source

        dataset_input_voltage, dataset_output_voltage = generate_dataset(training_steps, random=True, a=a, b=b)
        testset_input_voltage, testset_output_voltage = generate_dataset(5, a=a, b=b)

    # COMPUTE initial error
    if training_type == 'allostery':
        error = cost_function(G, weight_type, potential_target_file, update_initial_res=False) 
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

        if weight_type == 'best_choice':
            G_matrices = {}
            cost_func_vec = np.zeros(4)
            possible_weights = ['length', 'radius_base', 'rho', 'pressure']
            for weight_possible_inx in range(len(possible_weights)):

                G_matrices[f'{possible_weights[weight_possible_inx]}'] = G.copy(as_view=False)

                weight_type_step = possible_weights[weight_possible_inx]
                delta_weight_step = delta_weight[weight_possible_inx]
                learning_rate_step = learning_rate[weight_possible_inx]

                if training_type == 'voltage_divider' or training_type == 'regression': 
                    G_matrices[f'{possible_weights[weight_possible_inx]}'].nodes['3']['voltage'] = constant_source[weight_possible_inx]

                if training_type=='regression':
                    initial_error = cost_function_regression(G_matrices[f'{possible_weights[weight_possible_inx]}'], weight_type_step, testset_input_voltage, testset_output_voltage)
                    error = cost_function_regression(G_matrices[f'{possible_weights[weight_possible_inx]}'], weight_type, dataset_input_voltage, dataset_output_voltage, datastep=step)
                else:
                    initial_error = cost_function(G_matrices[f'{possible_weights[weight_possible_inx]}'], weight_type_step)

                update_weights_parallel(G_matrices[f'{possible_weights[weight_possible_inx]}'], training_type, error, weight_type_step, delta_weight_step, learning_rate_step, dataset_input_voltage, dataset_output_voltage, step=step)

                if training_type=='regression':
                    after_update_error = cost_function_regression(G_matrices[f'{possible_weights[weight_possible_inx]}'], weight_type_step, testset_input_voltage, testset_output_voltage)
                else:
                    after_update_error = cost_function(G_matrices[f'{possible_weights[weight_possible_inx]}'], weight_type_step)

                # print(possible_weights[weight_possible_inx], initial_error, after_update_error, after_update_error/initial_error)
                cost_func_vec[weight_possible_inx] = after_update_error
            
            # choosen_weight = np.argmin(cost_func_vec)
            choosen_weight = np.argmin(cost_func_vec/initial_error)
            print(choosen_weight)
            best_choice_file.write(f"{step+1}\t{choosen_weight}\n")
            G = G_matrices[f'{possible_weights[choosen_weight]}'].copy(as_view=False)
        else:
           
            update_weights_parallel(G, training_type, error, weight_type, delta_weight, learning_rate, dataset_input_voltage, dataset_output_voltage, datastep=step, relative_noise=relative_noise, n_cores = n_cores)
            
        if write_weights:
            write_weights_to_file(G, DATA_PATH_WEIGHT, step=step+1, weight_type=weight_type, multi_train=multi_train)
            
        # COMPUTE error
        if training_type == 'allostery':
            error = cost_function(G, weight_type, potential_target_file, update_initial_res=False)  
            print('Step:', step+1, error)
            mse_file.write(f"{step+1}\t{error/error_normalization}\n")
        elif training_type == 'regression':
            error = cost_function_regression(G, weight_type, dataset_input_voltage, dataset_output_voltage, step +1)
            test_error = cost_function_regression(G, weight_type, testset_input_voltage, testset_output_voltage)
            print('Step:', step+1, test_error)
            mse_file.write(f"{step+1}\t{test_error/error_normalization}\n")

        if save_final_graph==True and (step+1)==training_steps:
            nx.write_graphml(G, f"{DATA_PATH}/trained_graph_{weight_type}.graphml")
            
    if weight_type == 'best_choice':
        best_choice_file.close()                   
    mse_file.close()
    
    if multi_train:
        potential_target_file.close()

def test_regression(G, constant_source, constant_source_node, step, weight_type,  retrain=False, a=0.2, b=0.3):
        
    DATA_PATH = Path(f"data/regression{G.graph['name']}/relations_regression/")
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    if retrain:
        reg_file = open(f"{DATA_PATH}/relations_regression{weight_type}{step}_retrain.txt", "w") 
    else:   
        reg_file = open(f"{DATA_PATH}/relations_regression{weight_type}{step}.txt", "w") 

    DATA_PATH_W = Path(f"data/regression{G.graph['name']}/weights/{weight_type}/")
    data = np.loadtxt(f"{DATA_PATH_W}/{weight_type}{step}.txt", unpack=True)
    weight_vec = data[1]

    if weight_type == 'pressure' or weight_type == 'rho':

        for node in G.nodes():

            G.nodes[node][f'{weight_type}'] = weight_vec[int(node)]
    else:

        for index, edge in enumerate(G.edges):
            G.edges[edge][f'{weight_type}'] = weight_vec[index]

    # impose contant voltage source for enabling biasing 
    G.nodes[constant_source_node]['voltage'] = constant_source

    testset_input_voltage, testset_output_voltage = generate_dataset(15, a=a, b=b)

    error = 0
    for datastep in range(len(testset_input_voltage)):

        update_input_output_volt(G, testset_input_voltage, testset_output_voltage, datastep)

        if weight_type == 'resistance':
            circuit = networks.circuit_from_graph(G, type='resistors') 
            analysis = ahkab.new_op()
            # analysis = ahkab.new_tran(tstart=0, tstop=0.1, tstep=1e-3, x0=None)
        else:
            circuit = networks.circuit_from_graph(G, type='memristors') 
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

def apply_weight_noise(G, weight_type, relative_noise):
    Gp = G.copy(as_view=False)

    if weight_type in ["pressure", "rho"]:
        for node in Gp.nodes():
            w = Gp.nodes[node][weight_type]
            eps = np.random.uniform(-relative_noise, relative_noise)
            Gp.nodes[node][weight_type] = w * (1 + eps)
    else:
        for edge in Gp.edges():
            w = Gp.edges[edge][weight_type]
            eps = np.random.uniform(-relative_noise, relative_noise)
            Gp.edges[edge][weight_type] = w * (1 + eps)

    return Gp

from tqdm import tqdm
def robustness_trained(G, weight_type, training_type, step, relative_noise, n_samples=500):
    DATA_PATH = f"data/{training_type}{G.graph['name']}/"

    # Load trained weights
    data = np.loadtxt( f"{DATA_PATH}weights/{weight_type}/{weight_type}{step}.txt", unpack=True )
    weight_vec = data[1]

    G_opt = G.copy(as_view=False)

    if weight_type in ["pressure", "rho"]:
        for node in G_opt.nodes():
            G_opt.nodes[node][weight_type] = weight_vec[int(node)]
    else:
        for index, edge in enumerate(G_opt.edges()):
            G_opt.edges[edge][weight_type] = weight_vec[index]

    if training_type == 'regression':
        # Fixed test set
        testset_input_voltage, testset_output_voltage = generate_dataset(20)
        baseline_cost = cost_function_regression(G_opt, weight_type, testset_input_voltage, testset_output_voltage)
    else:
        baseline_cost = cost_function(G_opt, weight_type)

    perturbed_costs = []

    for _ in tqdm(range(n_samples)):
        G_noisy = apply_weight_noise(G_opt, weight_type, relative_noise)

        if training_type == 'regression':
            cost = cost_function_regression(G_noisy, weight_type, testset_input_voltage, testset_output_voltage)
        else:
            cost = cost_function(G_noisy, weight_type)

        perturbed_costs.append(cost)

    perturbed_costs = np.array(perturbed_costs)

    data = {'baseline_cost': baseline_cost, 'perturbed_costs': perturbed_costs}
    np.save(f"{DATA_PATH}mse/robustness_mse_{weight_type}.npy", data)