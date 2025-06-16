import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network 

import json
from itertools import product
from matplotlib import rc
import style
rc('font',**style.fontstyle)
rc('text', usetex=True)

trained_version_start, trained_version_end = 0, 20
training_version_list = list(range(trained_version_start, trained_version_end))
# plot params
MC_runs = int(5e3-1)
nodes_list = [4]
t_cut_list = [2] 
plist_arr =  [[0.8, 0.5, 0.8], [0.5, 0.8, 0.5]] #[[0.9, 0.6, 0.9], [0.6, 0.9, 0.6], [0.8, 0.5, 0.8], [0.5, 0.8, 0.5]]
p_slist_arr = [[1, 1], [0.5, 1], [1, 0.5]]

def load_RL_data():
    current_path = os.getcwd()
    save_path = os.path.join(current_path, '../..', 'data/global_agent_swap_sumInhom')

    # initialise the data dictionary
    data = {} 
    for el in product(*[nodes_list, t_cut_list, plist_arr, p_slist_arr, training_version_list]):
        el = json.dumps(el) # turn params to string for dict key
        data[el] = -1*np.ones(MC_runs)
        
    # loading the data and inserting it into the data dictionary 
    for d, p_slist in enumerate(p_slist_arr):
        p_s_str = "_".join(f"{val:.02f}" for val in p_slist_arr[d]) # turn params to string for save directory name
        for c, plist in enumerate(plist_arr):
            p_str = "_".join(f"{val:.02f}" for val in plist_arr[c]) # turn params to string for save directory name
            for b, t_cut in enumerate(t_cut_list):
                for a, nodes in enumerate(nodes_list):
                    for trained_version in range(trained_version_start, trained_version_end):
                        t_cut_cc = t_cut*Quantum_network(nodes=nodes, t_cut=t_cut, p_slist=p_slist, p_elist=plist).t_cut_cc_multiplier()
                        
                        delivery_times = [] # the delivery for one combination of parameters
                        # There are multiple versions of the simulation data, for each time I ran the simulation
                        version = 0
                        # Construct full path to relevant dataset
                        sim_dat_path = save_path+f'/env_cc_a_alt_o_hist/sim_datInd{trained_version}/sim_dat_cc_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p_str}_p_s_{p_s_str}_v{version}.pkl'
                        if os.path.exists(sim_dat_path) == True:# If it exists...
                            if version == 0:
                                assert os.path.exists(sim_dat_path) == True, f"no simulation data found for n_{nodes}_t_cut_{t_cut_cc}_p_{p_str}_p_s_{p_s_str}_v{version}"
                            while os.path.exists(sim_dat_path) == True: # read the file
                                with open(sim_dat_path, "rb") as file: # Unpickling
                                    sim_dat = pickle.load(file)
                                
                                delivery_times += sim_dat[0].tolist() # First index to get the regular time steps

                                version += 1

                                # Go to next version
                                sim_dat_path = save_path+f'/env_cc_a_alt_o_hist/sim_datInd{trained_version}/sim_dat_cc_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p_str}_p_s_{p_s_str}_v{version}.pkl'
                        else: # when there's not simulation for these parameters yet
                            print(f'no simulation data found for n_{nodes}_t_cut_{t_cut_cc}_p_{p_str}_p_s_{p_s_str} at training version {trained_version}')
                            max_steps = int(1e6)
                            delivery_times = max_steps*np.ones(MC_runs)

                        el = json.dumps((nodes, t_cut, plist, p_slist, trained_version)) # turn params to string for dict key
                        data[el] = np.array(delivery_times[:MC_runs])

                        print(f'n_{nodes}_t_cut_{t_cut_cc}_p_{p_str}_p_s_{p_s_str} at training version {trained_version}')
                        print(f'lenght dat = {len(delivery_times[:MC_runs])}')
                        assert len(delivery_times[:MC_runs]) == MC_runs

    return data

def load_swap_asap_data(setting):
    current_path = os.getcwd()
    save_path = os.path.join(current_path, '../..', 'data/global_agent_swap_sumInhom')

    # initialise the data dictionary
    data = {} 
    for el in product(*[nodes_list, t_cut_list, plist_arr, p_slist_arr]):
        el = json.dumps(el) # turn params to string for dict key
        data[el] = -1*np.ones(MC_runs)
        
    # loading the data and inserting it into the data dictionary 
    for d, p_slist in enumerate(p_slist_arr):
        p_s_str = "_".join(f"{val:.02f}" for val in p_slist_arr[d]) # turn params to string for save directory name
        for c, plist in enumerate(plist_arr):
            p_str = "_".join(f"{val:.02f}" for val in plist_arr[c]) # turn params to string for save directory name
            for b, t_cut in enumerate(t_cut_list):
                for a, nodes in enumerate(nodes_list):
                    t_cut_cc = t_cut
                    if setting == "swap-asap (predictive)" or setting == 'swap-asap (no cc effects)' or setting == "random":
                        t_cut_cc = t_cut*Quantum_network(nodes=nodes, t_cut=t_cut, p_slist=p_slist, p_elist=plist).t_cut_cc_multiplier()

                    delivery_times = [] # the delivery for one combination of parameters
                    # There are multiple versions of the simulation data, for each time I ran the simulation
                    version = 0
                    if setting == 'swap-asap (cc effects)':
                        sim_dat_path = save_path+f'/swap_asap_cc/sim_dat/sim_dat_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p_str}_p_s_{p_s_str}_v{version}.pkl'
                    elif setting == 'swap-asap (no cc effects)':
                        sim_dat_path = save_path+f'/swap_asap/sim_dat/sim_dat_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p_str}_p_s_{p_s_str}_v{version}.pkl'
                    elif setting == 'swap-asap (predictive)':
                        sim_dat_path = save_path+f'/swap_asap_with_pred/sim_dat/sim_dat_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p_str}_p_s_{p_s_str}_v{version}.pkl'

                    if setting == "swap-asap (predictive)" and version == 0:
                        assert os.path.exists(sim_dat_path) == True
                    while os.path.exists(sim_dat_path) == True:
                        with open(sim_dat_path, "rb") as file: # Unpickling
                            sim_dat = pickle.load(file)
                        delivery_times += sim_dat[0].tolist()
                        version += 1
                        # preparing the path to load the next version
                        if setting == 'swap-asap (cc effects)':
                            sim_dat_path = save_path+f'/swap_asap_cc/sim_dat/sim_dat_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p_str}_p_s_{p_s_str}_v{version}.pkl'
                        elif setting == 'swap-asap (no cc effects)':
                            sim_dat_path = save_path+f'/swap_asap/sim_dat/sim_dat_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p_str}_p_s_{p_s_str}_v{version}.pkl'
                        elif setting == 'swap-asap (predictive)':
                            sim_dat_path = save_path+f'/swap_asap_with_pred/sim_dat/sim_dat_'+f'n_{nodes}_t_cut_{t_cut_cc}_p_{p_str}_p_s_{p_s_str}_v{version}.pkl'

                    print(f'setting {setting}')
                    print(f' n {nodes}, t_cut {t_cut_cc}, p {p_str}, p_s {p_s_str} version = {version}')

                    el = json.dumps((nodes, t_cut, plist, p_slist)) # turn params to string for dict key
                    data[el] = np.array(delivery_times[:MC_runs])

                    assert np.shape(delivery_times[:MC_runs])[0] == MC_runs, f'not enough {setting} sim dat at n {nodes}, t_cut {t_cut_cc}, p {p_str}, p_s {p_s_str}, data shape {len(delivery_times[:MC_runs])}, location {sim_dat_path}'

    return data

if __name__ == "__main__":
    current_path = os.getcwd()
    data_path = os.path.join(current_path, '../..', 'data')

    # Load all the data
    data_RL = load_RL_data()
    data_swap_asap = load_swap_asap_data('swap-asap (cc effects)')
    data_swap_asap_no_cc = load_swap_asap_data('swap-asap (no cc effects)')
    data_swap_asap_pred = load_swap_asap_data('swap-asap (predictive)')

    param_list = []
    T_swap_asap_no_cc = []
    T_swap_asap = []
    T_swap_asap_pred = []
    T_RL = []

    std_T_swap_asap_no_cc = []
    std_T_swap_asap = []
    std_T_swap_asap_pred = []
    std_T_RL = []

    for params in product(*[nodes_list, t_cut_list, plist_arr, p_slist_arr]):
        params = json.dumps(params) # turn params to string for dict key
        param_list.append(params)

        # swap asap no cc
        expected_T = np.average(data_swap_asap_no_cc[params], axis=0)
        expected_T = np.where(expected_T >= 5*1e4, np.nan, expected_T)
        T_swap_asap_no_cc.append(expected_T)
        std_T_swap_asap_no_cc.append(np.std(data_swap_asap_no_cc[params]/np.sqrt(MC_runs), axis=0))

        # naive swap asap with cc
        expected_T = np.average(data_swap_asap[params], axis=0)
        expected_T = np.where(expected_T >= 5*1e4, np.nan, expected_T)
        T_swap_asap.append(expected_T)
        std_T_swap_asap.append(np.std(data_swap_asap[params]/np.sqrt(MC_runs), axis=0))

        # predictive swap asap with cc
        expected_T = np.average(data_swap_asap_pred[params], axis=0)
        expected_T = np.where(expected_T >= 5*1e4, np.nan, expected_T)
        T_swap_asap_pred.append(expected_T)
        std_T_swap_asap_pred.append(np.std(data_swap_asap_pred[params]/np.sqrt(MC_runs), axis=0))

        # RL
        expected_T_all_version = []
        best_expected_T = np.inf
        for trained_version in range(trained_version_start, trained_version_end): # pick the version with the lowest expected time
            paramsWversion = json.dumps(json.loads(params) + [trained_version])
            expected_T = np.average(data_RL[paramsWversion], axis=0)
            expected_T = np.where(expected_T >= 5*1e4, np.nan, expected_T)
            expected_T_all_version.append(expected_T)
            # to find the best version of the agent
            if expected_T< best_expected_T:
                best_expected_T = expected_T
                paramsWbestversion = paramsWversion
        expected_T = np.nanmin(expected_T_all_version, axis=0)
        T_RL.append(expected_T)
        # idx_max_T = np.argmax(-1*np.array(expected_T_all_version), axis=0)
        # best_version = trained_version_start + int(idx_max_T)
        # paramsWbestversion = json.dumps(json.loads(params) + [best_version])
        print('paramsWbestversion =', paramsWbestversion)
        std_T_RL.append(np.std(data_RL[paramsWbestversion]/np.sqrt(MC_runs), axis=0))

    #################
    # Plotting
    ################
    x = np.arange(len(param_list))
    width = 0.2
    plist_symbols   = {0: 'a',  1: 'b',  2: 'c',  3: 'd'}
    # making nice labels for the x axis
    p_slist_symbols = {    0: r'$\alpha$', 1: r'$\beta$', 2: r'$\gamma$'}  
    nice_labels = []
    for params_json in param_list:
        nodes, t_cut, plist, p_slist = json.loads(params_json)
        plist_idx   = next(i for i, p in enumerate(plist_arr)   if p == plist)
        p_slist_idx = next(i for i, p in enumerate(p_slist_arr) if p == p_slist)
        tup   = (plist_symbols[plist_idx], p_slist_symbols[p_slist_idx])  
        label = f"({tup[0]}, {tup[1]})"
        nice_labels.append(label)

    fig, ax = plt.subplots()
    ax.bar(x - width, T_swap_asap, width, label='WB swap-asap', color='#338000ff')  
    ax.bar(x, T_swap_asap_pred, width, label='Predictive swap-asap', color='#2c5aa0ff')         
    ax.bar(x + width, T_RL, width, label='RL agent', color='#d38d5fff')       
    ax.set_xticks(x)
    ax.set_xticklabels(nice_labels)
    ax.set_ylabel('Expected delivery time')
    # ax.set_title('Performance comparison of different strategies')
    ax.legend()
    save_directory = data_path+f'/figures/figInhom'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory) 
    plt.savefig(save_directory+'/figInhom.pdf',dpi=1200)
    plt.savefig(save_directory+'/figInhom.jpg',dpi=1200)

    #################
    # Plotting with Error bars
    #################
    x = np.arange(len(param_list))
    width = 0.2
    plist_symbols   = {0: 'a',  1: 'b',  2: 'c',  3: 'd'}
    # making nice labels for the x axis
    p_slist_symbols = {    0: r'$\alpha$', 1: r'$\beta$', 2: r'$\gamma$'}  
    nice_labels = []
    for params_json in param_list:
        nodes, t_cut, plist, p_slist = json.loads(params_json)
        plist_idx   = next(i for i, p in enumerate(plist_arr)   if p == plist)
        p_slist_idx = next(i for i, p in enumerate(p_slist_arr) if p == p_slist)
        tup   = (plist_symbols[plist_idx], p_slist_symbols[p_slist_idx])  
        label = f"({tup[0]}, {tup[1]})"
        nice_labels.append(label)

    fig, ax = plt.subplots()
    ax.bar(x - width, T_swap_asap, width, yerr=std_T_swap_asap, error_kw={
           'capsize': 5,      # size of the caps in points
           'capthick': 2,     # thickness of the cap lines
           'elinewidth': 1.5, # thickness of the error bar line itself
           'ecolor': 'black'  # color of the error bars
       }, label='WB swap-asap', color='#338000ff')  
    ax.bar(x, T_swap_asap_pred, width, yerr=std_T_swap_asap_pred, error_kw={
           'capsize': 5,      # size of the caps in points
           'capthick': 2,     # thickness of the cap lines
           'elinewidth': 1.5, # thickness of the error bar line itself
           'ecolor': 'black'  # color of the error bars
       }, label='Predictive swap-asap', color='#2c5aa0ff')         
    ax.bar(x + width, T_RL, width, yerr=std_T_RL, error_kw={
           'capsize': 5,      # size of the caps in points
           'capthick': 2,     # thickness of the cap lines
           'elinewidth': 1.5, # thickness of the error bar line itself
           'ecolor': 'black'  # color of the error bars
       }, label='RL agent', color='#d38d5fff')       
    ax.set_xticks(x)
    ax.set_xticklabels(nice_labels)
    ax.set_ylabel('Expected delivery time')
    # ax.set_title('Performance comparison of different strategies')
    ax.legend()
    save_directory = data_path+f'/figures/figInhom'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory) 
    plt.savefig(save_directory+'/figInhomwErr.pdf',dpi=1200)
    plt.savefig(save_directory+'/figInhomwErr.jpg',dpi=1200)



