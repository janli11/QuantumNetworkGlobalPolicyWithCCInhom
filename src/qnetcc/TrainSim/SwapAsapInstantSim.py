import os

from qnetcc.environments.QNSimulator.QuantumNetwork import Quantum_network 
import numpy as np
import pickle
import tqdm
import time
import random

class swap_asap_simulation(object):
    def __init__(self, nodes, t_cut, plist, p_slist, simulation_eps, time_out_mult=2,
                 cluster = 0):

        # network params
        self.nodes = nodes
        self.t_cut = t_cut
        self.plist = plist
        self.p_slist = p_slist
        self.p_str = "_".join(f"{val:.02f}" for val in plist)
        self.p_s_str = "_".join(f"{val:.02f}" for val in p_slist)

        # simulation params
        self.sim_eps = simulation_eps
        self.time_out_mult = time_out_mult
        self.max_steps = int(1e6)

        # paths
        self.abs_path = os.path.abspath(os.getcwd())
        self.save_path = os.path.join(self.abs_path, '../../../..', 'data/global_agent_swap_sumInhom')# path for saving simulation data, Trained models, etc. 
        self.cluster = cluster
        if self.cluster == 1:
            self.abs_path = os.path.abspath(os.getcwd())
            self.save_path = '/home/lijt/data1/quantumNetworkInhom'        
        self.sub_proj_rel_path = '/swap_asap'
        self.data_folder = self.save_path+self.sub_proj_rel_path+f'/sim_dat'
        self.file_name_template = f'/sim_dat_'+f'n_{self.nodes}_t_cut_{self.t_cut}_p_{self.p_str}_p_s_{self.p_s_str}'
        self.sim_dat_path_template = self.data_folder+self.file_name_template

    def simulate_policy(self):
        """simulating the swap asap policy

        Returns:
            _type_: _description_
        """
        T_list = []
        micro_T_list = []

        start = time.time()
        for run in tqdm.tqdm(range(self.sim_eps),leave=False):
            quantum_network = Quantum_network(self.nodes,self.t_cut,self.plist,self.p_slist)
            while not quantum_network.A_B_entangled():
                node_actions = quantum_network.instant_comm_swap_asap_actions()
                quantum_network.local_actions_update_network(node_actions)
                quantum_network.update_time_slots()
                stop = time.time()
                # don't allow entire simulation to take longer than certain amount of time
                if (stop-start) > self.sim_eps*self.time_out_mult:
                    break
            # if simulations takes to long, just make every episode length equal to max steps
            if (stop-start) > self.sim_eps*self.time_out_mult:
                T_list = [self.max_steps for i in range(self.sim_eps)]
                micro_T_list = [2*self.max_steps for i in range(self.sim_eps)]
                break

            T_list.append(quantum_network.time_slot)
            micro_T_list.append(quantum_network.micro_time_slot)
        self.save_and_load_simulation_data(T_list, micro_T_list)

        # saving the delivery times (at different time scales) with their std

        average_regular_time = np.average(T_list)
        std_regular_time = np.std(T_list)/np.sqrt(self.sim_eps)
        print(f' average regular time = {average_regular_time} with std {std_regular_time}')
        average_micro_time = np.average(micro_T_list)
        std_micro_time = np.std(micro_T_list)/np.sqrt(self.sim_eps)
        print(f' average micro time = {average_micro_time} with std {std_micro_time}')

    def simulate_policy_w_print(self, seed = None):
        """simulating the swap asap policy

        Returns:
            _type_: _description_
        """
        T_list = []
        micro_T_list = []

        # setting the random seed to same eps each time
        random.seed(seed)
        np.random.seed(seed)

        start = time.time()
        for run in tqdm.tqdm(range(self.sim_eps),leave=False):
            quantum_network = Quantum_network(self.nodes,self.t_cut,self.plist,self.p_slist)
            while not quantum_network.A_B_entangled():
                node_actions = quantum_network.instant_comm_swap_asap_actions()

                # priting after actions are selected but before they are applied
                print('--------------------------------------------------')
                print('---printing before: quantum_network.local_actions_update_network(node_actions)---')
                print(f'time step {quantum_network.time_slot}')
                print(f'action step {quantum_network.micro_time_slot%2}')
                print(f'actions: {node_actions}')
                print(f'actual state: {quantum_network.get_link_config()}')
                print(f'state end to end: {quantum_network.A_B_entangled()}')

                quantum_network.local_actions_update_network(node_actions)
                quantum_network.update_time_slots()
                stop = time.time()
                # don't allow entire simulation to take longer than certain amount of time
                if (stop-start) > self.sim_eps*self.time_out_mult:
                    break
            # if simulations takes to long, just make every episode length equal to max steps
            if (stop-start) > self.sim_eps*self.time_out_mult:
                T_list = [self.max_steps for i in range(self.sim_eps)]
                micro_T_list = [2*self.max_steps for i in range(self.sim_eps)]
                break

            # priting after actions are selected but before they are applied
            print('---------printing one more time after end-to-end has been reached----------')
            print('---------this micro time slot does not contribute to quantum_network.micro_time_slot----------')
            print(f'time step {quantum_network.time_slot}')
            print(f'action step {quantum_network.micro_time_slot%2}')
            print(f'actions: {node_actions}')
            print(f'actual state: {quantum_network.get_link_config()}')
            print(f'state end to end: {quantum_network.A_B_entangled()}')

            T_list.append(quantum_network.time_slot)
            micro_T_list.append(quantum_network.micro_time_slot)
        self.save_and_load_simulation_data(T_list, micro_T_list)

        # saving the delivery times (at different time scales) with their std

        average_regular_time = np.average(T_list)
        std_regular_time = np.std(T_list)/np.sqrt(self.sim_eps)
        print(f' average regular time = {average_regular_time} with std {std_regular_time}')
        average_micro_time = np.average(micro_T_list)
        std_micro_time = np.std(micro_T_list)/np.sqrt(self.sim_eps)
        print(f' average micro time = {average_micro_time} with std {std_micro_time}')
        
    def save_and_load_simulation_data(self, T_list, micro_T_list):
        """saving the simulated delivery times
        """

        version = 0                 
        sim_dat_path = self.sim_dat_path_template+f'_v{version}.pkl'
        sim_dat = np.array([T_list, micro_T_list])
        if os.path.exists(self.data_folder) == False:
            os.makedirs(self.data_folder)
        while os.path.exists(sim_dat_path) == True:
            version += 1
            sim_dat_path = self.sim_dat_path_template+f'_v{version}.pkl'
        with open(sim_dat_path, "wb") as file:   # Pickling
            pickle.dump(sim_dat, file)
        with open(sim_dat_path, "rb") as file:   # Unpickling
            sim_dat = pickle.load(file)

if __name__ == "__main__":
    nodes, t_cut, plist, p_slist, simulation_eps = 4, 2, 1, 1, int(1)
    swap_sim = swap_asap_simulation(nodes, t_cut, plist, p_slist, simulation_eps).simulate_policy_w_print()