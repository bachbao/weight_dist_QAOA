import copy
import math
import random
import pickle
import pennylane as qml
import networkit as nk 
import pynauty as nauty
from pennylane import numpy as np
from networkit.algebraic import adjacencyMatrix

QAOA_type = ['std', 
             'ma', 
             'orbit',
             'naive_wd', 
             'eigenbased_wd',
             'normalized_eigenbased_wd',
             'algebraic_dis']

def construct_hamiltonian(graph: nk.Graph):
    H_coeffs = []
    H_obs = []
    for u, v, w in graph.iterEdgesWeights():
        H_coeffs.append(-w/2)
        H_coeffs.append(w/2)
        H_obs.append(qml.Identity(u) @ qml.Identity(v))
        H_obs.append(qml.PauliZ(u) @ qml.PauliZ(v))
    hamiltonian = qml.Hamiltonian(H_coeffs, H_obs)
    return hamiltonian
    
class EigenBasedWeightedQAOA:
    def __init__(self, 
                 graph: nk.Graph,
                 num_layers: int,
                 qaoa_type: str, 
                 initial_betas: np.array = None,
                 initial_gammas: np.array = None,
                 seed: int = 42):
        self.graph = graph
        self.n_wires = self.graph.numberOfNodes()
        self.num_layers = num_layers
        self.cost_hamiltonian = construct_hamiltonian(self.graph) #qml.qaoa.maxcut(nx_graph)
        #print(self.cost_hamiltonian)
        assert qaoa_type in QAOA_type, "QAOA type is not supported!"
        self.qaoa_type = qaoa_type
        self.fiedler_vector = None
        self.eigenbased_graph = None
        self.weight_distribution = None
        self.rng = np.random.default_rng(seed=seed)
        if self.qaoa_type in ['eigenbased_wd', 'normalized_eigenbased_wd']:
            self.eigenbased_weighted_degree()
        elif self.qaoa_type == 'orbit':
            self.construct_orbit_order()
        elif self.qaoa_type == 'algebraic_dis':
            self.construct_algebraic_distance()
            
        if initial_betas is None and initial_gammas is None:
            self.setup_initial_param()
        else:
            self.beta_params = initial_betas
            self.gamma_params = initial_gammas
            
    def construct_algebraic_distance(self, 
                                     omega: float = 0.5, 
                                     n_iterations: int = 25,
                                     n_initializations: int = 10,
                                     p_norm: int = 2):
        # AD calculation
        Sij_over_k_iterations = []
        x_over_k_iterations = []
        for _ in range(n_initializations):
            x = np.random.uniform(size=self.n_wires, requires_grad=False)
            #print(x)
            for k in range(n_iterations):
                x_tilde = np.array([np.sum([self.graph.weight(i, j)* x[j] for j in self.graph.iterNeighbors(i)])/np.sum([self.graph.weight(i, j) for j in self.graph.iterNeighbors(i)]) for i in range(self.n_wires)])
                x = (1-omega) * x_tilde + omega * x
            x_over_k_iterations.append(x)
            Sij = []
            for u, v in self.graph.iterEdges():
                Sij.append(np.abs(x[u] - x[v])**p_norm)
            Sij_over_k_iterations.append(Sij)
        Sij_over_k_iterations = np.array(Sij_over_k_iterations)
        x_over_k_iterations = np.array(x_over_k_iterations)
        algebraic_distance = []
        for index in range(self.graph.numberOfEdges()):
            algebraic_distance.append(np.sum(Sij_over_k_iterations[:,index])**(1/p_norm))
        self.algebraic_distance = np.round(1/np.array(algebraic_distance))
        
        # Start the grouping nodes
        x_avg_value = [np.round(np.average(x_over_k_iterations[:,q])) for q in range(self.n_wires)]
        self.beta_orbit_indices = []
        self.unique_orbit = []
        unique_idx = 0
        for index in range(self.n_wires):
            if x_avg_value[index] not in self.unique_orbit:
                self.unique_orbit.append(x_avg_value[index])
                self.beta_orbit_indices.append(unique_idx)
                unique_idx += 1
            else:
                self.beta_orbit_indices.append(self.unique_orbit.index(x_avg_value[index]))
        assert len(self.beta_orbit_indices) == self.n_wires
        # Start the grouping edges
        self.AD_uniqueness = []
        self.AD_unique_order = []
        unique_idx = 0
        for index in range(self.graph.numberOfEdges()):
            if self.algebraic_distance[index] not in self.AD_uniqueness:
                self.AD_uniqueness.append(self.algebraic_distance[index])
                self.AD_unique_order.append(unique_idx)
                unique_idx += 1
            else:
                self.AD_unique_order.append(self.AD_uniqueness.index(self.algebraic_distance[index]))
        assert len(self.AD_unique_order) == self.graph.numberOfEdges()
    
    def construct_orbit_order(self):
        nauty_G = nauty.Graph(self.n_wires)
        for node in range(self.n_wires):
            node_neighbors = []
            for neighbor in self.graph.iterNeighbors(node):
                node_neighbors.append(neighbor)
            nauty_G.connect_vertex(node, node_neighbors)
        _, _, _, self.orbit, self.num_orbits = nauty.autgrp(nauty_G)
        self.beta_orbit_indices = list(set(self.orbit))
        assert len(self.beta_orbit_indices) == self.num_orbits
        # self.vertices_orbit = {}
        # for orbit in range(self.num_orbits):
        #     self.vertices_orbit[orbit] = []
        # for node in range(self.n_wires):
        #     self.vertices_orbit[self.orbit[node]].append(node)
        self.edges_orbit = []
        for u, v in self.graph.iterEdges():
            edge_orbit_order = [self.orbit[u], self.orbit[v]]
            edge_orbit_order.sort()
            if edge_orbit_order not in self.edges_orbit:
                self.edges_orbit.append(edge_orbit_order)
                
    def eigenbased_weighted_degree(self):
        eigenvals, eigenvectors = nk.algebraic.laplacianEigenvectors(self.graph, cutoff=1, reverse=True)
        self.fiedler_vector = eigenvectors[-1]
        rounding_fiedler_vector = np.round(self.fiedler_vector, 2)
        self.beta_orbit_indices = []
        self.unique_orbit = []
        unique_idx = 0
        for index in range(self.n_wires):
            if rounding_fiedler_vector[index] not in self.unique_orbit:
                self.unique_orbit.append(rounding_fiedler_vector[index])
                self.beta_orbit_indices.append(unique_idx)
                unique_idx += 1
            else:
                self.beta_orbit_indices.append(self.unique_orbit.index(rounding_fiedler_vector[index]))
        assert len(self.beta_orbit_indices) == self.n_wires
        
        self.eigenbased_graph = nk.Graph(n=self.n_wires, weighted=True, directed=False)
        total_connectivity = 0
        for u, v in self.graph.iterEdges():
            if np.abs(self.fiedler_vector[u] - self.fiedler_vector[v]) == 0:
                weight = 0
            else:
                weight = np.round(1/np.abs(self.fiedler_vector[u] - self.fiedler_vector[v])**2)
            total_connectivity += weight
            self.eigenbased_graph.addEdge(u, v, w = weight)
        if self.qaoa_type == "normalized_eigenbased_wd":
            max_value = 0
            for u, v, w in self.eigenbased_graph.iterEdgesWeights():
                self.eigenbased_graph.setWeight(u, v, np.round(total_connectivity/w, 2))
                if np.round(total_connectivity/w, 2) > max_value:
                    max_value = np.round(total_connectivity/w, 2)
            divider = math.floor(math.log10(max_value))
            for u, v, w in self.eigenbased_graph.iterEdgesWeights():
                self.eigenbased_graph.setWeight(u, v, w//(10**divider))
            
    def setup_initial_param(self, 
                            scaling_constant = 2*np.pi):
        if self.qaoa_type == "std":
            self.beta_params = scaling_constant * self.rng.random([1, self.num_layers], requires_grad=True)
            self.gamma_params = scaling_constant * self.rng.random([1, self.num_layers], requires_grad=True)
        elif self.qaoa_type == "ma":
            self.beta_params = scaling_constant * self.rng.random([self.n_wires, self.num_layers], requires_grad=True)
            self.gamma_params = scaling_constant * self.rng.random([self.graph.numberOfEdges(), self.num_layers], requires_grad=True)
        elif self.qaoa_type == "naive_wd":
            weight_list = [float(w) for _, _, w in self.graph.iterEdgesWeights()]
            self.weight_distribution = [float(w) for w in np.unique(weight_list)]
            self.beta_params = scaling_constant * self.rng.random([self.n_wires, self.num_layers], requires_grad=True)
            self.gamma_params = scaling_constant * self.rng.random([len(self.weight_distribution), self.num_layers], requires_grad=True)
        elif self.qaoa_type == "eigenbased_wd" or self.qaoa_type == "normalized_eigenbased_wd":
            weight_list = [float(w) for _, _, w in self.eigenbased_graph.iterEdgesWeights()]
            self.weight_distribution = [float(w) for w in np.unique(weight_list)]
            self.beta_params = scaling_constant * self.rng.random([len(self.unique_orbit), self.num_layers], requires_grad=True)
            self.gamma_params = scaling_constant * self.rng.random([len(self.weight_distribution), self.num_layers], requires_grad=True)
        elif self.qaoa_type == "orbit":
            self.beta_params = scaling_constant * self.rng.random([self.num_orbits, self.num_layers], requires_grad=True)
            self.gamma_params = scaling_constant * self.rng.random([len(self.edges_orbit), self.num_layers], requires_grad=True)
        elif self.qaoa_type == "algebraic_dis":
            self.beta_params = scaling_constant * self.rng.random([len(self.unique_orbit), self.num_layers], requires_grad=True)
            self.gamma_params = scaling_constant * self.rng.random([len(self.AD_uniqueness), self.num_layers], requires_grad=True)
        else:
            raise ValueError("The given qaoa type is not supported!")
    
    def perform_optimization(self, 
                             max_trials: int = 300,
                             step_size: float = 0.5,
                             threshold: float = 1e-5):
        def U_B(beta):
            for wire in range(self.n_wires):
                if self.qaoa_type == "ma" or self.qaoa_type == "naive_wd":
                    qml.RX(2 * beta[wire], wires=wire)
                elif self.qaoa_type == "eigenbased_wd" or self.qaoa_type == "normalized_eigenbased_wd" or self.qaoa_type == "algebraic_dis":
                    qml.RX(2 * beta[self.beta_orbit_indices[wire]], wires=wire)
                elif self.qaoa_type == "orbit":
                    orbit_order = self.orbit[wire]
                    qml.RX(2 * beta[self.beta_orbit_indices.index(orbit_order)], wires=wire)
                else:
                    qml.RX(2 * beta[0], wires=wire)
        def U_C(gamma):
            edge_cnt = 0
            for u, v, w in self.graph.iterEdgesWeights():
                qml.CNOT(wires=(u, v))
                if self.qaoa_type == "ma":
                    qml.RZ(2 * w * gamma[edge_cnt], wires=v)
                    edge_cnt += 1
                elif self.qaoa_type == "naive_wd":
                    weight_idx = self.weight_distribution.index(w)
                    qml.RZ(2 * w * gamma[weight_idx], wires=v)
                elif self.qaoa_type == "eigenbased_wd" or self.qaoa_type == "normalized_eigenbased_wd":
                    weight_idx = self.weight_distribution.index(self.eigenbased_graph.weight(u, v))
                    qml.RZ(2 * w * gamma[weight_idx], wires=v)
                elif self.qaoa_type == "orbit":
                    edge_orbit_order = [self.orbit[u], self.orbit[v]]
                    edge_orbit_order.sort()
                    qml.RZ(2 * w * gamma[self.edges_orbit.index(edge_orbit_order)], wires=v)
                elif self.qaoa_type == "algebraic_dis":
                    qml.RZ(2 * w * gamma[self.AD_unique_order[edge_cnt]], wires=v)
                    edge_cnt += 1
                else:
                    qml.RZ(2 * w * gamma[edge_cnt], wires=v)
                qml.CNOT(wires=(u, v))
        dev = qml.device("lightning.qubit", wires=self.n_wires)
        
        @qml.qnode(dev)
        def circuit(gammas, betas):
            for wire in range(self.n_wires):
                qml.Hadamard(wires=wire)
            for k in range(self.num_layers):
                U_C(gammas[:, k])
                U_B(betas[:, k])
            return qml.expval(self.cost_hamiltonian)

        def objective(gammas, betas):
            return circuit(gammas, betas)

        opt = qml.AdagradOptimizer(stepsize=step_size)
        last_gamma_params = self.gamma_params
        last_beta_params = self.beta_params
        for i in range(max_trials):
            self.gamma_params, self.beta_params = opt.step(objective, self.gamma_params, self.beta_params)
            if (i + 1) % 5 == 0:
                #print(f"Objective after step {i+1:3d}: {objective(self.gamma_params, self.beta_params): .7f}")
                #print(f"Difference between parmas: ", (np.linalg.norm(self.gamma_params - last_gamma_params),  np.linalg.norm(self.beta_params - last_beta_params)))
                if np.linalg.norm(self.gamma_params - last_gamma_params) + np.linalg.norm(self.beta_params - last_beta_params) <= threshold:
                    break
            last_gamma_params = self.gamma_params
            last_beta_params = self.beta_params
        return -objective(self.gamma_params, self.beta_params), self.gamma_params, self.beta_params