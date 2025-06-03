import math
import random
import pickle
import pennylane as qml
from pennylane import numpy as np
import networkit as nk 
from networkit.algebraic import adjacencyMatrix

QAOA_type = ['std', 
             'ma', 
             'naive_wd', 
             'eigenbased_wd',
             'normalized_eigenbased_wd',
             'algebraic_dis_wd']

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
        if initial_betas is None and initial_gammas is None:
            self.setup_initial_param()
        else:
            self.beta_params = initial_betas
            self.gamma_params = initial_gammas
    
    def eigenbased_weighted_degree(self):
        eigenvals, eigenvectors = nk.algebraic.laplacianEigenvectors(self.graph, cutoff=2, reverse=True)
        self.fiedler_vector = eigenvectors[-1]
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
            
    def setup_initial_param(self):
        if self.qaoa_type == "std":
            self.beta_params = 0.01 * self.rng.random([1, self.num_layers], requires_grad=True)
            self.gamma_params = 0.01 * self.rng.random([1, self.num_layers], requires_grad=True)
        elif self.qaoa_type == "ma":
            self.beta_params = 0.01 * self.rng.random([self.n_wires, self.num_layers], requires_grad=True)
            self.gamma_params = 0.01 * self.rng.random([self.graph.numberOfEdges(), self.num_layers], requires_grad=True)
        elif self.qaoa_type == "naive_wd":
            weight_list = [float(w) for _, _, w in self.graph.iterEdgesWeights()]
            self.weight_distribution = [float(w) for w in np.unique(weight_list)]
            self.beta_params = 0.01 * self.rng.random([self.n_wires, self.num_layers], requires_grad=True)
            self.gamma_params = 0.01 * self.rng.random([len(self.weight_distribution), self.num_layers], requires_grad=True)
        elif self.qaoa_type == "eigenbased_wd" or self.qaoa_type == "normalized_eigenbased_wd":
            weight_list = [float(w) for _, _, w in self.eigenbased_graph.iterEdgesWeights()]
            self.weight_distribution = [float(w) for w in np.unique(weight_list)]
            self.beta_params = 0.01 * self.rng.random([self.n_wires, self.num_layers], requires_grad=True)
            self.gamma_params = 0.01 * self.rng.random([len(self.weight_distribution), self.num_layers], requires_grad=True)
        elif self.qaoa_type == "algebraic_dis_wd":
            pass
    
    def perform_optimization(self, 
                             num_steps: int = 50,
                             step_size: float = 0.5):
        def U_B(beta):
            for wire in range(self.n_wires):
                if self.qaoa_type == "ma" or self.qaoa_type == "eigenbased_wd" or self.qaoa_type == "normalized_eigenbased_wd" or self.qaoa_type == "naive_wd":
                    qml.RX(2 * beta[wire], wires=wire)
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
        for i in range(num_steps):
            self.gamma_params, self.beta_params = opt.step(objective, self.gamma_params, self.beta_params)
            if (i + 1) % 5 == 0:
                pass
                #print(f"Objective after step {i+1:3d}: {objective(self.gamma_params, self.beta_params): .7f}")
        return -objective(self.gamma_params, self.beta_params), self.gamma_params, self.beta_params