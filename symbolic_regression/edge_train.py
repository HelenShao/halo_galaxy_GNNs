import numpy as np
import sys, os
from pysr import PySRRegressor

# Load data (from all graphs in test set)
dir = "/scratch/gpfs/hshao/Graph_NNs/random_subsampling/min_100_max_500/omega_m/sparsity_regularization/2_channels_add/pysr/edge_model/edge_model_data"

num_edges  = 10   # Selected random 10 edges from each graph
num_graphs = 8000
eqn_num    = 1

in_size  = 5   # vi, vj, 3 edge feat
out_size = 2   # 2 hidden channels
edge_model_data = np.zeros((num_edges*num_graphs, in_size+out_size), dtype=np.float32)
offset = 0

for i in range(num_graphs):
    size = offset+num_edges
    edge_data = np.load("%s/edge_model_data_%d.npy"%(dir, i))
    
    edge_model_data[offset:size] = edge_data
    offset = size

# Create input and output data for pysr
in_node         = edge_model_data[:,0]
in_neigh        = edge_model_data[:,1] 
in_edge         = edge_model_data[:,2:5]
latent_edge     = edge_model_data[:,5:]
num_edges = in_node.shape[0]

input = np.zeros((num_edges, 5), dtype=np.float32)
input[:,0] = in_node
input[:,1] = in_neigh
input[:,2:5] = in_edge

output = latent_edge

model = PySRRegressor(
    model_selection="best",  # Result is mix of simplicity+accuracy
    niterations=100000,
    binary_operators=["plus", "sub", "mult", "pow", "div"],
    unary_operators=[
        "inv(x) = 1/x",
        "neg",
        "abs",
        "log10",
        "log",
        "sqrt",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    
    #loss="loss(x, y) = (x - y)^2",
    # ^ Custom loss function (julia syntax)
    procs = 20,
    batch_size = 256
)

input_feat        = ['v_i', "v_j", "d", "a", "b"]
model.fit(input, output[:,eqn_num-1], variable_names = input_feat)
