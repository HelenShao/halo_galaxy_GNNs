import numpy as np
import matplotlib.pyplot as plt
import sys, os
from pysr import PySRRegressor

# Load data (from all graphs in test set)
dir = "../node_model_data"
num_nodes  = 10 # Selected 10 nodes from each graph
num_graphs = 8000
node_model_data = np.zeros((10*num_graphs, 5), dtype=np.float32)
offset = 0
#node_feat_num = 1

for i in range(num_graphs):
    size = offset+num_nodes
    node_data = np.load("%s/node_model_data_%d.npy"%(dir, i))
    
    node_model_data[offset:size] = node_data
    offset = size

# Create input and output data for pysr
in_node         = node_model_data[:,0]     # initial node feat, vel_i
hid_edge        = node_model_data[:,1:3]   # 3 latent edge feat (sum)
latent_node     = node_model_data[:,3:]    # 3 latent node feat (output)
num_nodes       = in_node.shape[0]         # total number of nodes used: 10*1000

input = np.zeros((num_nodes, 3), dtype=np.float32)
input[:,0], input[:,1:3] = in_node, hid_edge

output = latent_node

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
    batch_size = 128
)
input_feat = ["vi", "s1", "s2"]
model.fit(input, output[:,0]+output[:,1], variable_names = input_feat)
