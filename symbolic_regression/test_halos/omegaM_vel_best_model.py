import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader
import scipy.spatial as SS
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_cluster import knn_graph, radius_graph
from torch.nn import Sequential, Linear, ReLU, ModuleList
from torch_geometric.nn import MessagePassing, MetaLayer, LayerNorm, GCNConv
from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_min, scatter_add
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import datas
import os

"""Trial number 20
Value: -1.31638e+01
 Params: 
    hid_channels: 2
    lr: 0.0008459222423365806
    r: 0.05427228453952218
    wd: 0.00011206070251102291
HALOS_GNN_20.pt
loading model
total number of parameters in the model = 56"""

################################### INPUT #####################################
# Data parameters
seed      = 4
r         = 0.05427228453952218
trial     = 20
slurm     = 2
param     = "omega_m"
particles = [100, 200, 300, 400, 500]

# Training Parameters
batch_size    = 1

# Architecture parameters
num_catalogues = 1
dim_out        = 2             
n_layers       = 1
hid_channels   = 2
f_best_model   = '/scratch/gpfs/hshao/Graph_NNs/random_subsampling/min_100_max_500/omega_m/sparsity_regularization/2_channels_add/slurm%d/HALOS_GNN_%d.pt'%(slurm, trial)

# Model for updating edge attritbutes (used as hidden layer)
class EdgeModel(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels, residuals=False):
        super().__init__()
        
        self.residuals = residuals
        layers = [Linear(node_in*2 + edge_in, hid_channels),
                  ReLU(),
                  Linear(hid_channels, edge_out)]
        
        self.edge_mlp = Sequential(*layers)

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges, F_x = node feature vector
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.

        out = torch.cat([src, dest, edge_attr], dim=1)
        
        vi = torch.flatten(src)
        vj = torch.flatten(dest)
        d  = torch.flatten(edge_attr[:,0])
        a  = torch.flatten(edge_attr[:,1])
        b  = torch.flatten(edge_attr[:,2])
        
        out = self.edge_mlp(out)
        
        out[:,0] = torch.abs(((vi / 0.9484139) - (vj - 0.2123214)) * -1.3248432) + ((((d - 1.7348461) + b) + vj) * -0.12084719)
        out[:,1] = ((torch.abs(((vi - (vj * 1.0584362)) * 1.5344211) - -0.45368108) + ((vi - (vj * 1.0239582)) * 1.931712)) + 0.546892)
        
        if self.residuals:
            out = out + edge_attr
        return out

# Model for updating node attritbutes (used as hidden layer)
class NodeModel(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, hid_channels, residuals=True):
        super().__init__()

        self.residuals = residuals

        layers = [Linear(node_in + 1*edge_out, hid_channels),
                  ReLU(),
                  Linear(hid_channels, node_out)]
        
        self.node_mlp = Sequential(*layers)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes, F_x = node feature vector
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        row, col = edge_index
        out = edge_attr

        # Single pooling layer
        out1 = scatter_add(out, col, dim=0, dim_size=x.size(0))
        #out2 = scatter_max(out, col, dim=0, dim_size=x.size(0))[0]
        #out3 = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out1], dim=1)

        vi = out[:,0]
        s1 = out[:,1]
        s2 = out[:,2]
       
        out = self.node_mlp(out)
        n1 = ((((0.7660379 ** ((s2 / 0.3038425) + s1)) + ((0.12117091 ** s1) / -0.7256157)) * (1.2125463 ** vi)) + 0.12262904)
        n1_n2 = (0.7872602 - torch.sqrt(torch.log((0.1562228 ** (s2 + (((s1 + -3.283101) - (vi / 0.79082423)) * 0.31992579))) - -1.4462701)))
        
        out[:,0] = n1
        out[:,1] = n1_n2 - n1

        if self.residuals:
            out = out + x
        return out

# Graph Neural Network architecture, based on the Graph Network (arXiv:1806.01261)
# Employing the MetaLayer implementation in Pytorch-Geometric
class GNN(torch.nn.Module):
    def __init__(self, n_layers, hidden_channels, dim_out,residuals=False):
        super().__init__()

        self.n_layers = n_layers
        self.dim_out = dim_out

        # Input features: 
        node_in = 1 # vel mod
        edge_in = 3 # |p_i-p_j|, p_i*p_j, p_i*(p_i-p_j)
        node_out = hidden_channels
        edge_out = hidden_channels
        hid_channels = hidden_channels

        layers = []

        # Encoder graph block
        inlayer = MetaLayer(node_model=NodeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=residuals),
                            edge_model=EdgeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=residuals))

        layers.append(inlayer)

        # Change input node and edge feature sizes
        node_in = node_out
        edge_in = edge_out

        # Hidden graph blocks
        for i in range(n_layers-1):

            lay = MetaLayer(node_model=NodeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=residuals),
                            edge_model=EdgeModel(node_in, node_out, edge_in, edge_out, hid_channels, residuals=residuals))
            layers.append(lay)

        self.layers = ModuleList(layers)

        # Final aggregation layer
        self.outlayer = Sequential(Linear(1*node_out, hid_channels),
                              ReLU(),
                              Linear(hid_channels, hid_channels),
                              ReLU(),
                              Linear(hid_channels, hid_channels),
                              ReLU(),
                              Linear(hid_channels, self.dim_out))

    def forward(self, data):

        h, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Message passing layers
        for layer in self.layers:
            h, edge_attr, _ = layer(h, edge_index, edge_attr, data.batch)

        # Single pooling layer
        addpool = global_add_pool(h, data.batch)
        #meanpool = global_mean_pool(h, data.batch)
        #maxpool = global_max_pool(h, data.batch)

        addpool = torch.cat([addpool], dim=1)
        s1 = addpool[:,0]
        s2 = addpool[:,1]

        #out = addpool
        
        # Final linear layer
        out = self.outlayer(addpool)
        out[:,0] = (((((s2 / -0.18032177) + (s1 * 2.2054937)) + torch.abs((s2 * 0.9565731) + (s1 * 0.8225316))) * 0.00046277698) + -0.24634261)

        return out

# Use GPUs 
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')    

# Load best model
model = GNN(n_layers, hid_channels, dim_out, residuals=False).to(device)
if os.path.exists(f_best_model):
    print("loading model")
    model.load_state_dict(torch.load(f_best_model, map_location=device))
    
# Find test loss
print(f_best_model)

for particle in particles:
    
    # Create test loader
    dataset = datas.create_dataset(num_catalogues, r, param, particle)
    torch.manual_seed(seed)
    test_loader = datas.create_loaders(dataset, seed, batch_size)
    
    n_test = num_catalogues
    true = np.zeros((n_test, 1), dtype = np.float32)
    pred = np.zeros((n_test, 2), dtype = np.float32)

    # TEST
    i = -1 
    count, loss_test = 0, 0.0
    for data in test_loader:
        i +=1 
        data = data.to(device=device)
        target = data.y
        true[i] = target.cpu().detach().numpy() # For plotting
        output = model(data)
        y_out, err_out = output[:,0], output[:,1]
        pred[i] = output.cpu().detach().numpy() # For plotting

        # Compute loss as sum of two terms for likelihood-free inference
        loss_mse = torch.mean(torch.sum((y_out - data.y)**2.) , axis=0)
        loss_lfi = torch.mean(torch.sum(((y_out - data.y)**2. - err_out**2.)**2.) , axis=0)
        loss = torch.log(loss_mse) + torch.log(loss_lfi)

        loss_test += loss.cpu().detach().numpy()
        count += 1
    loss_test /= count


    print("-------------------- Loss:  Test ---------------------")
    print('%.4e'%(loss_test))
    print("")

    print("--------------------- Predicted vs Truth ----------------------")
    #np.savetxt("predicted_omegaM.txt", denorm_pred)
    #np.savetxt("true_omegaM.txt", denorm_true)
    np.savetxt("predicted3_omegaM_vel_%d.txt"%particle, pred)
    np.savetxt("true3_omegaM_vel_%d.txt"%particle, true)
    print("Saved true and predicted results: %d"%particle)
