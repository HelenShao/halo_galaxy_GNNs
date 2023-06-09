import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_cluster import knn_graph, radius_graph
from torch.nn import Sequential, Linear, ReLU, ModuleList
from torch_geometric.nn import MessagePassing, MetaLayer, LayerNorm, GCNConv
from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_min, scatter_add
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


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
        out = self.edge_mlp(out)
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

        out = self.node_mlp(out)
        if self.residuals:
            out = out + x
        return out

# Graph Neural Network architecture, based on the Graph Network (arXiv:1806.01261)
# Employing the MetaLayer implementation in Pytorch-Geometric
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, dim_out, residuals=False):
        super().__init__()

        #self.n_layers = n_layers
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
        n_layers = 1
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

        out = torch.cat([addpool], dim=1)
        #out = addpool
        
        # Final linear layer
        out = self.outlayer(out)

        return out