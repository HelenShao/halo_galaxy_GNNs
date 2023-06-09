import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader
import scipy.spatial as SS
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_cluster import knn_graph, radius_graph
from torch.nn import Sequential, Linear, ReLU, ModuleList
from torch_geometric.nn import MessagePassing, MetaLayer, LayerNorm, GCNConv
from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_min, scatter_add
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

# Compute KDTree and get edges and edge features
def get_edges(pos, r):

    # 1. Get edges

    # Create the KDTree and look for pairs within a distance r
    kd_tree = SS.KDTree(pos, boxsize=1.0001)
    edge_index = kd_tree.query_pairs(r, output_type="ndarray")

    # Add reverse pairs
    reversepairs = np.zeros((edge_index.shape[0],2))
    for i, pair in enumerate(edge_index):
        reversepairs[i] = np.array([pair[1], pair[0]])
    edge_index = np.append(edge_index, reversepairs, 0)

    edge_index = edge_index.astype(int)

    # Write in pytorch-geometric format
    edge_index = edge_index.reshape((2,-1))
    num_pairs = edge_index.shape[1]

    # 2. Get edge attributes

    row, col = edge_index
    diff = pos[row]-pos[col]
    #print(np.sum(diff), np.size(diff))
    #print(np.mean(diff))

    # Take into account periodic boundary conditions, correcting the distances
    for i, pos_i in enumerate(diff):
        for j, coord in enumerate(pos_i):
            if coord > r:
                diff[i,j] -= 1.  # Boxsize normalize to 1
            elif -coord > r:
                diff[i,j] += 1.  # Boxsize normalize to 1

    # Get translational and rotational invariant features
    
    # Distance
    dist = np.linalg.norm(diff, axis=1)
    
    # Centroid of galaxy catalogue
    centroid = np.mean(pos,axis=0)
    
    # Unit vectors of node, neighbor and difference vector
    unitrow = (pos[row]-centroid)/np.linalg.norm((pos[row]-centroid), axis=1).reshape(-1,1)
    unitcol = (pos[col]-centroid)/np.linalg.norm((pos[col]-centroid), axis=1).reshape(-1,1)
    unitdiff = diff/dist.reshape(-1,1)
    
    # Dot products between unit vectors
    cos1 = np.array([np.dot(unitrow[i,:].T,unitcol[i,:]) for i in range(num_pairs)])
    cos2 = np.array([np.dot(unitrow[i,:].T,unitdiff[i,:]) for i in range(num_pairs)])
    
    # Normalize distance by linking radius
    dist /= r

    # Concatenate to get all edge attributes
    edge_attr = np.concatenate([dist.reshape(-1,1), cos1.reshape(-1,1), cos2.reshape(-1,1)], axis=1)

    return edge_index, edge_attr

# Create graph data
def create_graph(vel_file, pos_file, sim_num, r):
    
    # Read halo positions
    pos = np.load(pos_file)
    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    
    # Normalize to fit inside boxsize of 1
    x_min, x_max = 1e-05, 24.99998
    y_min, y_max = 0.0, 24.99996
    z_min, z_max = 1e-05, 24.99998

    x = (x - x_min)/(x_max - x_min)
    y = (y - y_min)/(y_max - y_min)
    z = (z - z_min)/(z_max - z_min)
    
    pos = np.column_stack((x,y,z))
    
    # Read halo velocities
    vel = np.load(vel_file)
    
    # Normalize velocity modulus
    mean, std = 188.83379, 129.66516
    vel = (vel - mean)/std
    vel = vel.reshape(-1,1)
    
    # Get the output to be predicted by the GNN: cosmological parameter(s) (normalized)
    y = np.load("/scratch/gpfs/hshao/Graph_NNs/random_subsampling/HaloData/omega_m.npy")[sim_num]

    # Get edges and edge features
    edge_index, edge_attr = get_edges(pos, r,)

    # Construct the graph
    graph = Data(x=torch.tensor(vel, dtype=torch.float32),
                 y=torch.tensor(y, dtype=torch.float32),
                 edge_index=torch.tensor(edge_index, dtype=torch.long),
                 edge_attr=torch.tensor(edge_attr, dtype=torch.float32))

    return graph

# Create dataset
def create_dataset(num_catalogs, r, seed):
    # Create data container
    dataset = []
    
    # Number of random N-values. N = particle density baseline
    num_random = 10
    
    # Shuffle the simulations
    np.random.seed(seed)
    indexes = np.arange(num_catalogs)
    #np.random.shuffle(indexes)
    
    for i in indexes:
        for j in range(num_random):
            
            # Get position and vel files --> make graph and append to dataset
            pos_file = "/scratch/gpfs/hshao/Graph_NNs/random_subsampling/min_100_max_500/HaloData/LH_%d_pos_%d.npy"%(i, j)
            vel_file = "/scratch/gpfs/hshao/Graph_NNs/random_subsampling/min_100_max_500/HaloData/LH_%d_vel_%d.npy"%(i, j)

            # Create graph and append to data container
            dataset.append(create_graph(vel_file, pos_file, i, r))
        
    return dataset

# Create dataloaders for train, valid, test
def create_loaders(dataset, seed, batch_size):

    # First 800 simulations --> first 8000 catalogs --> train
    # Next 200 simulations --> next 2000 catalogs --> valid
    # Next 200 simulations --> next 2000 catalogs --> test
    
    num_workers = 10
    size_train, offset_train = int(0.8 * len(dataset)), 0
    size_valid, offset_valid = int(0.1 * len(dataset)), int(0.8 * len(dataset))
    size_test,  offset_test  = int(0.1 * len(dataset)), int(0.9 * len(dataset))

    train_dataset = dataset[offset_train:size_train]
    valid_dataset = dataset[offset_valid:offset_valid+size_valid]
    test_dataset = dataset[offset_test:offset_test+size_test]

    torch.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, valid_loader, test_loader