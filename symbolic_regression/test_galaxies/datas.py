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

# Compute KDTree and get edges and edge features
def get_edges(pos, r):
    
    # Get edges

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
def create_graph(pos_file, other_file, r, num_catalog, param):
    
    # Read halo catalogue
    pos = np.load(pos_file)/1000  # Convert to Mpc
    
    # Read the catalog: halo positions
    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    
    # Normalize to fit inside boxsize of 1
    x_min, x_max = 0.0, 25.0
    y_min, y_max = 0.0, 25.0
    z_min, z_max = 0.0, 25.0
    
    x = (x - x_min)/(x_max - x_min)
    y = (y - y_min)/(y_max - y_min)
    z = (z - z_min)/(z_max - z_min)
    
    pos = np.column_stack((x,y,z))
    
    # Get edges and edge features
    edge_index, edge_attr = get_edges(pos, r)

    # Get the output to be predicted by the GNN: normalize cosmological parameter(s)
    if param == "sigma8":
        y = np.loadtxt("/projects/QUIJOTE/CAMELS/Sims/CosmoAstroSeed_params_Astrid.txt", usecols=2)[num_catalog] 
        
        # Normalize output
        min, max = 0.6, 1.0
        y = (y-min)/(max-min)
        
    elif param == "omega_m":
        y = np.loadtxt("/projects/QUIJOTE/CAMELS/Sims/CosmoAstroSeed_params_Astrid.txt", usecols=1)[num_catalog] 
        
        # Normalize output
        min, max = 0.1, 0.5
        y = (y-min)/(max-min)
    
    # Get the node features 
    if param == "sigma8":
        mass = np.load(other_file)
        mass = np.log10(mass)
        mean, std = 10.604683, 0.5476572
        mass = (mass - mean) / std
        x = torch.tensor(mass.reshape(-1,1), dtype=torch.float32)
       
    if param == "omega_m":
        vel = np.load(other_file)
        mean, std = 188.22827, 129.45273
        vel = (vel - mean) / std
        x = torch.tensor(vel.reshape(-1,1), dtype=torch.float32)

    # Construct the graph
    graph = Data(x=x,
                 y=torch.tensor(y, dtype=torch.float32),
                 edge_index=torch.tensor(edge_index, dtype=torch.long),
                 edge_attr=torch.tensor(edge_attr, dtype=torch.float32))

    return graph

# Create dataset
def create_dataset(num_catalogs, r, param, particles):
    # Create data container
    dataset = []
    
    if param == "sigma8":
        
        for i in range(num_catalogs):
            pos_file = "/scratch/gpfs/hshao/GNN_galaxies/Astrid/galaxy_data_3_10/%d/galaxy_pos_%d.npy"%(i, particles)
            mass_file = "/scratch/gpfs/hshao/GNN_galaxies/Astrid/galaxy_data_3_10/%d/galaxy_mass_%d.npy"%(i, particles)
        
            # Create graph and append to data container
            dataset.append(create_graph(pos_file, mass_file, r, i, param))
        
    elif param == "omega_m":
        
        for i in range(num_catalogs):
            pos_file = "/scratch/gpfs/hshao/GNN_galaxies/Astrid/galaxy_data_3_10/%d/galaxy_pos_%d.npy"%(i, particles)
            vel_file = "/scratch/gpfs/hshao/GNN_galaxies/Astrid/galaxy_data_3_10/%d/galaxy_vel_%d.npy"%(i, particles)
        
            # Create graph and append to data container
            dataset.append(create_graph(pos_file, vel_file, r, i, param))
        
    return dataset

# Create dataloaders for train, valid, test
def create_loaders(dataset, seed, batch_size):

    #np.random.seed(seed)
    #np.random.shuffle(dataset)
    
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return test_loader