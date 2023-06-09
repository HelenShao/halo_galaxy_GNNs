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
import architecture, datas

################################# Objective Function #############################
class objective(object):
    def __init__(self, num_catalogues, device, n_max, n_min, num_epochs, dim_out, seed, batch_size): 
        
        self.num_catalogues     = num_catalogues
        self.device             = device
        self.n_max              = n_max
        self.n_min              = n_min
        self.num_epochs         = num_epochs
        self.dim_out            = dim_out
        self.seed               = seed 
        self.batch_size         = batch_size
    
    def __call__(self, trial):
        
        # Files for saving results and best model
        f_text_file   = 'HALOS_GNN_%d.txt'%(trial.number)
        f_best_model  = 'HALOS_GNN_%d.pt'%(trial.number)
        
        # Create datasets
        r = trial.suggest_float("r", 1e-2, .1, log=False)
        dataset = datas.create_dataset(num_catalogues, r, seed)

        # Create Dataloaders
        torch.manual_seed(seed)
        train_loader, valid_loader, test_loader = datas.create_loaders(dataset, seed, batch_size)
        
        # Generate the model
        # n_layers        = trial.suggest_int("n_layers", min_layers, max_layers)
        hidden_channels = trial.suggest_int("hid_channels", n_min, n_max)
        model = architecture.GNN(hidden_channels, dim_out, residuals=False).to(device)
        
        # Define optimizer
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        wd = trial.suggest_float("wd", 1e-8, 1e1, log=True)
        optimizer = getattr(optim, "Adam")(model.parameters(), lr=lr, weight_decay = wd)
        
        # Train the model
        min_valid = 1e40
        for epoch in range(num_epochs):
            model.train()
            count, loss_train = 0, 0.0
            for data in train_loader:        
                # Forward Pass
                data = data.to(device=device)
                output = model(data)
                y_out, err_out = output[:,0], output[:,1]
                
                # Compute loss as sum of two terms for likelihood-free inference
                loss_mse = torch.mean(torch.sum((y_out - data.y)**2.) , axis=0)
                loss_lfi = torch.mean(torch.sum(((y_out - data.y)**2. - err_out**2.)**2.) , axis=0)
                loss = torch.log(loss_mse) + torch.log(loss_lfi)
                
                loss_train += loss.cpu().detach().numpy()

                # Backward propogation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation of the model.
            model.eval() 
            count, loss_valid = 0, 0.0
            for data in valid_loader:
                data = data.to(device=device)
                output = model(data)
                y_out, err_out = output[:,0], output[:,1]
                
                # Compute loss as sum of two terms for likelihood-free inference
                loss_mse = torch.mean(torch.sum((y_out - data.y)**2.) , axis=0)
                loss_lfi = torch.mean(torch.sum(((y_out - data.y)**2. - err_out**2.)**2.) , axis=0)
                loss = torch.log(loss_mse) + torch.log(loss_lfi)
                
                loss_valid += loss.cpu().detach().numpy()
                count += 1
            loss_valid /= count

            if loss_valid<min_valid:  
                min_valid = loss_valid
                torch.save(model.state_dict(), f_best_model)
            f = open(f_text_file, 'a')
            f.write('%d %.5e %.5e\n'%(epoch, loss_valid, min_valid))
            f.close()

        return min_valid

##################################### INPUT #######################################
# Data Parameters
seed       = 4
num_catalogues = 1000

# Training Parameters
num_epochs = 700
batch_size = 8

# Architecture Parameters
n_min = 1             # Minimum number of neurons in hidden layers
n_max = 2             # Maximum number of neurons in hidden layers
#min_layers = 1          # Minimum number of hidden layers
#max_layers = 2          # Maximum number of hidden layers
dim_out = 2             # Output: [mean, std]

# Optuna Parameters
n_trials   = 1000
study_name = 'HALOS_GNN_params'
n_jobs     = 1
storage_m  = 'sqlite:///./database.db'

############################## Start OPTUNA Study ###############################

# Use GPUs if avaiable
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

if __name__ == "__main__":
    
    # define the optuna study and optimize it
    objective = objective(num_catalogues, device, n_max, n_min, num_epochs, dim_out, seed, batch_size)
    
    # !! Optimization direction = minimize valid_loss !!
    sampler = optuna.samplers.TPESampler(n_startup_trials=8) 
    study = optuna.create_study(study_name=study_name, sampler=sampler,direction="minimize",
                                load_if_exists = True, storage=storage_m)
    study.optimize(objective, n_trials=n_trials, n_jobs = n_jobs)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    # Print parameters of the best trial
    trial = study.best_trial
    print("Best trial: number {}".format(trial.number))
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
