import numpy as np
import sys, os, time
import torch 
import torch.nn as nn
import datas
import architecture
import optuna

################################### INPUT #####################################
seed           = 4
batch_size     = 1
num_catalogues = 1000    

# Optuna input
study_name  = 'HALOS_GNN_params'
storage     = 'sqlite:///./database.db'
model_ranks = [0,1,2,3]
      
# Use GPUs 
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')   

# Get Optuna parameters
def read_database(storage, study_name, model_rank):

    # load the optuna study
    study = optuna.load_study(study_name=study_name, storage=storage)

    # get the scores of the study trials
    values = np.zeros(len(study.trials))
    completed = 0
    for i,t in enumerate(study.trials):
        values[i] = t.value
        if t.value is not None:  completed += 1

    # get the info of the best trial
    indexes = np.argsort(values)
    for i in [model_rank]: 
        trial = study.trials[indexes[i]]
        print("\nTrial number {}".format(trial.number))
        print("Value: %.5e"%trial.value)
        print(" Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        num          = trial.number 
        r            = trial.params['r']
        hid_channels = trial.params['hid_channels']
        lr           = trial.params['lr']
        wd           = trial.params['wd']

    return num, r, hid_channels, lr, wd

# load model
def load_model(storage, study_name, model_rank, device):

    # get the best trial from the database
    num, r, hid_channels, lr, wd = read_database(storage, study_name, model_rank)

    # get the name of the file with the network weights
    f_best_model = 'HALOS_GNN_%d.pt'%(num)
    print(f_best_model)

    # load the model weights
    model = architecture.GNN(hid_channels, dim_out=2, residuals=False).to(device)
    if os.path.exists(f_best_model):
        print("loading model")
        model.load_state_dict(torch.load(f_best_model, map_location=device))
    else:
        raise Exception('model doesnt exists!!!')
        
    network_total_params = sum(p.numel() for p in model.parameters())
    print('total number of parameters in the model = %d'%network_total_params)
    
    return model

# Create valid and test loaders
def create_loaders(storage, study_name, model_rank, num_catalogues, batch_size, seed):
    num, r, hid_channels, lr, wd = read_database(storage, study_name, model_rank)
    dataset = datas.create_dataset(num_catalogues, r, seed)

    torch.manual_seed(seed)
    train_loader, valid_loader, test_loader = datas.create_loaders(dataset, seed, batch_size)
    
    return valid_loader, test_loader

###############################################################################################################
for model_rank in model_ranks:
    # Load best model and data
    model = load_model(storage, study_name, model_rank, device)
    valid_loader, test_loader = create_loaders(storage, study_name, model_rank, num_catalogues, batch_size, seed)

    # Find validation and test loss

    # VALIDATION
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

    # TEST

    # Store results for plotting:
    n_test = int(num_catalogues*.1*10)
    true = np.zeros((n_test, 1), dtype = np.float32)
    pred = np.zeros((n_test, 2), dtype = np.float32)

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

    print("-------------------- Loss: Valid and Test ---------------------")
    print('%.4e %.4e'%(loss_valid, loss_test))
    print("")

    print("--------------------- Predicted vs Truth ----------------------")
    num, r, hid_channels, lr, wd = read_database(storage, study_name, model_rank)
    np.savetxt("predicted_%d.txt"%num, pred)
    np.savetxt("true_%d.txt"%num, true)
    print("Saved true and predicted results")

