import torch
import torch.nn as nn
import math
import numpy as np
import argparse
torch.set_default_dtype(torch.float64)
torch.set_default_device('cuda')  # current device is 0
seed = 40
torch.manual_seed(seed)

def I2916(X):
    return torch.sqrt(torch.pow(X[:,0],2) + torch.pow(X[:,1],2) - 2 * X[:,0] * X[:,1] * torch.cos(X[:,2] - X[:,3]))

def I2916_dimensionless(X,Y,X_dimensionless):
    X_dimensionless[:,0] = torch.div(X[:,1],X[:,0])
    X_dimensionless[:,1] = X[:,2]
    X_dimensionless[:,2] = X[:,3]
    Y_dimensionless = torch.div(Y,X[:,0])
    return X_dimensionless, Y_dimensionless

def I262(X):
    return torch.arcsin(X[:,0]*torch.sin(X[:,1]))

def I262_dimensionless(X,Y,X_dimensionless):
    return X,Y

def I62(X):
    num = torch.exp(-(torch.pow(X[:,0],2))/(2*torch.pow(X[:,1],2)))
    denom = torch.sqrt(2*math.pi*torch.pow(X[:,1],2))
    return num / denom

def I62_dimensionless(X,Y,X_dimensionless):
    return X,Y

def I1312(X):
    return X[:,4]*X[:,0]*X[:,1]*(1/X[:,3] - 1/X[:,2])

def I1312_dimensionless(X,Y,X_dimensionless):
    X_dimensionless[:,0] = torch.div(X[:,1],X[:,0])
    X_dimensionless[:,1] = torch.div(X[:,3],X[:,2])
    Y_dimensionless = X[:,2]*torch.div(Y,X[:,4]*torch.pow(X[:,0],2))
    return X_dimensionless, Y_dimensionless

def I918(X):
    num = X[:,0]*X[:,1]*X[:,2]
    denom = torch.pow(X[:,4]-X[:,3],2) + torch.pow(X[:,6]-X[:,5],2) + torch.pow(X[:,8]-X[:,7],2)
    return num / denom

def I918_dimensionless(X,Y,X_dimensionless):
    X_dimensionless[:,0] = torch.div(X[:,1],X[:,0])
    X_dimensionless[:,1] = torch.div(X[:,4],X[:,3])
    X_dimensionless[:,2] = torch.div(X[:,5],X[:,3])
    X_dimensionless[:,3] = torch.div(X[:,6],X[:,3])
    X_dimensionless[:,4] = torch.div(X[:,7],X[:,3])
    X_dimensionless[:,5] = torch.div(X[:,8],X[:,3])
    Y_dimensionless = torch.pow(X[:,3],2)*torch.div(Y,X[:,2]*torch.pow(X[:,0],2))
    return X_dimensionless, Y_dimensionless

def III952(X):
    term1 = (X[:,0]*X[:,1]*X[:,2])/(X[:,3]/(2*math.pi))
    term2 = torch.pow(torch.sin((X[:,4]-X[:,5])*X[:,2]/2),2) / torch.pow((X[:,4]-X[:,5])*X[:,2]/2,2)
    return term1 * term2

def III952_dimensionless(X,Y,X_dimensionless):
    X_dimensionless[:,0] = torch.div(X[:,3],X[:,0]*X[:,1]*X[:,2])
    X_dimensionless[:,1] = X[:,4]*X[:,2]
    X_dimensionless[:,2] = X[:,5]*X[:,2]
    return X_dimensionless, Y

def generate_data(x_dim,x_dimensionless_dim,n_samples,equation,equation_dimensionless,domains):
    # Generate uniform samples and scale them to the desired ranges
    X = torch.empty((n_samples, x_dim))
    X_dimensionless = torch.empty((n_samples, x_dimensionless_dim))
    for i in range(x_dim):
        domain = domains[i]
        domain_low = domain[0]
        domain_high = domain[1]
        X[:, i] = domain_low + (domain_high - domain_low) * torch.rand(n_samples)
    Y = equation(X)
    X_dimensionless, Y_dimensionless = equation_dimensionless(X,Y,X_dimensionless)
    return X_dimensionless, Y_dimensionless

def load_data(filename):
    data = np.loadtxt(filename)
    X = torch.tensor(data[:,0:-1], dtype=torch.float64)
    Y = torch.tensor(data[:,-1], dtype=torch.float64)
    Y = Y.unsqueeze(dim=1)
    return X,Y

x,y = load_data("../../Feynman_without_units/I.13.12")

# Compute mean and standard deviation from the training data
mean = x.mean(dim=0)
std = x.std(dim=0)
mean_y = y.mean(dim=0)
std_y = y.std(dim=0)

# Normalize training and test data
x_norm = (x - mean) / std
y_norm = (y - mean_y) / std_y

in_dim = x.shape[1]
width = 5
model = nn.Sequential(
    nn.Linear(in_dim, width),
    nn.Tanh(),
    nn.Linear(width, width),
    nn.Tanh(),
    nn.Linear(width, width),
    nn.Tanh(),
    nn.Linear(width, width),
    nn.Tanh(),
    nn.Linear(width, width),
    nn.Tanh(),
    nn.Linear(width, 1)
)
res1 = nn.Sequential(
    nn.Linear(in_dim, width),
    nn.Tanh(),
    nn.Linear(width, width),
    nn.Tanh(),
    nn.Linear(width, width),
    nn.Tanh(),
    nn.Linear(width, width),
    nn.Tanh(),
    nn.Linear(width, width),
    nn.Tanh(),
    nn.Linear(width, 1)
)
res2 = nn.Sequential(
    nn.Linear(in_dim, width),
    nn.Tanh(),
    nn.Linear(width, width),
    nn.Tanh(),
    nn.Linear(width, width),
    nn.Tanh(),
    nn.Linear(width, width),
    nn.Tanh(),
    nn.Linear(width, width),
    nn.Tanh(),
    nn.Linear(width, 1)
)
res3 = nn.Sequential(
    nn.Linear(in_dim, width),
    nn.Tanh(),
    nn.Linear(width, width),
    nn.Tanh(),
    nn.Linear(width, width),
    nn.Tanh(),
    nn.Linear(width, width),
    nn.Tanh(),
    nn.Linear(width, width),
    nn.Tanh(),
    nn.Linear(width, 1)
)
model.load_state_dict(torch.load("./results/I1312/seed40/I1312_network_0_MSE_Tanh_5-5-5-5_depth5-5-5-5_initMSE_initTanh_seed40.pt", weights_only=True))
model.eval()

res1.load_state_dict(torch.load("./results/I1312/seed40/I1312_network_1_MSE_Tanh_5-5-5-5_depth5-5-5-5_initMSE_initTanh_seed40.pt", weights_only=True))
res1.eval()

res2.load_state_dict(torch.load("./results/I1312/seed40/I1312_network_2_MSE_Tanh_5-5-5-5_depth5-5-5-5_initMSE_initTanh_seed40.pt", weights_only=True))
res2.eval()

res3.load_state_dict(torch.load("./results/I1312/seed40/I1312_network_3_MSE_Tanh_5-5-5-5_depth5-5-5-5_initMSE_initTanh_seed40.pt", weights_only=True))
res3.eval()

eps = np.load("./results/I1312/seed40/I1312_epsilons_MSE_Tanh_5-5-5-5_depth5-5-5-5_initMSE_initTanh_seed40.npy")

fourth_test_pred = model(x_norm) + eps[0]*res1(x_norm) + eps[1]*res2(x_norm) + eps[2]*res3(x_norm)
rmse_loss_fn_torch = lambda x, y: torch.sqrt(torch.mean(torch.pow(x-y, 2)))
with torch.no_grad():
    print(rmse_loss_fn_torch(fourth_test_pred, y_norm).item())

val_x, val_y = generate_data(5, 2, 1000000, I1312, I1312_dimensionless, [(1,5),(1,5),(1,5),(1,5),(1,5)])
val_x_norm = (val_x - mean) / std
val_y_norm = (val_y - mean_y) / std_y
val_pred = model(val_x_norm) + eps[0]*res1(val_x_norm) + eps[1]*res2(val_x_norm) + eps[2]*res3(val_x_norm)
with torch.no_grad():
    print(rmse_loss_fn_torch(val_pred.squeeze(1), val_y_norm).item())