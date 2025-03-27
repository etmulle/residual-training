import torch
import math

def I2916(x):
    return torch.sqrt(torch.pow(X[:,0],2) + torch.pow(X[:,1],2) - 2 * X[:,0] * X[:,1] * torch.cos(X[:,2] - X[:,3]))

def I2916_dimensionless(X,Y,X_dimensionless):
    X_dimensionless[:,0] = torch.div(X[:,1],X[:,0])
    X_dimensionless[:,1] = X[:,2]
    X_dimensionless[:,2] = X[:,3]
    Y_dimensionless = torch.div(Y,X[:,0])
    return X_dimensionless, Y_dimensionless

def I262(x):
    return torch.arcsin(X[:,0]*torch.sin(X[:,1]))

def I262_dimensionless(X,Y,X_dimensionless):
    return X,Y

def I62(x):
    num = torch.exp(-(torch.pow(X[:,0],2))/(2*torch.pow(X[:,1],2)))
    denom = torch.sqrt(2*math.pi*torch.pow(X[:,1],2))
    return num / denom

def I62_dimensionless(X,Y,X_dimensionless):
    return X,Y

def I1312(x):
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
    Y_dimensionless = torch.pow(X[:,3],2)*torch.div(Y,X[:,2]torch.pow(X[:,0],2))
    return X_dimensionless, Y_dimensionless

def III952(x):
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
        domain_low, domain_high = *domains[i]
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

