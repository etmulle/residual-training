import os
import pickle
import numpy as np
import torch

def load_data(filename):
    data = np.loadtxt(filename)
    X = torch.tensor(data[:, 0:-1], dtype=torch.float64)
    Y = torch.tensor(data[:, -1], dtype=torch.float64).unsqueeze(dim=1)
    return X, Y

def normalize_data(x, y, normalize_y):
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    x_norm = (x - mean) / std
    if normalize_y:
        mean_y = y.mean(dim=0)
        std_y = y.std(dim=0)
        y_norm = (y - mean_y) / std_y
        return x_norm, y_norm
    else:
        print("x normalized, y left unnormalized")
        return x_norm, y

def weighted_mse_loss(predictions, targets):
    weights = torch.abs(targets)
    loss = weights * (predictions - targets) ** 2
    return torch.mean(loss)

def save_results(epsilons, init_NN, init_loss, init_activation, models, example, loss_func, activation, neurons, depths, loss_histories, seed):
    os.makedirs(f"./results/{example}/seed{seed}", exist_ok=True)

    # Save epsilons
    eps = np.array([e.item() for e in epsilons])
    np.save(f"./results/{example}/seed{seed}/{example}_epsilons_{loss_func}_{activation}_{neurons}_depth{depths}_init{init_loss}_init{init_activation}_seed{seed}.npy", eps)

    # Save models
    for i, network in enumerate([init_NN, *models]):
        torch.save(network.state_dict(), f"./results/{example}/seed{seed}/{example}_network_{i}_{loss_func}_{activation}_{neurons}_depth{depths}_init{init_loss}_init{init_activation}_seed{seed}.pt")

    # Save loss history
    with open(f"./results/{example}/seed{seed}/{example}_losses_{loss_func}_{activation}_{neurons}_depth{depths}_init{init_loss}_init{init_activation}_seed{seed}.pkl", "wb") as f:
        pickle.dump(loss_histories, f)
    print("Data saved to " + f"./results/{example}/seed{seed}")
