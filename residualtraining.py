import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize

def extract_params(mlp):
    """ Extracts parameters from an MLP and returns them as a flattened vector with shape/length metadata. """
    params = [p.data.cpu().numpy().flatten() for p in mlp.parameters()]
    shapes = [p.shape for p in mlp.parameters()]
    lengths = [p.numel() for p in mlp.parameters()]
    return np.concatenate(params), shapes, lengths

def initialize_NN(in_dim, out_dim, width, depth, activation):
    """Initializes a neural network with the given depth and width, then extracts initial parameters."""
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    layers = [nn.Linear(in_dim, width), activation]  # Input layer
    
    # Hidden layers based on depth parameter
    for _ in range(depth - 1):  # depth - 1 because input to first hidden layer is already added
        layers.append(nn.Linear(width, width))
        layers.append(activation)
    
    layers.append(nn.Linear(width, out_dim))  # Output layer
    
    mlp = nn.Sequential(*layers).to(device)
    
    param_vector, shapes, lengths = extract_params(mlp)
    return mlp, param_vector, shapes, lengths

def bfgs_optimize(model, loss_fn, data, target, max_iter=100, tolerance=1e-60):
    """
    Optimizes a PyTorch model using SciPy's BFGS method and tracks training loss.
    
    Args:
        model (torch.nn.Module): The PyTorch model to optimize.
        loss_fn (callable): The loss function.
        data (torch.Tensor): Input data.
        target (torch.Tensor): Target labels.
        max_iter (int): Maximum iterations for BFGS.
    
    Returns:
        loss_history (list): Training loss at each iteration.
    """

    loss_history = []  # Store loss values

    # Convert model parameters to a flat NumPy array
    def get_params():
        return np.concatenate([p.cpu().detach().numpy().flatten() for p in model.parameters()])

    # Load a flat NumPy array back into model parameters
    def set_params(params):
        with torch.no_grad():
            start = 0
            for p in model.parameters():
                size = p.numel()
                p.copy_(torch.tensor(params[start:start + size]).reshape(p.shape))
                start += size

    # Compute loss given parameters
    def loss_wrapper(params):
        set_params(params)
        loss = loss_fn(model(data), target)
        return loss.item()

    # Compute gradients given parameters
    def grad_wrapper(params):
        set_params(params)
        loss = loss_fn(model(data), target)
        loss.backward()  # Compute gradients
        grads = np.concatenate([p.grad.cpu().detach().numpy().flatten() for p in model.parameters()])
        model.zero_grad()  # Reset gradients
        return grads

    # Callback function to track loss
    def callback(params):
        loss = loss_wrapper(params)
        loss_history.append(loss)

    # Get initial parameters
    initial_params = get_params()

    # Run SciPy BFGS optimization with loss tracking
    result = minimize(loss_wrapper, initial_params, jac=grad_wrapper, method="BFGS",
                      options={'disp': True, 'gtol': tolerance, 'maxiter': max_iter},
                      callback=callback)

    # Set the model parameters to optimized values
    set_params(result.x)

    if result.success:
        print("Optimization successful!")
    else:
        print("Optimization failed:", result.message)

    return loss_history  # Return the recorded loss history

def residual_training(x_train, y_train, init_width, init_depth, init_activation, init_loss_func, widths, depths, activation, loss_func):
    """ Performs residual training using multiple neural networks and tracks loss. """
    in_dim = x_train.shape[-1]
    out_dim = y_train.shape[-1]
    init_NN, param_vector, shapes, lengths = initialize_NN(in_dim, out_dim, init_width, init_depth, init_activation)
    print("Training initial network")
    loss_init = bfgs_optimize(init_NN, init_loss_func, x_train, y_train, max_iter=25000)
    print("Initial network trained")
    n = len(widths)
    with torch.no_grad():
        residual = y_train - init_NN(x_train)
        eps = torch.max(torch.abs(residual))
        rhat = residual / eps

    models, epsilons, loss_histories = [], [eps], [loss_init]  # Track losses
    for i in range(n):
        res_NN, param_vector, shapes, lengths = initialize_NN(in_dim, out_dim, widths[i], depths[i], activation)
        print("Training residual network " + str(i))
        loss_res = bfgs_optimize(res_NN, loss_func, x_train, rhat, max_iter=25000)
        print("Residual network " + str(i) + " trained")

        with torch.no_grad():
            residual -= epsilons[-1] * res_NN(x_train)
            epsilons.append(epsilons[-1] * torch.max(torch.abs(rhat - res_NN(x_train))))
            rhat = residual / epsilons[-1]

        models.append(res_NN)
        loss_histories.append(loss_res)  # Store loss history

    return init_NN, models, epsilons, loss_histories