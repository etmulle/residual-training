import argparse
import torch
import torch.nn as nn
import numpy as np
from utils import load_data, normalize_data, save_results, weighted_mse_loss  # Replace with actual module
from residualtraining import residual_training

def main():
    torch.set_default_dtype(torch.float64)
    torch.set_default_device('cuda')  # current device is 0
    print(torch.__version__)
    parser = argparse.ArgumentParser(description='Train a residual network.')
    parser.add_argument('--SEED', type=int, default=39, help='Random seed')
    parser.add_argument('--EQUATION_NUMBER', type=str, default='I918', help='Equation number')
    parser.add_argument('--INIT_LOSS_FUNC', type=str, default='MSE', help='Initial network loss function')
    parser.add_argument('--INIT_ACTIVATION_NAME', type=str, default='Tanh', help='Initial network activation function')
    parser.add_argument('--LOSS_FUNC', type=str, default='MSE', help='Loss function')
    parser.add_argument('--ACTIVATION_NAME', type=str, default='Tanh', help='Activation function')
    parser.add_argument('--NEURON_WIDTHS', type=str, default='5-5-5-5', help='Neuron widths')
    parser.add_argument('--NEURON_DEPTHS', type=str, default='5-5-5-5', help='Neuron depths')
    parser.add_argument('--DATA_PATH', type=str, default='../../Feynman_without_units/I.9.18', help='Data path')
    
    args = parser.parse_args()
    print("Seed: ", args.SEED)
    print("Equation: ", args.EQUATION_NUMBER)
    print("Loss function: ", args.LOSS_FUNC)
    print("Activation function: ", args.ACTIVATION_NAME)
    print("Network widths: ", args.NEURON_WIDTHS)
    print("Network depth: ", args.NEURON_DEPTHS)
    print("Data path: ", args.DATA_PATH)
    
    torch.manual_seed(args.SEED)
    np.random.seed(args.SEED)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rmse_loss_fn_torch = lambda x, y: torch.sqrt(torch.mean(torch.pow(x-y, 2)))
    
    ACTIVATION = getattr(nn, args.ACTIVATION_NAME)()
    INITIAL_ACTIVATION = getattr(nn, args.INIT_ACTIVATION_NAME)()
    
    if args.LOSS_FUNC == "MSE":
        LOSS_FUNCTION = nn.MSELoss()
    elif args.LOSS_FUNC == "Huber":
        LOSS_FUNCTION = nn.HuberLoss(delta=0.5)
    elif args.LOSS_FUNC == "MAE":
        LOSS_FUNCTION = nn.L1Loss()
    elif args.LOSS_FUNC == "Weighted":
        LOSS_FUNCTION = weighted_mse_loss
    else:
        print("Invalid loss function")
        return
    
    if args.INIT_LOSS_FUNC == "MSE":
        INITIAL_LOSS_FUNCTION = nn.MSELoss()
    elif args.INIT_LOSS_FUNC == "Huber":
        INITIAL_LOSS_FUNCTION = nn.HuberLoss(delta=0.5)
    elif args.INIT_LOSS_FUNC == "MAE":
        INITIAL_LOSS_FUNCTION = nn.L1Loss()
    elif args.INIT_LOSS_FUNC == "Weighted":
        INITIAL_LOSS_FUNCTION = weighted_mse_loss
    else:
        print("Invalid loss function")
        return
    
    NEURON_WIDTHS_LIST = list(map(int, args.NEURON_WIDTHS.split('-')))
    NEURON_DEPTHS_LIST = list(map(int, args.NEURON_DEPTHS.split('-')))
    
    INITIAL_NETWORK_WIDTH = NEURON_WIDTHS_LIST[0]
    INITIAL_NETWORK_DEPTH = NEURON_DEPTHS_LIST[0]
    RES_NETWORK_WIDTHS = NEURON_WIDTHS_LIST[1:]
    RES_NETWORK_DEPTHS = NEURON_DEPTHS_LIST[1:]
    
    x, y = load_data(args.DATA_PATH)
    x_norm, y_norm = normalize_data(x, y, normalize_y=False)
    
    print("Starting training...")
    init_NN, models, epsilons, losses = residual_training(
        x_norm, y_norm, INITIAL_NETWORK_WIDTH, INITIAL_NETWORK_DEPTH, INITIAL_ACTIVATION, INITIAL_LOSS_FUNCTION,
        RES_NETWORK_WIDTHS, RES_NETWORK_DEPTHS, ACTIVATION, LOSS_FUNCTION
    )
    
    error = init_NN(x_norm)
    print("RMSE after initial network: ", rmse_loss_fn_torch(y_norm, error).item())
    print("Inf norm error after initial network: ", torch.max(torch.abs(y_norm - error)).item())

    for i, (e, model) in enumerate(zip(epsilons, models)):
        error += e * model(x_norm)
        print(f"RMSE after adding term {i+1}: {rmse_loss_fn_torch(y_norm, error).item()}")
        print(f"Inf norm error after adding term {i+1}: {torch.max(torch.abs(y_norm - error)).item()}")

    print("Final RMSE: ", rmse_loss_fn_torch(y_norm, error).item())
    print("Final inf norm error: ", torch.max(torch.abs(y_norm - error)).item())

    
    print("Epsilons: ")
    for e in epsilons:
        print(e.item())
    
    save_results(epsilons, init_NN, args.INIT_LOSS_FUNC, args.INIT_ACTIVATION_NAME, models, args.EQUATION_NUMBER, args.LOSS_FUNC,
                 args.ACTIVATION_NAME, args.NEURON_WIDTHS, args.NEURON_DEPTHS, losses, args.SEED)

if __name__ == '__main__':
    main()
