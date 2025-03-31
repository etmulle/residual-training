# Residual Training

## Package requirements

Required packages are in requirements.txt. You can install them by navigating to the code directory and typing

    pip install -r requirements.txt

## Example Usage for Training
    python main.py --SEED 40 --EQUATION_NUMBER I1312 --LOSS_FUNC    MSE --ACTIVATION_NAME Tanh --NEURON_WIDTHS 5-5-5-5 --NEURON_DEPTHS  5-5-5-5 --DATA_PATH ../../Feynman_without_units/I.13.12
    
## Command-Line Arguments Summary

Below is a list of command-line arguments along with their descriptions:

| Argument                 | Type  | Default Value                     | Description |
|--------------------------|------|---------------------------------|-------------|
| `--SEED`                | `int`  | `39`                            | Random seed for reproducibility. |
| `--EQUATION_NUMBER`      | `str`  | `'I918'`                        | Identifier for the equation being used. |
| `--INIT_LOSS_FUNC`       | `str`  | `'MSE'`                          | Loss function for the initial network. |
| `--INIT_ACTIVATION_NAME` | `str`  | `'Tanh'`                         | Activation function for the initial network. |
| `--LOSS_FUNC`           | `str`  | `'MSE'`                          | Loss function to be used in training the residual networks. |
| `--ACTIVATION_NAME`     | `str`  | `'Tanh'`                         | Activation function to be used in the residual networks. |
| `--NEURON_WIDTHS`       | `str`  | `'5-5-5-5'`                      | Defines the widths (number of neurons per layer) of each of the networks. 5-5-5-5 means one initial network and three residual networks which each have 5 neurons in each hidden layer. |
| `--NEURON_DEPTHS`       | `str`  | `'5-5-5-5'`                      | Defines the depths of each of the neural networks. 5-5-5-5 means one initial network and three residual networks which each have 5 hidden layers.|
| `--DATA_PATH`           | `str`  | `'../../Feynman_without_units/I.9.18'` | Path to the training dataset. |

These arguments allow customization of the neural networks' structures, training settings, and dataset location.

## Example Usage for Validation
Modify necessary paths and data, and then run:

    python validation.py
