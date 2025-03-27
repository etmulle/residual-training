import itertools
import subprocess
from multiprocessing import Pool

# Generate all parameter combinations
seeds = [40, 41, 42]
equation_numbers = ["I1312", "I262", "I2916", "I62", "I918", "III952"]
equation_strings = {"I1312":"I.13.12", "I262":"I.26.2", "I2926":"I.29.16", "I62":"I.6.2", "I918":"I.9.18", "III952":"III.9.52"}
loss_funcs = ["MSE", "MAE", "Huber", "Weighted"]
activations = ["Tanh", "ReLU", "SiLU"]
neuron_widths = ["5-5-5-5", "5-10-15-20", "20-20-20-20"]
neuron_depths = ["5-5-5-5"]

# Generate all possible combinations
param_combinations = list(itertools.product(seeds, equation_numbers, loss_funcs, activations, neuron_widths, neuron_depths))

# Loop through each combination and run main.py
for params in param_combinations:
    seed, eq_num, loss_func, activation, widths, depths = params
    data_path = "../../Feynman_without_units/" + str(equation_strings[eq_num])
    cmd = [
        "python", "main.py",
        "--SEED", str(seed),
        "--EQUATION_NUMBER", eq_num,
        "--LOSS_FUNC", loss_func,
        "--ACTIVATION_NAME", activation,
        "--NEURON_WIDTHS", widths,
        "--NEURON_DEPTHS", depths,
        "--DATA_PATH", data_path
    ]

    print(f"Running: {' '.join(cmd)}")  # Print the command for reference
    subprocess.run(cmd)  # Run the script