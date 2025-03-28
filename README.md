# Residual Training

## Example Usage for Training
    python main.py --SEED 40 --EQUATION_NUMBER I1312 --LOSS_FUNC    MSE --ACTIVATION_NAME Tanh --NEURON_WIDTHS 5-5-5-5 --NEURON_DEPTHS  5-5-5-5 --DATA_PATH ../../Feynman_without_units/I.13.12

## Example Usage for Validation
Modify necessary paths and data, and then run:

    python validation.py
