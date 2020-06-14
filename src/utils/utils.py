import torch
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt

def make_mlp(input_dim, hidden_dims, output_dim, activation_fn, output_activation_fn):
    layer_arr = [input_dim] + hidden_dims + [output_dim]
    model = []

    for i in range(len(layer_arr) - 1):
        model += [nn.Linear(layer_arr[i], layer_arr[i+1])]
        if i != len(layer_arr) - 1:
            model += [activation_fn()]

        model += [output_activation_fn()]

    return nn.Sequential(*model)
