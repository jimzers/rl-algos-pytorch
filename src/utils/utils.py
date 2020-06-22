import torch
import torch.nn as nn

import numpy as np
import math

import matplotlib
import matplotlib.pyplot as plt

def make_mlp(layer_arr, activation_fn, output_activation_fn=nn.Identity):
    # layer_arr = [input_dim] + list(hidden_dims) + [output_dim]
    model = []

    for i in range(len(layer_arr) - 1):
        # print(layer_arr)
        # print(layer_arr[i])
        model += [nn.Linear(layer_arr[i], layer_arr[i+1])]
        if i != len(layer_arr) - 2:
            model += [activation_fn()]

    model += [output_activation_fn()]

    return nn.Sequential(*model)


def init_weights(layer):
    if type(layer) == nn.Linear:
        f = 1 / np.sqrt(layer.weight.data.size()[0])
        torch.nn.init.uniform_(layer.weight.data, -f, f)
        torch.nn.init.uniform_(layer.bias.data, -f, f)


def init_xav_weights(layer):
    if type(layer) == nn.Linear:
        f_xavier = math.sqrt(6.0 / (layer.weight.data.size(1) + layer.weight.data.size(0)))
        torch.nn.init.uniform_(layer.weight.data, -f_xavier, f_xavier)
        torch.nn.init.uniform_(layer.bias.data, -f_xavier, f_xavier)