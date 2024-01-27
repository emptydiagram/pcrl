import random

import numpy as np
import torch

def get_activation_fn(name):
    if name == 'relu':
        return torch.relu
    elif name == 'sigmoid':
        return torch.sigmoid
    elif name == 'identity':
        return lambda x: x
    elif name == 'tanh':
        return torch.tanh
    else:
        raise NotImplementedError(f"Activation function {name} not supported.")

def get_activation_fn_deriv(name):
    if name == 'relu':
        return lambda x, y: (x > 0).float()
    elif name == 'sigmoid':
        return lambda x, y: y * (1 - y)
    elif name == 'identity':
        return lambda x, y: torch.ones_like(x)
    elif name == 'tanh':
        return lambda x, y: 1 - y**2
    else:
        raise NotImplementedError(f"Activation function {name} not supported.")

def init_gaussian(dims, stddev, device):
    return torch.empty(dims, requires_grad=False, device=device).normal_(mean=0.0, std=stddev)

def init_uniform(dims, low, high, device):
    return torch.empty(dims, requires_grad=False, device=device).uniform_(low, high)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)