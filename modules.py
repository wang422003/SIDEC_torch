import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .utils import initialize_weights

__all__ = ['VDE']


def Layer(i, o, activation=None, p=0., bias=True):
    model = [nn.Linear(i, o, bias=bias)]
    if activation == 'SELU':
        model += [nn.SELU(inplace=True)]
    elif activation == 'RELU':
        model += [nn.ReLU(inplace=True)]
    elif activation == 'LeakyReLU':
        model += [nn.LeakyReLU(inplace=True)]
    elif activation == 'Sigmoid':
        model += [nn.Sigmoid()]
    elif activation == 'Tanh':
        model += [nn.Tanh()]
    elif activation == 'Swish':
        model += [Swish()]
    elif type(activation) is str:
        raise ValueError('{} activation not implemented.'.format(activation))

    if p > 0.:
        model += [nn.Dropout(p)]
    return nn.Sequential(*model)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class Encoder(nn.Module):
    """Encoder network for dimensionality reduction to latent space"""
    def __init__(self, input_size, output_size=1, hidden_layer_depth=5,
                 hidden_size=1024, activation='Swish', dropout_rate=0.):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.input_layer = Layer(input_size, hidden_size,
                                 activation=activation, p=dropout_rate)
        net = [Layer(hidden_size, hidden_size, activation=activation,
                     p=dropout_rate) for _ in range(hidden_layer_depth)]
        self.hidden_network = nn.Sequential(*net)
        self.output_layer = Layer(hidden_size, output_size)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_network(out)
        out = self.output_layer(out)
        return out


class Lambda(nn.Module):
    """Application of Gaussian noise to the latent space"""
    def __init__(self, i=1, o=1, scale=1E-3):
        super(Lambda, self).__init__()

        self.scale = scale
        self.z_mean = nn.Linear(i, o)
        self.z_log_var = nn.Linear(i, o)

    def forward(self, x):
        self.mu = self.z_mean(x)
        self.log_v = self.z_log_var(x)
        eps = self.scale * Variable(torch.randn(*self.log_v.size())
                                    ).type_as(self.log_v)
        return self.mu + torch.exp(self.log_v / 2.) * eps


class Decoder(nn.Module):
    """Decoder network for reconstruction from latent space"""
    def __init__(self, output_size, input_size=1, hidden_layer_depth=5,
                 hidden_size=1024, activation='Swish', dropout_rate=0.):
        super(Decoder, self).__init__()
        self.input_layer = Layer(input_size, input_size, activation=activation)

        net = [Layer(input_size, hidden_size,
                     activation=activation, p=dropout_rate)]
        net += [Layer(hidden_size, hidden_size, activation=activation,
                      p=dropout_rate) for _ in range(hidden_layer_depth)]

        self.hidden_network = nn.Sequential(*net)
        self.output_layer = Layer(hidden_size, output_size)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_network(out)
        out = self.output_layer(out)
        return out


if __name__ == "__main__":
    print("This is modules.py")
