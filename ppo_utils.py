import torch
from torch import nn
from torch.distributions import Normal, Independent
import numpy as np
import torch


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, env, hidden_size=32):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space


        # implement the rest
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_space, hidden_size)), nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_space), std=0.01),
        )
        
        log_std =  np.ones(action_space, dtype=np.float32)
        self.actor_logstd = torch.nn.Parameter(torch.as_tensor(log_std))
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_space, hidden_size)), nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1)))


    def forward(self, state):
        # Get mean of a Normal distribution (the output of the neural network)
        action_mean = self.actor_mean(state)

        # Make sure action_logstd matches dimensions of action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)

        # Exponentiate the log std to get actual std
        action_std = torch.exp(action_logstd)

        return Normal(loc=action_mean, scale=action_std), self.critic(state).squeeze(-1)
