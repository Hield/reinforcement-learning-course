import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.distributions import Categorical
import numpy as np
import  torch

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, env, hidden_size=32):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        
        print(action_space)
        print(state_space)
        
        self.fc1_a = torch.nn.Linear(state_space, hidden_size)
        self.fc2_a = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3_a = torch.nn.Linear(hidden_size, action_space)

        self.fc1_c = torch.nn.Linear(state_space, hidden_size)
        self.fc2_c = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3_c = torch.nn.Linear(hidden_size, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight, 0, 1e-1)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        
        
        
        x_a = self.fc1_a(x)
        x_a = F.relu(x_a)
        x_a = self.fc2_a(x_a)
        x_a = F.relu(x_a)
        x_a = self.fc3_a(x_a)

        x_c = self.fc1_c(x)
        x_c = F.relu(x_c)
        x_c = self.fc2_c(x_c)
        x_c = F.relu(x_c)
        x_c = self.fc3_c(x_c)

        print(x_a)
        action_probs = F.softmax(x_a, dim=-1)
        action_dist = Categorical(action_probs)

        return x_a, x_c


        # implement the rest
