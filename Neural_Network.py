# ---------------------- Import Required Libraries --------------------

import torch 

import torch.nn as nn

import torch.nn.functional as F

# ---------------------- Defining the Network Class --------------------

class Network(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(Network, self).__init__()
        # Define the layers
        self.linear = nn.Linear(dim_input, dim_output)
    
    def forward(self, x):
        return self.linear(x)
    
# ---------------------- Defining the Q-Network Class --------------------

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):

        super(QNetwork, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
    
    def forward(self, state):
        
        x = torch.relu(self.fc1(torch.tensor(state)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  