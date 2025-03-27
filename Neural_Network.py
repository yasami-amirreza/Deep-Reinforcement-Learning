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
    def __init__(self, s, dim_output):

        super(QNetwork, self).__init__()
        # Define the layers
        self.linear = nn.Linear(dim_input, dim_output)
    
    def forward(self, x):
        return self.linear(x)    