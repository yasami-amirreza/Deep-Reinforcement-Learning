# ---------------------- Import Required Libraries --------------------

import gym

import importlib

import Neural_Network

importlib.reload(Neural_Network)

from Neural_Network import Network

import torch.optim as optim

# ---------------------- Create the Environment --------------------

# Create the fozen lake environment
env = gym.make("FrozenLake-v1")

# reset the intial state of the environment
state = env.reset()

# Take an action
state, reward, done, info = env.step(env.action_space.sample())

# render the environment
env.render()

env.close()

# ---------------------- Create the Network --------------------

DI = env.state_space.shape()

DO = env.action_space.shape()

network = Network(dim_input=DI, dim_output=DO)

optimizer  = optim.Adam(network.parameters(), lr=0.0001)