# ---------------------- Import Required Libraries --------------------

import gym

import importlib

import Neural_Network

importlib.reload(Neural_Network)

from Neural_Network import Network

import torch.optim as optim

# ---------------------- Create the Environment --------------------

# Create the fozen lake environment

env = gym.make("LunarLander-v2")

# reset the intial state of the environment
state = env.reset()

# Take an action
state, reward, done, info = env.step(env.action_space.sample())

# render the environment
env.render()

env.close()

# ---------------------- Create the Network --------------------

DI = 8

DO = 4

network = Network(dim_input=DI, dim_output=DO)

optimizer  = optim.Adam(network.parameters(), lr=0.0001)

# ---------------------- Learning the Policy --------------------

num_episodes = 1000

for episode in range(1, num_episodes):
    
    state, reward, done, info = env.reset()
    
    done = False
    
    while not done:
        
        action = select_action(network, state)
        
        next_state, reward, terminate, truncate, _ = (env.step(action))
        
        done = terminate or truncate
        
        loss = calculate_loss(network, state, action, next_state, reward, done)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        state = next_state
        
        
        