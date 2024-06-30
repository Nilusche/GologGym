# alphazero.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from utils.mcts import GologNode

class AlphaZeroNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(AlphaZeroNet, self).__init__()
        print(input_shape)
        self.conv1 = nn.Conv2d(input_shape[0], 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * input_shape[1] * input_shape[2], 1024)
        self.fc_policy = nn.Linear(1024, num_actions)
        self.fc_value = nn.Linear(1024, 1)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        policy = self.fc_policy(x)
        value = torch.tanh(self.fc_value(x))
        return policy, value
    


class AlphaZeroNode(GologNode):
    def __init__(self, game, parent, done, observation, action_index, policy_net):
        super().__init__(game, parent, done, observation, action_index)
        self.policy_net = policy_net
    
    def create_child(self):
        if self.done:
            return

        legal_actions = self.get_legal_actions()
        games = [deepcopy(self.game) for _ in legal_actions]
        
        state_tensor = torch.tensor(self.observation, dtype=torch.float32).unsqueeze(0)
        policy, _ = self.policy_net(state_tensor)
        policy = policy.detach().numpy().flatten()
        
        child = {}
        for action, game in zip(legal_actions, games):
            observation, reward, _, done, _ = game.step(action)
            action_index = self.get_action_index(action)
            child[tuple(action)] = AlphaZeroNode(game, self, done, observation, action_index, self.policy_net)
        self.child = child

    def get_action_index(self, action):
        # Convert multi-discrete action to a single index for policy
        return np.ravel_multi_index(action, self.game.action_space.nvec)

def AlphaZero_Player_MCTS(root, policy_net, num_simulations=100):
    for _ in range(num_simulations):
        root.explore()
    
    next_tree, next_action = root.next()
    next_tree.detach_parent()
    return next_tree, next_action
   

def train_network(net, memory, optimizer, batch_size=64):
    if len(memory) < batch_size:
        return
    samples = random.sample(memory, batch_size)
    states, policies, values = zip(*samples)
    
    states = torch.tensor(np.array(states), dtype=torch.float32)
    policies = torch.tensor(np.array(policies), dtype=torch.float32)
    values = torch.tensor(np.array(values), dtype=torch.float32)
    
    optimizer.zero_grad()
    out_policies, out_values = net(states)
    loss_policy = nn.CrossEntropyLoss()(out_policies, policies)
    loss_value = nn.MSELoss()(out_values, values)
    loss = loss_policy + loss_value
    loss.backward()
    optimizer.step()

def self_play(env, policy_net, num_games):
    memory = deque(maxlen=10000)
    for _ in range(num_games):
        done = False
        observation, _ = env.reset()
        root = AlphaZeroNode(env, None, False, observation, None, policy_net)
        game_memory = []
        
        while not done:
            root, best_action = AlphaZero_Player_MCTS(root, policy_net)
            observation, reward, _, done, _ = env.step(best_action)
            game_memory.append((observation, best_action))
            env.render()
        
        # Generate training examples
        for obs, action in game_memory:
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            policy, value = policy_net(state_tensor)
            policy = policy.detach().numpy().flatten()
            memory.append((obs, policy, reward))
    
    return memory

def train_alphazero(env, num_iterations, num_games_per_iteration):
    policy_net = AlphaZeroNet(env.observation_space.shape, np.prod(env.action_space.nvec))
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    
    for _ in range(num_iterations):
        memory = self_play(env, policy_net, num_games_per_iteration)
        train_network(policy_net, memory, optimizer)

    