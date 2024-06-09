import torch
import torch.nn as nn
import torch.optim as optim
import random
from math import sqrt
from examples.blocksworld_golog import env
from copy import deepcopy

class AlphaZeroNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(AlphaZeroNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = torch.softmax(self.policy_head(x), dim=-1)
        value = torch.tanh(self.value_head(x))
        return policy, value

# Assuming the state observation is represented as a 1D tensor and the action space is discrete
observation = env.reset()
input_dim = len(observation)
action_dim = env.action_space.n
policy_value_net = AlphaZeroNetwork(input_dim, action_dim)
optimizer = optim.Adam(policy_value_net.parameters(), lr=0.001)

class AlphaZeroNode:
    def __init__(self, game, parent, done, observation, action_index):
        self.child = {}
        self.T = 0
        self.N = 0
        self.P = 0  # Prior probability
        self.Q = 0  # Mean action value
        self.game = game
        self.observation = observation
        self.done = done
        self.parent = parent
        self.action_index = action_index

    def get_UCB_score(self, c_puct=1.0):
        if self.N == 0:
            return float('inf')
        return self.Q + c_puct * self.P * sqrt(self.parent.N) / (1 + self.N)

    def detach_parent(self):
        del self.parent
        self.parent = None

    def create_child(self, policy_value_net):
        if self.done:
            return

        observation_tensor = torch.tensor(self.observation, dtype=torch.float32).unsqueeze(0)
        policy, value = policy_value_net(observation_tensor)
        actions = list(range(self.game.action_space.n))

        for action, p in zip(actions, policy.detach().numpy()[0]):
            new_game = deepcopy(self.game)
            obs, reward, done, _ = new_game.step(action)
            child = AlphaZeroNode(new_game, self, done, obs, action)
            child.P = p
            self.child[action] = child

        self.V = value.item()

    def explore(self, policy_value_net):
        current = self
        while current.child:
            max_UCB = max(child.get_UCB_score() for child in current.child.values())
            best_actions = [action for action, child in current.child.items() if child.get_UCB_score() == max_UCB]
            if not best_actions:
                print("Error: zero length ", max_UCB)
            action = random.choice(best_actions)
            current = current.child[action]

        if current.N == 0:
            current.T += current.rollout(policy_value_net)
        else:
            current.create_child(policy_value_net)
            if current.child:
                current = random.choice(list(current.child.values()))
            current.T += current.rollout(policy_value_net)

        current.N += 1
        parent = current
        while parent.parent:
            parent = parent.parent
            parent.N += 1
            parent.T += current.T

    def rollout(self, policy_value_net):
        if self.done:
            return 0

        v = 0
        done = False
        new_game = deepcopy(self.game)
        rollout_steps = 0

        while not done and rollout_steps < 100:
            observation_tensor = torch.tensor(new_game.get_observation(), dtype=torch.float32).unsqueeze(0)
            policy, value = policy_value_net(observation_tensor)
            action_probabilities = policy.detach().numpy()[0]
            
            # Sample an action according to the policy network's probabilities
            action_index = random.choices(range(new_game.action_space.n), weights=action_probabilities, k=1)[0]
            
            obs, reward, done, _ = new_game.step(action_index)
            v += reward
            rollout_steps += 1
            if done:
                new_game.reset()
                new_game.close()
                break

        return v

    def next(self):
        if self.done:
            raise ValueError("game has ended")

        if not self.child:
            raise ValueError('no children found and game hasn\'t ended')

        max_N = max(child.N for child in self.child.values())
        best_children = [child for child in self.child.values() if child.N == max_N]
        return random.choice(best_children), random.choice(list(self.child.keys()))

def Policy_Player_AlphaZero(mytree, policy_value_net, num_simulations):
    for _ in range(num_simulations):
        mytree.explore(policy_value_net)

    next_tree, next_action = mytree.next()
    next_tree.detach_parent()
    return next_tree, next_action


import random
from collections import deque

def self_play(env, policy_value_net, num_simulations):
    observation = env.reset()
    done = False
    game_data = []
    mytree = AlphaZeroNode(env, None, False, observation, 0)
    while not done:
        mytree, action_index = Policy_Player_AlphaZero(mytree, policy_value_net, num_simulations)
        #check action_index for preconditions
        action, args = env.action_arg_combinations[action_index]
        reward = -1
        if not env.state.actions[action].precondition(env.state, *args):
            continue
        observation, reward, done, _ = env.step(action_index)
        action_probs = [child.N / mytree.N for child in mytree.child.values()]
        game_data.append((observation, action_probs, None))  # Save state and MCTS probs
    winner = 1 if reward > 0 else -1
    for i in range(len(game_data)):
        game_data[i] = (game_data[i][0], game_data[i][1], winner)
    return game_data

def train(policy_value_net, optimizer, data):
    policy_value_net.train()
    for state, mcts_probs, winner in data:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        mcts_probs_tensor = torch.tensor(mcts_probs, dtype=torch.float32).unsqueeze(0)
        winner_tensor = torch.tensor([winner], dtype=torch.float32).unsqueeze(0)

        optimizer.zero_grad()
        policy, value = policy_value_net(state_tensor)
        value_loss = nn.MSELoss()(value, winner_tensor)
        policy_loss = nn.CrossEntropyLoss()(policy, mcts_probs_tensor)
        loss = value_loss + policy_loss
        loss.backward()
        optimizer.step()

def main():
    num_episodes = 5  # Number of self-play episodes
    num_simulations = 100  # Number of MCTS simulations per move
    batch_size = 32  # Training batch size

    replay_buffer = deque(maxlen=10000)
    for episode in range(num_episodes):
        game_data = self_play(env, policy_value_net, num_simulations)
        replay_buffer.extend(game_data)

        if len(replay_buffer) >= batch_size:
            mini_batch = random.sample(replay_buffer, batch_size)
            train(policy_value_net, optimizer, mini_batch)

        print(f"Episode {episode + 1}/{num_episodes} completed")

    # Evaluate the trained network
    reward_e = 0
    observation = env.reset()
    done = False
    new_game = deepcopy(env)
    mytree = AlphaZeroNode(new_game, None, False, observation, 0)
    step_counter = 0

    while not done:
        mytree, action_index = Policy_Player_AlphaZero(mytree, policy_value_net, num_simulations)
        observation, reward, done, _ = env.step(action_index)
        reward_e += reward
        action = env.action_arg_combinations[action_index]
        print(f"Step {step_counter}: Executing action: {env.state.actions[action[0]].name} with args {action[1:]}")

        env.render()

        if done:
            print('reward_e ' + str(reward_e))
            print("Game over!")
            break

        step_counter += 1

if __name__ == "__main__":
    main()