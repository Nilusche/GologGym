import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
from collections import namedtuple, deque
from copy import deepcopy
from itertools import product
import random
from math import sqrt, log
import matplotlib.pyplot as plt
from examples.blocksworld_golog import env

observation_space_size = env.observation_space.shape[0]
action_space_size = env.action_space.n
c = 1.4


# Neural Network definitions
class PolicyV(keras.Model):
    def __init__(self, observation_space_size):
        super(PolicyV, self).__init__()
        self.dense1 = keras.layers.Dense(64, activation='relu')
        self.dense2 = keras.layers.Dense(64, activation='relu')
        self.v_out = keras.layers.Dense(1)

    def call(self, input):
        x = self.dense1(input)
        x = self.dense2(x)
        return self.v_out(x)

class PolicyP(keras.Model):
    def __init__(self, action_space_size):
        super(PolicyP, self).__init__()
        self.dense1 = keras.layers.Dense(64, activation='relu')
        self.dense2 = keras.layers.Dense(64, activation='relu')
        self.p_out = keras.layers.Dense(action_space_size, activation='softmax')

    def call(self, input):
        x = self.dense1(input)
        x = self.dense2(x)
        return self.p_out(x)

class Node:
    def __init__(self, game, done, parent, observation, action_index, action_space_size):
        self.child = None
        self.T = 0
        self.N = 0
        self.game = game
        self.observation = observation
        self.done = done
        self.parent = parent
        self.action_index = action_index
        self.nn_v = 0
        self.nn_p = np.zeros(action_space_size)

    def getUCBscore(self):
        if self.N == 0:
            return float('inf')
        top_node = self
        if top_node.parent:
            top_node = top_node.parent
        value_score = (self.T / self.N)
        prior_score = c * self.parent.nn_p[self.action_index] * sqrt(log(top_node.N) / self.N)
        return value_score + prior_score

    def detach_parent(self):
        del self.parent
        self.parent = None

    def create_child(self):
        if self.done:
            return
        actions = list(range(action_space_size))
        games = [deepcopy(self.game) for _ in actions]
        child = {}
        action_index = 0
        for action, game in zip(actions, games):
            observation, reward, _, done, _ = game.step(action)
            child[action] = Node(game, done, self, observation, action_index, action_space_size)
            action_index += 1
        self.child = child

    def explore(self):
        current = self
        while current.child:
            child = current.child
            max_U = max(c.getUCBscore() for c in child.values())
            actions = [a for a, c in child.items() if c.getUCBscore() == max_U]
            action = random.choice(actions)
            current = child[action]
        if current.N < 1:
            current.nn_v, current.nn_p = current.rollout()
            current.T = current.T + current.nn_v
        else:
            current.create_child()
            if current.child:
                current = random.choice(current.child)
            current.nn_v, current.nn_p = current.rollout()
            current.T = current.T + current.nn_v
        current.N += 1
        parent = current
        while parent.parent:
            parent = parent.parent
            parent.N += 1
            parent.T = parent.T + current.T

    def rollout(self):
        if self.done:
            return 0, np.zeros(action_space_size)
        else:
            obs = np.array([self.observation])
            v = policy_v(obs)
            p = policy_p(obs)
            return v.numpy().flatten()[0], p.numpy().flatten()

    def next(self):
        if self.done:
            raise ValueError("game has ended")
        if not self.child:
            raise ValueError("no children found and game hasn't ended")
        child = self.child
        max_N = max(node.N for node in child.values())
        probs = [node.N / max_N for node in child.values()]
        probs /= np.sum(probs)
        next_children = random.choices(list(child.values()), weights=probs)[0]
        return next_children, next_children.action_index, next_children.observation, probs, self.observation

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["obs", "v", "p_obs", "p"])

    def add(self, obs, v, p, p_obs):
        e = self.experience(obs, v, p, p_obs)
        self.memory.append(e)

    def sample(self):
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        return len(self.memory)

def Policy_Player_MCTS(mytree):
    for i in range(200):
        mytree.explore()
    next_tree, next_action, obs, p, p_obs = mytree.next()
    next_tree.detach_parent()
    return next_tree, next_action, obs, p, p_obs



BUFFER_SIZE = 1000
BATCH_SIZE = 128
UPDATE_EVERY = 1
episodes = 250
1
rewards = []
moving_average = []
v_losses = []
p_losses = []

MAX_REWARD = 500

replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

policy_v = PolicyV(observation_space_size)
policy_v.compile(optimizer=keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanSquaredError()])

policy_p = PolicyP(action_space_size)
policy_p.compile(optimizer=keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalCrossentropy()])

for e in range(episodes):
    reward_e = 0
    game = env
    observation, _ = game.reset()
    done = False
    new_game = deepcopy(game)
    mytree = Node(new_game, False, None, observation, 0, action_space_size)

    obs = []
    ps = []
    p_obs = []
    step = 0

    while not done:
        step += 1
        mytree, action, ob, p, p_ob = Policy_Player_MCTS(mytree)
        obs.append(ob)
        ps.append(p)
        p_obs.append(p_ob)
        _, reward, _, done, _ = game.step(action)
        reward_e += reward

        if done:
            for i in range(len(obs)):
                replay_buffer.add(obs[i], reward_e, p_obs[i], ps[i])
            game.close()
            break

    rewards.append(reward_e)
    moving_average.append(np.mean(rewards[-100:]))

    if (e + 1) % UPDATE_EVERY == 0 and len(replay_buffer) > BATCH_SIZE:
        experiences = replay_buffer.sample()
        inputs = np.array([experience.obs for experience in experiences])
        targets_v = np.array([experience.v / MAX_REWARD for experience in experiences])
        loss_v = policy_v.train_on_batch(inputs, targets_v)
        v_losses.append(loss_v)

        inputs = np.array([experience.p_obs for experience in experiences])
        targets_p = np.array([experience.p for experience in experiences])
        loss_p = policy_p.train_on_batch(inputs, targets_p)
        p_losses.append(loss_p)

        # Plotting
        plt.plot(rewards)
        plt.plot(moving_average)
        plt.show()
        plt.plot(v_losses)
        plt.show()
        plt.plot(p_losses)
        plt.show()
        print('moving average:', np.mean(rewards[-20:]))