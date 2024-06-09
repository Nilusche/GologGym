import gym
from gym import spaces
import copy
from itertools import product
import pygame

class GologEnv(gym.Env):
    def __init__(self, initial_state, goal_function, actions, reward_function=None):
        super(GologEnv, self).__init__()
        self.initial_state = initial_state
        self.state = copy.deepcopy(initial_state)
        self.goal_function = goal_function
        self.reward_function = reward_function if reward_function else self.default_reward_function
        self.actions = actions
        self.state.actions = actions  # Ensure actions are accessible via state
        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = spaces.Discrete(len(initial_state.fluents))
        self.done = False
        self.reset()

    def reset(self):
        self.done = False
        self.state = copy.deepcopy(self.initial_state)
        self.state.actions = self.actions  # Ensure actions are accessible via state
        return self.get_observation()
    
    def get_observation(self):
        observation = {}
        for fluent in self.state.fluents:
            observation[fluent] = self.state.fluents[fluent].value
        return observation

    def step(self, action):
        action_index, args = action
        action = self.state.actions[action_index]
        if action.precondition(self.state, *args):
            action.effect(self.state, *args)
            reward = self.reward_function(self.state)
            self.done = self.goal_function(self.state)
            return self.get_observation(), reward, self.done, {}
        else:
            return self.get_observation(), -1, self.done, {}

    def default_reward_function(self, state):
        return 100 if self.goal_function(state) else -1

    def render(self, mode='human'):
        state_description = []
        for fluent, value in self.state.fluents.items():
            state_description.append(f"{fluent} is {value.value}")
        print("Current State:")
        print("\n".join(state_description))

    def generate_valid_args(self, action_index):
        action = self.actions[action_index]
        return action.generate_valid_args(self.state)
    
    def close(self):
        pygame.quit()
