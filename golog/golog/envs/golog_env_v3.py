import gymnasium as gym
from gymnasium import spaces
import copy
from itertools import product
import numpy as np
from typing import List

class GologEnv_v3(gym.Env):
    def __init__(self, initial_state, goal_function, actions, reward_function=None, terminal_condition=None, time_constraint=np.inf):
        super(GologEnv_v3, self).__init__()
        self.initial_state = initial_state
        self.state = copy.deepcopy(self.initial_state)
        self.goal_function = goal_function
        self.reward_function = reward_function if reward_function else self.default_reward_function
        self.actions = actions
        self.state.actions = actions
        self.time_constraint = time_constraint
        self.time = 0
        self.terminal_condition = terminal_condition if terminal_condition else lambda state: False
        
        # Define action space
        action_spaces = [spaces.Discrete(len(actions))]
        for action in actions:
            for arg_domain in action.arg_domains:
                action_spaces.append(spaces.Discrete(len(initial_state.symbols[arg_domain])))
        self.action_space = spaces.MultiDiscrete([space.n for space in action_spaces])
        
        # Calculate the maximum domain size for fluents
        self.max_domain_size = max(len(domain) for domain in initial_state.symbols.values())
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.calculate_observation_space_size(), self.max_domain_size), dtype=np.int32)
        self.done = False
        self.reset()

    def reset(self, seed=None, options=None):
        self.done = False
        self.state = copy.deepcopy(self.initial_state)
        self.state.actions = self.actions
        self.time = 0
        return self.get_observation(), {}

    def get_observation(self):
        observation = np.zeros((self.calculate_observation_space_size(), self.max_domain_size), dtype=np.int32)
        index = 0
        for fluent in self.state.fluents:
            value = self.state.fluents[fluent].value
            encoded = self._encode_fluent_value(fluent, value)
            observation[index, :len(encoded)] = encoded
            index += 1
        return observation

    def _encode_fluent_value(self, fluent, value):
        domain = self.state.fluents[fluent].domain
        encoded = np.zeros(self.max_domain_size, dtype=np.int32)
        if value in domain:
            idx = domain.index(value)
            encoded[idx] = 1
        return encoded
    
    def calculate_observation_space_size(self):
        return len(self.state.fluents)

    def step(self, action):
        action_index = action[0]
        args = action[1:]
        action_obj = self.state.actions[action_index]
        arg_values = [self.state.symbols[domain][arg] for domain, arg in zip(action_obj.arg_domains, args)]
        terminal = False
        reward = 0
        self.time += 1
        if action_obj.precondition(self.state, *arg_values):
            action_obj.effect(self.state, *arg_values)
            reward = self.reward_function(self.state)
            self.done = self.goal_function(self.state)
        if self.done or self.time >= self.time_constraint or self.terminal_condition(self.state):
            terminal = True
            self.done = True

        return self.get_observation(), reward, terminal, self.done, {}

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
    
    def action_masks(self) -> List[bool]:
        total_actions = sum(self.action_space.nvec)
        possible_actions = np.arange(total_actions)
        invalid_actions = []

        for action_index in range(self.action_space.nvec[0]):
            for args_combination in product(*(range(n) for n in self.action_space.nvec[1:])):
                action_combination = [action_index] + list(args_combination)
                action_obj = self.state.actions[action_index]
                arg_values = [self.state.symbols[domain][arg] for domain, arg in zip(action_obj.arg_domains, args_combination)]
                if not action_obj.precondition(self.state, *arg_values):
                    flat_index = np.ravel_multi_index(tuple(action_combination), self.action_space.nvec)
                    invalid_actions.append(flat_index)

        return [action not in invalid_actions for action in possible_actions]

    def close(self):
        pass

class GologFluent:
    def __init__(self, domain, value):
        self.domain = domain
        self.value = value

    def set_value(self, value):
        if value in self.domain:
            self.value = value

    def __repr__(self):
        return str(self.value)

class GologState:
    def __init__(self):
        self.symbols = {}
        self.actions = []
        self.fluents = {}

    def add_symbol(self, symbol, domain):
        self.symbols[symbol] = domain
    
    def add_fluent(self, fluent, domain, initial_value):
        self.fluents[fluent] = GologFluent(domain, initial_value)

    def add_action(self, action):
        self.actions.append(action)
    
    def execute_action(self, action_index, *args):
        action = self.actions[action_index]
        if action.precondition(self, *args):
            action.effect(self, *args)
            return True
        return False

    def __hash__(self):
        return hash(frozenset((fluent, fl.value) for fluent, fl in self.fluents.items()))

    def __eq__(self, other):
        return all(self.fluents[fluent].value == other.fluents[fluent].value for fluent in self.fluents)

class GologAction:
    def __init__(self, name, precondition, effect, arg_domains):
        self.name = name
        self.precondition = precondition
        self.effect = effect
        self.arg_domains = arg_domains

    def generate_valid_args(self, state):
        try:
            domains = [state.symbols[domain] for domain in self.arg_domains]
            return list(product(*domains))
        except KeyError as e:
            print(f"Key error: {e}")
            return []
