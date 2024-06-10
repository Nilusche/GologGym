import gymnasium as gym
from gymnasium import spaces
import copy
from itertools import product
import numpy as np

class GologEnv(gym.Env):
    def __init__(self, initial_state, goal_function, actions, reward_function=None, terminal_condition=None, time_constraint=100):
        super(GologEnv, self).__init__()
        self.initial_state = initial_state
        self.state = copy.deepcopy(initial_state)
        self.goal_function = goal_function
        self.reward_function = reward_function if reward_function else self.default_reward_function
        self.actions = actions
        self.state.actions = actions  # Ensure actions are accessible via state
        self.time_constraint = time_constraint  # Default time constraint
        self.time = 0
        self.terminal_condition = terminal_condition if terminal_condition else lambda state: False
        
        # Generate all valid action-argument combinations
        self.action_arg_combinations = []
        for action_index, action in enumerate(actions):
            valid_args = action.generate_valid_args(self.state)
            for args in valid_args:
                self.action_arg_combinations.append((action_index, args))
        
        self.action_space = spaces.Discrete(len(self.action_arg_combinations))
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.get_observation()),), dtype=np.int32)
        self.done = False
        self.reset()

    def reset(self, seed=None):
        self.done = False
        self.state = copy.deepcopy(self.initial_state)
        self.state.actions = self.actions  # Ensure actions are accessible via state
        self.time = 0
        #create a numeric representation of the initial state
        return self.get_observation(), {}
    
    def get_observation(self):
        observation = []
        for fluent in self.state.fluents:
            value = self.state.fluents[fluent].value
            observation.extend(self._encode_fluent_value(fluent, value))
        return np.array(observation, dtype=np.int32)

    def _encode_fluent_value(self, fluent, value):
        # Convert fluent and value to a numerical representation for all fluents
        encoded = []
        for domain in self.state.symbols.keys():
            if value in self.state.symbols[domain]:
                idx = self.state.symbols[domain].index(value)
                encoded = [0] * len(self.state.symbols[domain])
                encoded[idx] = 1
        
        return encoded

    def step(self, action):
        action_index, args = self.action_arg_combinations[action]
        action = self.state.actions[action_index]
        terminal = False
        truncated = False
        reward = -1
        self.time += 1
        if action.precondition(self.state, *args):
            action.effect(self.state, *args)
            reward = self.reward_function(self.state)
            self.done = self.goal_function(self.state)     
        if self.done:
            terminal = True
            self.done = True
        if self.time >= self.time_constraint:
            terminal = True
            self.done = True
        if self.terminal_condition(self.state):
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
    
    def close(self):
        pass  # No pygame used here

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
        domains = [state.symbols[domain] for domain in self.arg_domains]
        return list(product(*domains))
