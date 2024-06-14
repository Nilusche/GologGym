import gymnasium as gym
from gymnasium import spaces
import copy
from itertools import product
import numpy as np

class GologEnv_v2(gym.Env):
    def __init__(self, initial_state, goal_function, actions, reward_function=None, terminal_condition=None, time_constraint=np.inf):
        super(GologEnv_v2, self).__init__()
        self.initial_state = initial_state
        self.state = copy.deepcopy(initial_state)
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
        self.fluent_domain_sizes = {fluent: len(domain) for fluent, domain in initial_state.fluents.items()}
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.calculate_observation_space_size(), max(self.fluent_domain_sizes.values())), dtype=np.int32)
        self.done = False
        self.reset()

    def reset(self, seed=None, options=None):
        self.done = False
        self.state = copy.deepcopy(self.initial_state)
        self.state.actions = self.actions
        self.time = 0
        return self.get_observation(), {}

    def get_observation(self):
        max_domain_size = max(self.fluent_domain_sizes.values())
        observation = np.zeros((self.calculate_observation_space_size(), max_domain_size), dtype=np.int32)
        index = 0
        for fluent in self.state.fluents:
            value = self.state.fluents[fluent].value
            encoded = self._encode_fluent_value(fluent, value)
            observation[index, :len(encoded)] = encoded
            index += 1
        return observation

    def _encode_fluent_value(self, fluent, value):
        domain = self.state.fluents[fluent].domain
        encoded = np.zeros(self.fluent_domain_sizes[fluent], dtype=np.int32)
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
        reward = -100
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
    
    def __len__(self):
        return len(self.domain)

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
