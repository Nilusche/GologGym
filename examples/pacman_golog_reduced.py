import golog
from golog.envs.golog_env_v3 import GologState, GologAction
import gymnasium as gym
import numpy as np

def move_precondition(state, pacman, direction):
    pos = state.fluents['pos(pacman)'].value
    if direction == 'up':
        new_position = (pos[0] - 1, pos[1])
    elif direction == 'down':
        new_position = (pos[0] + 1, pos[1])
    elif direction == 'left':
        new_position = (pos[0], pos[1] - 1)
    elif direction == 'right':
        new_position = (pos[0], pos[1] + 1)
    
    # Check if the new position is within bounds and not a wall
    if new_position in state.symbols['wall']:
        return False
    if not (0 <= new_position[0] < state.grid_size[0] and 0 <= new_position[1] < state.grid_size[1]):
        return False
    return True

def move_effect(state, pacman, direction):
    pos = state.fluents['pos(pacman)'].value
    if direction == 'up':
        new_position = (pos[0] - 1, pos[1])
    elif direction == 'down':
        new_position = (pos[0] + 1, pos[1])
    elif direction == 'left':
        new_position = (pos[0], pos[1] - 1)
    elif direction == 'right':
        new_position = (pos[0], pos[1] + 1)
    
    state.fluents['pos(pacman)'].set_value(new_position)

def eat_precondition(state, pacman, dot):
    return state.fluents['pos(pacman)'].value == state.fluents[f'pos({dot})'].value

def eat_effect(state, pacman, dot):
    state.fluents[f'dot_eaten({dot})'].set_value(True)

def pacman_goal(state):
    return all(state.fluents[f'dot_eaten({dot})'].value for dot in state.symbols['dot'])

# Initialize the initial state with a dynamic number of dots
def initialize_pacman_state(num_dots):
    initial_state = GologState()
    initial_state.add_symbol('pacman', ['pacman'])
    initial_state.add_symbol('ghost', ['ghost'])
    initial_state.add_symbol('direction', ['up', 'down', 'left', 'right'])
    initial_state.add_symbol('wall', [(1, 1), (1, 2), (2, 1)])
    initial_state.grid_size = (5, 5)  # Assuming a 5x5 grid

    dots = [f'dot{i}' for i in range(1, num_dots + 1)]
    initial_state.add_symbol('dot', dots)

    initial_state.add_fluent('pos(pacman)', [(i, j) for i in range(5) for j in range(5)], (0, 0))
    initial_state.add_fluent('pos(ghost)', [(i, j) for i in range(5) for j in range(5)], (4, 4))

    for i, dot in enumerate(dots):
        initial_state.add_fluent(f'pos({dot})', [(i, j) for i in range(5) for j in range(5)], (i % 5, (i + 2) % 5))
        initial_state.add_fluent(f'dot_eaten({dot})', [True, False], False)

    # Define actions
    move_action = GologAction('move', move_precondition, move_effect, ['pacman', 'direction'])
    eat_action = GologAction('eat', eat_precondition, eat_effect, ['pacman', 'dot'])

    initial_state.add_action(move_action)
    initial_state.add_action(eat_action)

    actions = [move_action, eat_action]

    return initial_state, actions

# Define the reward function
def reward_function(state):
    if pacman_goal(state):
        return 100
    else:
        return -1

# Create the environment
num_dots = 5  # Example number of dots
initial_state, actions = initialize_pacman_state(num_dots)
env = gym.make('Golog-v1', initial_state=initial_state, goal_function=pacman_goal, actions=actions, reward_function=reward_function, time_constraint=50)