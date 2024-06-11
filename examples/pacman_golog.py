import golog
import time
import numpy as np
from golog.envs.golog_env import GologState, GologAction
import gymnasium as gym
import pygame

# Use fixed seed for np.random
seed = 42
np.random.seed(seed)

# Define the Pac-Man environment
initial_state = GologState()

# Define domains
initial_state.add_symbol('actor', ['pacman', 'dot', 'capsule', 'ghost', 'wall'])
initial_state.add_symbol('location', [(x, y) for x in range(5) for y in range(5)])
initial_state.add_symbol('direction', [(0, 1), (1, 0), (0, -1), (-1, 0)])
initial_state.add_symbol('item', ['dot', 'capsule', 'ghost', 'wall'])

# Define fluents
initial_state.add_fluent('at(pacman)', initial_state.symbols['location'], (0, 0))
initial_state.add_fluent('powered_up', [True, False], False)
initial_state.add_fluent('dots_remaining', list(range(26)), 13)
initial_state.add_fluent('ghosts_remaining', list(range(26)), 1)
initial_state.add_fluent('scared', [True, False], False)

# Initialize food, capsules, ghosts, and walls in specific locations
for location in initial_state.symbols['location']:
    # hard code the location of 13 dots 
    has_dots = location in [(0, 1), (0, 2), (0, 3),(4, 1), (4, 3) , (1, 1), (4, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 3), (2, 4)]
    initial_state.add_fluent(f'at(dot,{location})', [True, False], has_dots)
    initial_state.add_fluent(f'dot_eaten({location})', [True, False], False)
    initial_state.add_fluent(f'at(capsule,{location})', [True, False], location in [(2, 2)])
    initial_state.add_fluent(f'capsule_eaten({location})', [True, False], False)
    initial_state.add_fluent(f'at(ghost,{location})', [True, False], location in [(1, 4)])
    initial_state.add_fluent(f'ghost_eaten({location})', [True, False], False)
    initial_state.add_fluent(f'at(wall,{location})', [True, False], location in [(3, 1), (3, 3),(0, 4), (1, 0), ])

# Define adjacency fluents for Pac-Man to objects
def initialize_adjacency_fluents(state):
    for location in state.symbols['location']:
        state.add_fluent(f'adjacent_to_dot({location})', [True, False], False)
        state.add_fluent(f'adjacent_to_capsule({location})', [True, False], False)
        state.add_fluent(f'adjacent_to_ghost({location})', [True, False], False)
        state.add_fluent(f'adjacent_to_wall({location})', [True, False], False)

initialize_adjacency_fluents(initial_state)

# Update adjacency fluents based on Pac-Man's location
def update_adjacency_fluents(state):
    pacman_loc = state.fluents['at(pacman)'].value
    for loc in state.symbols['location']:
        adj_locs = adjacent_locations(pacman_loc)
        state.fluents[f'adjacent_to_dot({loc})'].set_value(any(state.fluents[f'at(dot,{adj_loc})'].value for adj_loc in adj_locs))
        state.fluents[f'adjacent_to_capsule({loc})'].set_value(any(state.fluents[f'at(capsule,{adj_loc})'].value for adj_loc in adj_locs))
        state.fluents[f'adjacent_to_ghost({loc})'].set_value(any(state.fluents[f'at(ghost,{adj_loc})'].value for adj_loc in adj_locs))
        state.fluents[f'adjacent_to_wall({loc})'].set_value(any(state.fluents[f'at(wall,{adj_loc})'].value for adj_loc in adj_locs))

def adjacent_locations(location):
    x, y = location
    return [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)] if (x + dx, y + dy) in initial_state.symbols['location']]

# Define actions
def move_precondition(state, direction):
    from_loc = state.fluents['at(pacman)'].value
    to_loc = (from_loc[0] + direction[0], from_loc[1] + direction[1])
    if to_loc not in state.symbols['location']:
        return False
    if state.fluents[f'at(wall,{to_loc})'].value:
        return False
    if state.fluents[f'at(ghost,{to_loc})'].value:
        return False
    return True

def move_effect(state, direction):
    from_loc = state.fluents['at(pacman)'].value
    to_loc = (from_loc[0] + direction[0], from_loc[1] + direction[1])
    state.fluents['at(pacman)'].set_value(to_loc)
    state.fluents['scared'].set_value(any(state.fluents[f'adjacent_to_ghost({to_loc})'].value for adj_loc in adjacent_locations(to_loc)))
    update_adjacency_fluents(state)

def eat_precondition(state, item, location):
    return state.fluents['at(pacman)'].value == location and state.fluents[f'at({item},{location})'].value

def eat_effect(state, item, location):
    state.fluents[f'at({item},{location})'].set_value(False)
    if item == 'dot':
        state.fluents[f'dot_eaten({location})'].set_value(True)
        state.fluents['dots_remaining'].set_value(state.fluents['dots_remaining'].value - 1)
    elif item == 'capsule':
        state.fluents[f'capsule_eaten({location})'].set_value(True)
        state.fluents['powered_up'].set_value(True)
    elif item == 'ghost':
        state.fluents[f'ghost_eaten({location})'].set_value(True)
        state.fluents['ghosts_remaining'].set_value(state.fluents['ghosts_remaining'].value - 1)
    update_adjacency_fluents(state)

actions = [
    GologAction('move', move_precondition, move_effect, ['direction']),
    GologAction('eat', eat_precondition, eat_effect, ['item', 'location'])
]

# Define goal
def pacman_goal(state):
    return all(not state.fluents[f'at(dot,{loc})'].value for loc in state.symbols['location']) and \
           all(not state.fluents[f'at(capsule,{loc})'].value for loc in state.symbols['location']) and \
           all(state.fluents[f'dot_eaten({loc})'].value for loc in state.symbols['location'])

# Define reward function
def pacman_reward(state):
    reward = -1  # Base penalty for each step to encourage efficiency

    pacman_loc = state.fluents['at(pacman)'].value

    # Check if all dots are eaten (goal)
    if pacman_goal(state):
        return 100  # Large reward for achieving the goal

    # Reward for eating a dot the first time
    for loc in state.symbols['location']:
        if state.fluents[f'dot_eaten({loc})'].value and pacman_loc == loc and not state.fluents[f'at(dot,{loc})'].value:
            reward += 50  # Increased reward for eating a dot
            state.fluents[f'dot_eaten({loc})'].set_value(False)  # Reset dot eaten status

    # Reward for eating a capsule the first time
    for loc in state.symbols['location']:
        if state.fluents[f'capsule_eaten({loc})'].value and pacman_loc == loc and not state.fluents[f'at(capsule,{loc})'].value:
            reward += 100  # Reward for eating a capsule
            state.fluents[f'capsule_eaten({loc})'].set_value(False)  # Reset capsule eaten status

    # Reward for eating a ghost the first time
    if state.fluents['powered_up'].value:
        for loc in state.symbols['location']:
            if state.fluents[f'ghost_eaten({loc})'].value and pacman_loc == loc and not state.fluents[f'at(ghost,{loc})'].value:
                reward += 200  # Large reward for eating a ghost
                state.fluents[f'ghost_eaten({loc})'].set_value(False)  # Reset ghost eaten status

    # Penalty for encountering a ghost while not powered up
    if not state.fluents['powered_up'].value:
        for loc in state.symbols['location']:
            if state.fluents[f'at(ghost,{loc})'].value and pacman_loc == loc:
                reward -= 1000  # Large penalty for encountering a ghost

    return reward

# Define terminal condition
def terminated(state):
    return pacman_goal(state) or any(state.fluents[f'at(ghost,{loc})'].value and state.fluents['at(pacman)'].value == loc for loc in state.symbols['location'])


#env = gym.make('Golog-v0', initial_state=initial_state, goal_function=pacman_goal, reward_function=pacman_reward, actions=actions, terminal_condition=terminated, time_constraint=100)
env = gym.make('Golog-v2', initial_state=initial_state, goal_function=pacman_goal, reward_function=pacman_reward, actions=actions, terminal_condition=terminated, time_constraint=50)

def render():
    pygame.init()
    screen = pygame.display.set_mode((300, 300))
    pygame.display.set_caption("Pac-Man")

    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    YELLOW = (255, 255, 0)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    ORANGE = (255, 165, 0)
    GREEN = (0, 255, 0)

    # Draw grid
    for x in range(5):
        for y in range(5):
            rect = pygame.Rect(x * 60, y * 60, 60, 60)
            pygame.draw.rect(screen, WHITE, rect, 1)

    # Draw dots, capsules, ghosts, and walls
    for x in range(5):
        for y in range(5):
            loc = (x, y)
            if env.state.fluents[f'at(dot,{loc})'].value:
                pygame.draw.circle(screen, WHITE, (x * 60 + 30, y * 60 + 30), 5)
            if env.state.fluents[f'at(capsule,{loc})'].value:
                pygame.draw.circle(screen, ORANGE, (x * 60 + 30, y * 60 + 30), 5)
            if env.state.fluents[f'at(ghost,{loc})'].value:
                pygame.draw.circle(screen, RED, (x * 60 + 30, y * 60 + 30), 7)
            if env.state.fluents[f'at(wall,{loc})'].value:
                pygame.draw.rect(screen, BLUE, (x * 60 + 10, y * 60 + 10, 40, 40))

    # Draw Pac-Man
    pacman_loc = env.state.fluents['at(pacman)'].value
    pygame.draw.circle(screen, YELLOW, (pacman_loc[0] * 60 + 30, pacman_loc[1] * 60 + 30), 10)

    pygame.display.update()
    time.sleep(0.1)

env.render = render
