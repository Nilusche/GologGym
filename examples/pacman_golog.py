import golog
import time
from golog.envs.golog_env import GologState, GologAction
import gymnasium as gym
import pygame



# Define the Pac-Man environment
initial_state = GologState()

# Define domains
initial_state.add_symbol('actor', ['pacman', 'dot', 'ghost'])
initial_state.add_symbol('location', [(x, y) for x in range(3) for y in range(3)])
initial_state.add_symbol('direction', [(0, 1), (1, 0), (0, -1), (-1, 0)])

# Define fluents
initial_state.add_fluent('at(pacman)', initial_state.symbols['location'], (0, 0))
initial_state.add_fluent('scared', [True, False], False)
initial_state.add_fluent('nearest_dot_distance', list(range(9)), 8)  # Initial distance to the nearest dot
initial_state.add_fluent('dots_remaining', list(range(10)), 3)  # Initial number of dots
initial_state.add_fluent('nearest_ghost_distance', list(range(9)), 8)  # Initial distance to the nearest ghost

for location in initial_state.symbols['location']:
    #add food to every location except the starting location and the ghost location in a 3x3 grid
    has_dot = location != (0, 0) and location != (2, 2)
    initial_state.add_fluent(f'at(dot,{location})', [True, False], has_dot)
    initial_state.add_fluent(f'dot_eaten({location})', [True, False], False)
    initial_state.add_fluent(f'at(ghost,{location})', [True, False], location == (2, 2))

# Helper function to check adjacency
def adjacent(loc1, loc2):
    x1, y1 = loc1
    x2, y2 = loc2
    return abs(x1 - x2) + abs(y1 - y2) == 1

# Helper function to calculate Manhattan distance
def manhattan_distance(loc1, loc2):
    return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

# Define the primitive actions
def move_direction_precondition(state, direction):
    from_loc = state.fluents['at(pacman)'].value
    to_loc = (from_loc[0] + direction[0], from_loc[1] + direction[1])
    return from_loc == state.fluents['at(pacman)'].value and to_loc in state.symbols['location']

def move_direction_effect(state, direction):
    from_loc = state.fluents['at(pacman)'].value
    to_loc = (from_loc[0] + direction[0], from_loc[1] + direction[1])
    state.fluents['at(pacman)'].set_value(to_loc)

    # Update scared fluent
    state.fluents['scared'].set_value(any(adjacent(to_loc, loc) for loc in state.symbols['location'] if state.fluents[f'at(ghost,{loc})'].value))
    
    # Update nearest dot distance
    nearest_dot_distance = min(manhattan_distance(to_loc, loc) for loc in state.symbols['location'] if state.fluents[f'at(dot,{loc})'].value)
    state.fluents['nearest_dot_distance'].set_value(nearest_dot_distance)
    
    # Update nearest ghost distance
    nearest_ghost_distance = min(manhattan_distance(to_loc, loc) for loc in state.symbols['location'] if state.fluents[f'at(ghost,{loc})'].value)
    state.fluents['nearest_ghost_distance'].set_value(nearest_ghost_distance)

def eat_dot_precondition(state, location):
    return state.fluents['at(pacman)'].value == location and state.fluents[f'at(dot,{location})'].value

def eat_dot_effect(state, location):
    state.fluents[f'at(dot,{location})'].set_value(False)
    state.fluents[f'dot_eaten({location})'].set_value(True)
    state.fluents['dots_remaining'].set_value(state.fluents['dots_remaining'].value - 1)

# Add actions to the environment
actions = [
    GologAction('move_direction', move_direction_precondition, move_direction_effect, ['direction']),
    GologAction('eat_dot', eat_dot_precondition, eat_dot_effect, ['location']),
]

# Define goal
def pacman_goal(state):
    return all(not state.fluents[f'at(dot,{loc})'].value for loc in state.symbols['location'])

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

    # Penalty for encountering a ghost
    for loc in state.symbols['location']:
        if state.fluents[f'at(ghost,{loc})'].value and pacman_loc == loc:
            reward -= 1000  # Large penalty for encountering a ghost

    # Encourage movement towards dots by penalizing distance to the nearest dot
    nearest_dot_distance = state.fluents['nearest_dot_distance'].value
    reward -= nearest_dot_distance  # Penalize for distance to nearest dot

    return reward

def terminated(state):
    return pacman_goal(state) or any(state.fluents[f'at(ghost,{loc})'].value and state.fluents['at(pacman)'].value == loc for loc in state.symbols['location'])

env = gym.make('Golog-v0', initial_state=initial_state, goal_function=pacman_goal, reward_function=pacman_reward, actions=actions, terminal_condition=terminated, time_constraint=30)


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

    # Grid dimensions
    grid_size = 100
    margin = 5

    screen.fill(BLACK)

    # Draw the grid
    for row in range(3):
        for col in range(3):
            color = WHITE
            pygame.draw.rect(screen,
                            color,
                            [(margin + grid_size) * col + margin,
                            (margin + grid_size) * row + margin,
                            grid_size,
                            grid_size])

    # Draw dots
    for loc in initial_state.symbols['location']:
        if env.state.fluents[f'at(dot,{loc})'].value:
            pygame.draw.circle(screen,
                            BLUE,
                            [(margin + grid_size) * loc[0] + margin + grid_size // 2,
                                (margin + grid_size) * loc[1] + margin + grid_size // 2],
                            10)

    # Draw ghosts
    for loc in initial_state.symbols['location']:
        if env.state.fluents[f'at(ghost,{loc})'].value:
            pygame.draw.circle(screen,
                            RED,
                            [(margin + grid_size) * loc[0] + margin + grid_size // 2,
                                (margin + grid_size) * loc[1] + margin + grid_size // 2],
                            20)

    # Draw Pac-Man
    pacman_loc = env.state.fluents['at(pacman)'].value
    pygame.draw.circle(screen,
                    YELLOW,
                    [(margin + grid_size) * pacman_loc[0] + margin + grid_size // 2,
                        (margin + grid_size) * pacman_loc[1] + margin + grid_size // 2],
                    25)

    pygame.display.update()
    time.sleep(0.5)
env.render = render