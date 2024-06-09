import golog
import time
from golog.envs.golog_env import GologState, GologAction
import gymnasium as gym
import pygame



# Define the Pac-Man environment
initial_state = GologState()
initial_state.add_symbol('location', [(x, y) for x in range(3) for y in range(3)])
initial_state.add_symbol('dot', [(x, y) for x in range(3) for y in range(3)])
initial_state.add_symbol('capsule', [(x, y) for x in range(3) for y in range(3)])
initial_state.add_symbol('ghost', [(x, y) for x in range(3) for y in range(3)])

initial_state.add_fluent('loc(pacman)', initial_state.symbols['location'], (0, 0))
initial_state.add_fluent('ghost_present((2, 2))', [True, False], True)
initial_state.add_fluent('pacman_powered_up()', [True, False], False)
initial_state.add_fluent('power_up_timer()', list(range(11)), 0)

for dot in initial_state.symbols['dot']:
    initial_state.add_fluent(f'dot_present({dot})', [True, False], dot in [(0, 1), (1, 2), (2, 0)])

for capsule in initial_state.symbols['capsule']:
    initial_state.add_fluent(f'capsule_present({capsule})', [True, False], capsule in [(1, 1)])

for ghost in initial_state.symbols['ghost']:
    if ghost != (2, 2):  # Avoid reinitializing (2, 2)
        initial_state.add_fluent(f'ghost_present({ghost})', [True, False], False)

# Helper function to check adjacency
def adjacent(from_loc, to_loc):
    fx, fy = from_loc
    tx, ty = to_loc
    return abs(fx - tx) + abs(fy - ty) == 1

# Define actions
def move_precondition(state, from_loc, to_loc):
    return state.fluents[f'loc(pacman)'].value == from_loc and adjacent(from_loc, to_loc)

def move_effect(state, from_loc, to_loc):
    state.fluents[f'loc(pacman)'].set_value(to_loc)
    if state.fluents['pacman_powered_up()'].value:
        state.fluents['power_up_timer()'].set_value(state.fluents['power_up_timer()'].value - 1)
        if state.fluents['power_up_timer()'].value == 0:
            state.fluents['pacman_powered_up()'].set_value(False)

def eat_dot_precondition(state, d):
    return state.fluents[f'loc(pacman)'].value == d and state.fluents[f'dot_present({d})'].value

def eat_dot_effect(state, d):
    state.fluents[f'dot_present({d})'].set_value(False)

def eat_capsule_precondition(state, c):
    return state.fluents[f'loc(pacman)'].value == c and state.fluents[f'capsule_present({c})'].value

def eat_capsule_effect(state, c):
    state.fluents[f'capsule_present({c})'].set_value(False)
    state.fluents['pacman_powered_up()'].set_value(True)
    state.fluents['power_up_timer()'].set_value(10)

def eat_ghost_precondition(state, g):
    return state.fluents['pacman_powered_up()'].value and state.fluents[f'loc(pacman)'].value == g and state.fluents[f'ghost_present({g})'].value

def eat_ghost_effect(state, g):
    state.fluents[f'ghost_present({g})'].set_value(False)

# Add actions to the environment
actions = [
    GologAction('move', move_precondition, move_effect, ['location', 'location']),
    GologAction('eat_dot', eat_dot_precondition, eat_dot_effect, ['dot']),
    GologAction('eat_capsule', eat_capsule_precondition, eat_capsule_effect, ['capsule']),
    GologAction('eat_ghost', eat_ghost_precondition, eat_ghost_effect, ['location']),
]

# Define goal
def pacman_goal(state):
        return all(not state.fluents[f'dot_present({d})'].value for d in initial_state.symbols['dot'])

# Define reward function
def pacman_reward(state):
    if pacman_goal(state):
        return 100
    if state.fluents['pacman_powered_up()'].value:
        return 5
    for d in initial_state.symbols['dot']:
        if state.fluents[f'dot_present({d})'].value and state.fluents[f'loc(pacman)'].value == d:
            return 10
    for c in initial_state.symbols['capsule']:
        if state.fluents[f'capsule_present({c})'].value and state.fluents[f'loc(pacman)'].value == c:
            return 20
    for g in initial_state.symbols['ghost']:
        if state.fluents['pacman_powered_up()'].value and state.fluents[f'loc(pacman)'].value == g and state.fluents[f'ghost_present({g})'].value:
            return 50
    return -1

env = gym.make('Golog-v0', initial_state=initial_state, goal_function=pacman_goal, reward_function=pacman_reward, actions=actions)

def render():
    pygame.init()
    screen = pygame.display.set_mode((300, 300))
    pygame.display.set_caption("Pac-Man")

    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    YELLOW = (255, 255, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
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
    for dot in initial_state.symbols['dot']:
        if env.state.fluents[f'dot_present({dot})'].value:
            pygame.draw.circle(screen,
                            BLUE,
                            [(margin + grid_size) * dot[0] + margin + grid_size // 2,
                                (margin + grid_size) * dot[1] + margin + grid_size // 2],
                            10)

    # Draw capsules
    for capsule in initial_state.symbols['capsule']:
        if env.state.fluents[f'capsule_present({capsule})'].value:
            pygame.draw.circle(screen,
                            GREEN,
                            [(margin + grid_size) * capsule[0] + margin + grid_size // 2,
                                (margin + grid_size) * capsule[1] + margin + grid_size // 2],
                            15)

    # Draw ghosts
    for ghost in initial_state.symbols['ghost']:
        if env.state.fluents[f'ghost_present({ghost})'].value:
            pygame.draw.circle(screen,
                            RED,
                            [(margin + grid_size) * ghost[0] + margin + grid_size // 2,
                                (margin + grid_size) * ghost[1] + margin + grid_size // 2],
                            20)

    # Draw Pac-Man
    pacman_loc = env.state.fluents['loc(pacman)'].value
    pygame.draw.circle(screen,
                    YELLOW,
                    [(margin + grid_size) * pacman_loc[0] + margin + grid_size // 2,
                        (margin + grid_size) * pacman_loc[1] + margin + grid_size // 2],
                    25)

    pygame.display.update()
    time.sleep(0.5)
env.render = render