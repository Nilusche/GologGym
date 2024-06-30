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
    state.fluents['nearest_dot'].set_value(min(state.symbols['dot'], key=lambda x: np.linalg.norm(np.array(new_position) - np.array(state.fluents[f'pos({x})'].value))))

def eat_precondition(state, pacman, dot):
    return state.fluents['pos(pacman)'].value == state.fluents[f'pos({dot})'].value and not state.fluents[f'dot_eaten({dot})'].value

def eat_effect(state, pacman, dot):
    state.fluents[f'dot_eaten({dot})'].set_value(True)
    state.fluents['nearest_dot'].set_value(min(state.symbols['dot'], key=lambda x: np.linalg.norm(np.array(state.fluents['pos(pacman)'].value) - np.array(state.fluents[f'pos({x})'].value))))

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
    initial_state.add_fluent('pos(ghost)', [(i, j) for i in range(5) for j in range(5)], (2, 3))

    for i, dot in enumerate(dots):
        initial_state.add_fluent(f'pos({dot})', [(i, j) for i in range(5) for j in range(5)], (i % 5, (i + 2) % 5))
        initial_state.add_fluent(f'dot_eaten({dot})', [True, False], False)

    initial_state.add_fluent('nearest_dot', dots, dots[0])

    # Define actions
    move_action = GologAction('move', move_precondition, move_effect, ['pacman', 'direction'])
    eat_action = GologAction('eat', eat_precondition, eat_effect, ['pacman', 'dot'])

    initial_state.add_action(move_action)
    initial_state.add_action(eat_action)

    actions = [move_action, eat_action]

    return initial_state, actions

# Define the reward function
def reward_function(state):
    reward = -1

    #reward for reaching a dot that has not been eaten
    reward += 5* sum(state.fluents[f'pos(pacman)'].value == state.fluents[f'pos(dot{i})'].value and not state.fluents[f'dot_eaten(dot{i})'].value for i in range(1, len(state.symbols['dot']) + 1))
    #rewad for eating a dot
    reward += 10 * sum(state.fluents[f'dot_eaten({dot})'].value for dot in state.symbols['dot'])

    if pacman_goal(state):
        return 200
    
    return reward

# Create the environment
num_dots = 5  # Example number of dots
initial_state, actions = initialize_pacman_state(num_dots)
env = gym.make('Golog-v4', initial_state=initial_state, goal_function=pacman_goal, actions=actions, reward_function=reward_function, time_constraint=20)

#implement custom render function with pygame
import pygame 
import time

def render(): 
    pygame.init()
    screen = pygame.display.set_mode((300, 300))
    pygame.display.set_caption("Pac-Man")
    #fill screen with black
    screen.fill((0, 0, 0))
    #draw the grid
    grid_size = 5
    for i in range(grid_size):
        for j in range(grid_size):
            pygame.draw.rect(screen,(0, 0, 0) , (i * 60, j * 60, 60, 60))
            pygame.draw.rect(screen,(255, 255, 255) , (i * 60, j * 60, 60, 60), 2)
    #draw dots, pacman, ghost and walls
    for x in range(5):
        for y in range(5):
            loc = (x, y)
            
            #draw ghost as red circle
            if env.state.fluents['pos(ghost)'].value == loc:
                pygame.draw.circle(screen, (255, 0, 0), (x * 60 + 30, y * 60 + 30), 15)
            #draw dots as white circles
            if any(env.state.fluents[f'pos(dot{i})'].value == loc and not env.state.fluents[f'dot_eaten(dot{i})'].value for i in range(1, num_dots + 1)):
                pygame.draw.circle(screen, (255, 255, 255), (x * 60 + 30, y * 60 + 30), 5)
            #draw walls as white rectangles
            if loc in env.state.symbols['wall']:
                pygame.draw.rect(screen, (0, 255, 255), (x * 60, y * 60, 50, 50))
            #draw pacman as yellow circle
            if env.state.fluents['pos(pacman)'].value == loc:
                pygame.draw.circle(screen, (255, 255, 0), (x * 60 + 30, y * 60 + 30), 15)
    pygame.display.update()
    time.sleep(0.1)

env.render = render

