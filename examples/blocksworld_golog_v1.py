import golog
from golog.envs.golog_env import GologState, GologAction
import gymnasium as gym

def stack_precondition(state, x, y):
    return x != y and x != 'table' and state.fluents[f'loc({x})'].value != y and not any(state.fluents[f'loc({z})'].value == x for z in state.symbols['block'])

def stack_effect(state, x, y):
    state.fluents[f'loc({x})'].set_value(y)

def blocksworld_goal(state):
    return state.fluents['loc(a)'].value == 'table' and state.fluents['loc(b)'].value == 'a' and state.fluents['loc(c)'].value == 'b'


def reward_function(state):
    #return partial reward based on the goal
    if blocksworld_goal(state):
        return 100
    elif state.fluents['loc(a)'].value == 'table' and state.fluents['loc(b)'].value != 'a' and state.fluents['loc(c)'].value != 'b':
        return 15
    elif state.fluents['loc(a)'].value == 'table' and state.fluents['loc(b)'].value == 'a' and state.fluents['loc(c)'].value != 'b':
        return 30
    elif state.fluents['loc(a)'].value != 'table' and state.fluents['loc(b)'].value == 'a' and state.fluents['loc(c)'].value == 'b':
        return 30
    else:
        return -100


initial_state = GologState()
initial_state.add_symbol('block', ['a', 'b', 'c'])
initial_state.add_symbol('location', ['a', 'b', 'c', 'table'])
initial_state.add_fluent('loc(a)', ['a', 'b', 'c', 'table'], 'c')
initial_state.add_fluent('loc(b)', ['a', 'b', 'c', 'table'], 'table')
initial_state.add_fluent('loc(c)', ['a', 'b', 'c', 'table'], 'b')

stack_action = GologAction('stack', stack_precondition, stack_effect, ['block', 'location'])
initial_state.add_action(stack_action)

actions = [stack_action]

env = gym.make('Golog-v1', initial_state=initial_state, goal_function=blocksworld_goal, actions=actions, reward_function=reward_function)